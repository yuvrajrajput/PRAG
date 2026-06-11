"""
PRAG end-to-end pipeline: MedQA questions + textbook retrieval + Paninian rules.

Flow:
  1. Load question
  2. Retrieve textbook chunks
  3. Evaluate Paninian rules on retrieved context
  4. Build prompt from rule-approved chunks only (PRAG) vs all chunks (standard RAG)
  5. Select MCQ answer via BiomedBERT embeddings or keyword overlap fallback
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.medqa_loader import MedQADataset, OPTION_ORDER
from src.knowledge.textbook_store import (
    DEFAULT_TEXTBOOK_DIR,
    DEFAULT_VECTOR_STORE,
    TextbookStore,
    _chunk_textbook,
)
from src.rules.paninian_rule_engine import PaniniRuleEngine

OUTPUT_DIR = Path(r"d:\PRAG\outputs")
BENCHMARK_PATH = OUTPUT_DIR / "benchmark_results.json"
BIOMED_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


# ---------------------------------------------------------------------------
# Patient context extraction (feeds Anuvrtti / antaranga rules)
# ---------------------------------------------------------------------------


def extract_patient_context(question: str) -> dict[str, Any]:
    """Heuristically extract vignette facts for the Paninian rule engine."""
    text = question.lower()
    context: dict[str, Any] = {
        "conditions": [],
        "medications": [],
        "allergies": [],
    }

    age_match = re.search(r"\b(\d{1,3})[- ]year[- ]old\b", text)
    if age_match:
        context["age"] = int(age_match.group(1))

    if any(term in text for term in ("pregnant", "pregnancy", "gestation", "gravid")):
        context["pregnant"] = True
        context["pregnancy"] = True

    gfr_match = re.search(r"\bgfr\s*(?:of\s*)?(\d{2,3})\b", text)
    if gfr_match:
        context["gfr"] = int(gfr_match.group(1))

    condition_terms = {
        "renal failure": ["renal failure", "kidney failure", "esrd", "dialysis"],
        "asthma": ["asthma", "status asthmaticus"],
        "liver disease": ["cirrhosis", "liver disease", "hepatic failure"],
        "hyperkalemia": ["hyperkalemia", "high potassium"],
    }
    for label, terms in condition_terms.items():
        if any(t in text for t in terms):
            context["conditions"].append(label)

    med_terms = [
        "warfarin", "metformin", "lisinopril", "ibuprofen", "aspirin",
        "insulin", "heparin", "lithium", "fluoxetine",
    ]
    context["medications"] = [m for m in med_terms if m in text]

    allergy_terms = ["penicillin allergy", "allergy to penicillin", "pcn allergy"]
    if any(t in text for t in allergy_terms):
        context["allergies"].append("penicillin")

    return context


def make_question_id(record: dict[str, Any], split: str, index: int) -> str:
    """Stable question id for benchmark rows."""
    digest = hashlib.md5(record["question"].encode("utf-8")).hexdigest()[:10]
    return f"{split}_{index}_{digest}"


# ---------------------------------------------------------------------------
# Retrieval (FAISS or keyword fallback)
# ---------------------------------------------------------------------------


class KeywordTextbookRetriever:
    """Word-overlap retriever used when the FAISS index is not yet built."""

    def __init__(self, textbook_dir: Path | str | None = None) -> None:
        textbook_dir = Path(textbook_dir or DEFAULT_TEXTBOOK_DIR)
        self.chunks: list[dict[str, Any]] = []
        for book_path in sorted(textbook_dir.glob("*.txt")):
            text = book_path.read_text(encoding="utf-8", errors="replace")
            self.chunks.extend(_chunk_textbook(text, book_path.stem))
        print(f"[KeywordRetriever] Indexed {len(self.chunks)} chunks (fallback mode)")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_terms:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in self.chunks:
            chunk_terms = set(re.findall(r"[a-z0-9]+", chunk["text"].lower()))
            overlap = len(query_terms & chunk_terms)
            if overlap == 0:
                continue
            score = overlap / (len(query_terms) ** 0.5)
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            hit = dict(chunk)
            hit["score"] = float(score)
            hit["rank"] = rank
            results.append(hit)
        return results


def load_retriever() -> TextbookStore | KeywordTextbookRetriever:
    """Load FAISS store if available, otherwise keyword fallback."""
    store_path = Path(DEFAULT_VECTOR_STORE)
    index_file = store_path / "index.faiss"
    if index_file.is_file():
        store = TextbookStore()
        store.load(store_path)
        return store
    print("[PRAG] Vector store not found — using keyword retriever fallback.")
    print(f"[PRAG] Build index with: python src/knowledge/textbook_store.py")
    return KeywordTextbookRetriever()


# ---------------------------------------------------------------------------
# Answer selection (BiomedBERT or keyword overlap)
# ---------------------------------------------------------------------------


class MCQAnswerer:
    """Select A–E using BiomedBERT cosine similarity, or keyword overlap on CPU."""

    def __init__(self, model_name: str = BIOMED_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._backend = "keyword"

    def _try_load_biomedbert(self) -> bool:
        if self._model is not None:
            return self._backend == "biomedbert"
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._backend = "biomedbert"
            print(f"[MCQAnswerer] Loaded {self.model_name}")
            return True
        except Exception as exc:
            print(f"[MCQAnswerer] BiomedBERT unavailable ({exc}); using keyword matching.")
            self._backend = "keyword"
            return False

    def _embed_text(self, text: str) -> Any:
        import torch

        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self._model(**tokens)
        return outputs.last_hidden_state[:, 0, :].squeeze(0)

    @staticmethod
    def _keyword_score(context: str, option_text: str) -> float:
        ctx_terms = set(re.findall(r"[a-z0-9]+", context.lower()))
        opt_terms = set(re.findall(r"[a-z0-9]+", option_text.lower()))
        if not opt_terms:
            return 0.0
        return len(ctx_terms & opt_terms) / len(opt_terms)

    def select(
        self,
        question: str,
        options: dict[str, str],
        context: str,
    ) -> str:
        """Return the selected option letter (A–E)."""
        if self._try_load_biomedbert() and context.strip():
            import torch

            q_emb = self._embed_text(question)
            ctx_emb = self._embed_text(context[:2000])
            query_emb = torch.nn.functional.normalize(q_emb + ctx_emb, dim=0)

            best_letter = "A"
            best_score = float("-inf")
            for letter in OPTION_ORDER:
                if letter not in options:
                    continue
                opt_emb = self._embed_text(options[letter])
                opt_emb = torch.nn.functional.normalize(opt_emb, dim=0)
                score = float(torch.dot(query_emb, opt_emb))
                if score > best_score:
                    best_score = score
                    best_letter = letter
            return best_letter

        best_letter = "A"
        best_score = float("-inf")
        combined = f"{question}\n{context}"
        for letter in OPTION_ORDER:
            if letter not in options:
                continue
            score = self._keyword_score(combined, options[letter])
            if score > best_score:
                best_score = score
                best_letter = letter
        return best_letter


# ---------------------------------------------------------------------------
# PRAG pipeline
# ---------------------------------------------------------------------------


class PRAGPipeline:
    """Connects MedQA loading, retrieval, Paninian rules, and MCQ answering."""

    def __init__(
        self,
        retriever: TextbookStore | KeywordTextbookRetriever | None = None,
        rule_engine: PaniniRuleEngine | None = None,
        answerer: MCQAnswerer | None = None,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever or load_retriever()
        self.rule_engine = rule_engine or PaniniRuleEngine()
        self.answerer = answerer or MCQAnswerer()
        self.top_k = top_k

    def _retrieve(self, query: str) -> list[dict[str, Any]]:
        return self.retriever.retrieve(query, top_k=self.top_k)

    @staticmethod
    def _join_chunks(chunks: list[dict[str, Any]]) -> str:
        parts = []
        for chunk in chunks:
            source = chunk.get("source_book", "unknown")
            parts.append(f"[{source}]\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def _filter_rule_approved_chunks(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        patient_context: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], Any]:
        """Keep chunks that pass per-chunk Paninian evaluation (no block action)."""
        approved: list[dict[str, Any]] = []
        combined_trace: list[dict[str, Any]] = []

        for chunk in chunks:
            result = self.rule_engine.evaluate(
                query=query,
                retrieved_context=chunk["text"],
                patient_context=patient_context,
            )
            combined_trace.extend(result.trace)
            if result.allowed:
                approved.append(chunk)

        overall_context = self._join_chunks(chunks)
        overall_result = self.rule_engine.evaluate(
            query=query,
            retrieved_context=overall_context,
            patient_context=patient_context,
        )
        for entry in overall_result.trace:
            if entry["rule_id"] not in {t["rule_id"] for t in combined_trace}:
                combined_trace.append(entry)

        return approved, overall_result

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        if not context.strip():
            context = "(No rule-approved context available.)"
        return (
            "You are a medical exam tutor. Use ONLY the provided textbook context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Select the single best answer (A–E)."
        )

    def run_question(
        self,
        record: dict[str, Any],
        question_id: str,
        *,
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run full PRAG + standard RAG on one MedQA record."""
        question_text = MedQADataset.get_question_text(record)
        correct = MedQADataset.get_correct_answer(record)
        patient = patient_context or extract_patient_context(record["question"])
        self.rule_engine.set_patient_context(patient)

        retrieved = self._retrieve(record["question"])
        retrieved_context = self._join_chunks(retrieved)

        approved_chunks, rule_result = self._filter_rule_approved_chunks(
            record["question"], retrieved, patient
        )
        approved_context = self._join_chunks(approved_chunks)

        prag_prompt = self._build_prompt(question_text, approved_context)
        standard_prompt = self._build_prompt(question_text, retrieved_context)

        prag_letter = self.answerer.select(
            record["question"], record["options"], approved_context
        )
        standard_letter = self.answerer.select(
            record["question"], record["options"], retrieved_context
        )

        prag_answer = record["options"].get(prag_letter, prag_letter)
        standard_answer = record["options"].get(standard_letter, standard_letter)

        principles = sorted({
            entry.get("principle", "UtsargaApavada")
            for entry in rule_result.trace
        })

        return {
            "question_id": question_id,
            "question": record["question"],
            "correct_answer": correct["answer"],
            "correct_answer_idx": correct["answer_idx"],
            "retrieved_chunks": len(retrieved),
            "chunks_after_rules": len(approved_chunks),
            "rules_fired": len(rule_result.trace),
            "rules_blocked": len(rule_result.blocked),
            "rule_trace": rule_result.trace,
            "prag_answer": prag_answer,
            "prag_answer_idx": prag_letter,
            "standard_rag_answer": standard_answer,
            "standard_rag_answer_idx": standard_letter,
            "prag_correct": prag_letter == correct["answer_idx"],
            "standard_rag_correct": standard_letter == correct["answer_idx"],
            "paninian_principles_applied": principles,
            "prag_prompt_preview": prag_prompt[:500],
            "standard_prompt_preview": standard_prompt[:500],
            "rule_allowed": rule_result.allowed,
            "requires_specialist": rule_result.requires_specialist,
        }


def run_comparison(
    n: int = 50,
    split: str = "dev",
    seed: int = 42,
    output_path: Path | str | None = BENCHMARK_PATH,
) -> dict[str, Any]:
    """
    Benchmark PRAG vs standard RAG on n MedQA questions.

    Saves per-question results and prints an accuracy summary table.
    """
    dataset = MedQADataset()
    dataset.load_split(split)  # type: ignore[arg-type]
    samples = dataset.sample(n, seed=seed)

    pipeline = PRAGPipeline()
    results: list[dict[str, Any]] = []

    print(f"\n[PRAG] Running comparison on {n} {split} questions...")
    for idx, record in enumerate(samples):
        qid = make_question_id(record, split, idx)
        result = pipeline.run_question(record, qid)
        results.append(result)
        status = "OK" if result["prag_correct"] else "MISS"
        print(
            f"  [{idx + 1}/{n}] {status}  "
            f"PRAG={result['prag_answer_idx']}  "
            f"RAG={result['standard_rag_answer_idx']}  "
            f"GT={result['correct_answer_idx']}  "
            f"rules={result['rules_fired']}"
        )

    prag_correct = sum(1 for r in results if r["prag_correct"])
    rag_correct = sum(1 for r in results if r["standard_rag_correct"])
    prag_acc = 100.0 * prag_correct / n if n else 0.0
    rag_acc = 100.0 * rag_correct / n if n else 0.0
    delta = prag_acc - rag_acc

    rules_fired_avg = sum(r["rules_fired"] for r in results) / n if n else 0.0
    chunks_filtered_avg = sum(
        r["retrieved_chunks"] - r["chunks_after_rules"] for r in results
    ) / n if n else 0.0

    summary = {
        "n": n,
        "split": split,
        "seed": seed,
        "prag_accuracy": round(prag_acc, 2),
        "standard_rag_accuracy": round(rag_acc, 2),
        "accuracy_delta": round(delta, 2),
        "prag_correct": prag_correct,
        "standard_rag_correct": rag_correct,
        "avg_rules_fired": round(rules_fired_avg, 2),
        "avg_chunks_filtered_by_rules": round(chunks_filtered_avg, 2),
        "answerer_backend": pipeline.answerer._backend,
        "retriever_type": type(pipeline.retriever).__name__,
    }

    payload = {"summary": summary, "results": results}
    output_path = Path(output_path or BENCHMARK_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    _print_summary_table(summary)
    print(f"\n[PRAG] Results saved to {output_path}")
    return payload


def _print_summary_table(summary: dict[str, Any]) -> None:
    width = 52
    print("\n" + "=" * width)
    print(" PRAG vs Standard RAG — Benchmark Summary")
    print("=" * width)
    print(f"  Questions evaluated : {summary['n']} ({summary['split']})")
    print(f"  Retriever           : {summary['retriever_type']}")
    print(f"  Answerer backend    : {summary['answerer_backend']}")
    print("-" * width)
    print(f"  Standard RAG accuracy: {summary['standard_rag_accuracy']:6.2f}%  "
          f"({summary['standard_rag_correct']}/{summary['n']})")
    print(f"  PRAG accuracy        : {summary['prag_accuracy']:6.2f}%  "
          f"({summary['prag_correct']}/{summary['n']})")
    print(f"  Delta (PRAG - RAG)   : {summary['accuracy_delta']:+6.2f}%")
    print("-" * width)
    print(f"  Avg rules fired      : {summary['avg_rules_fired']:.2f}")
    print(f"  Avg chunks filtered  : {summary['avg_chunks_filtered_by_rules']:.2f}")
    print("=" * width)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PRAG medical QA pipeline")
    parser.add_argument("--compare", type=int, default=0, help="Run n-question benchmark")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.compare > 0:
        run_comparison(n=args.compare, split=args.split, seed=args.seed)
        return

    dataset = MedQADataset()
    dataset.load_split("dev")
    record = dataset.records[0]
    pipeline = PRAGPipeline()
    result = pipeline.run_question(record, make_question_id(record, "dev", 0))

    print("\n[PRAG] Single-question demo")
    print(f"  Question ID    : {result['question_id']}")
    print(f"  Retrieved      : {result['retrieved_chunks']} chunks")
    print(f"  After rules    : {result['chunks_after_rules']} chunks")
    print(f"  Rules fired    : {result['rules_fired']}")
    print(f"  PRAG answer    : {result['prag_answer_idx']} — {result['prag_answer'][:60]}")
    print(f"  Standard RAG   : {result['standard_rag_answer_idx']}")
    print(f"  Correct        : {result['correct_answer_idx']}")
    print(f"  Principles     : {result['paninian_principles_applied']}")


if __name__ == "__main__":
    _main()
