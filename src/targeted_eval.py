"""
Targeted PRAG evaluation on medically sensitive MedQA dev questions.

Filters dev questions by clinical risk keywords, runs full PRAG vs standard RAG,
and reports where Paninian rules fire and block retrieved context.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.medqa_loader import MedQADataset
from src.prag_pipeline import (
    DEFAULT_ANSWERER_MODEL,
    PRAGPipeline,
    extract_patient_context,
    make_question_id,
)

OUTPUT_PATH = Path(r"d:\PRAG\outputs\targeted_results.json")

KEYWORD_GROUPS: list[list[str]] = [
    ["renal failure", "kidney disease", "gfr"],
    ["pregnant", "pregnancy", "gestation"],
    ["nsaid", "ibuprofen", "aspirin"],
    ["warfarin", "anticoagulant"],
    ["elderly", "age 65", "age 70", "age 75"],
]


def find_matched_keywords(question: str) -> list[str]:
    """Return all keywords found in the question text (case-insensitive)."""
    text = question.lower()
    matched: list[str] = []
    for group in KEYWORD_GROUPS:
        for keyword in group:
            if keyword in text and keyword not in matched:
                matched.append(keyword)
    return matched


def is_targeted_question(question: str) -> bool:
    return bool(find_matched_keywords(question))


def _chunk_summary(chunk: dict[str, Any], *, preview_len: int = 300) -> dict[str, Any]:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "source_book": chunk.get("source_book"),
        "char_start": chunk.get("char_start"),
        "text_preview": chunk.get("text", "")[:preview_len],
    }


def _rule_summaries(trace: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    summaries: list[dict[str, str]] = []
    for entry in trace:
        rule_id = entry.get("rule_id", "")
        if rule_id in seen:
            continue
        seen.add(rule_id)
        summaries.append({
            "rule_id": rule_id,
            "message": entry.get("message", ""),
            "action": entry.get("action", ""),
            "principle": entry.get("principle", ""),
        })
    return summaries


def run_targeted_question(
    pipeline: PRAGPipeline,
    record: dict[str, Any],
    question_id: str,
    matched_keywords: list[str],
) -> dict[str, Any]:
    """Run one targeted question and capture blocked context details."""
    correct = MedQADataset.get_correct_answer(record)
    patient = extract_patient_context(record["question"])
    pipeline.rule_engine.set_patient_context(patient)

    retrieved = pipeline._retrieve(record["question"])
    approved, rule_result = pipeline._filter_rule_approved_chunks(
        record["question"], retrieved, patient
    )

    approved_ids = {chunk["chunk_id"] for chunk in approved}
    blocked_chunks = [
        _chunk_summary(chunk)
        for chunk in retrieved
        if chunk["chunk_id"] not in approved_ids
    ]

    approved_context = pipeline._join_chunks(approved)
    retrieved_context = pipeline._join_chunks(retrieved)

    prag_letter = pipeline.answerer.select(
        record["question"], record["options"], approved_context
    )
    standard_letter = pipeline.answerer.select(
        record["question"], record["options"], retrieved_context
    )

    return {
        "question_id": question_id,
        "question": record["question"],
        "matched_keywords": matched_keywords,
        "rules_fired": len(rule_result.trace),
        "rules_blocked": len(rule_result.blocked),
        "rules": _rule_summaries(rule_result.trace),
        "rule_trace": rule_result.trace,
        "blocked_chunks": blocked_chunks,
        "blocked_chunk_count": len(blocked_chunks),
        "retrieved_chunks": len(retrieved),
        "chunks_after_rules": len(approved),
        "prag_answer": record["options"].get(prag_letter, prag_letter),
        "prag_answer_idx": prag_letter,
        "standard_rag_answer": record["options"].get(standard_letter, standard_letter),
        "standard_rag_answer_idx": standard_letter,
        "correct_answer": correct["answer"],
        "correct_answer_idx": correct["answer_idx"],
        "prag_correct": prag_letter == correct["answer_idx"],
        "standard_rag_correct": standard_letter == correct["answer_idx"],
        "answers_differ": prag_letter != standard_letter,
        "requires_specialist": rule_result.requires_specialist,
        "paninian_principles_applied": sorted({
            entry.get("principle", "UtsargaApavada")
            for entry in rule_result.trace
        }),
    }


def measure_rule_stats(
    pipeline: PRAGPipeline,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Lightweight rule-only pass (no MCQ answering) for firing-rate comparison."""
    rules_fired_count = 0
    blocked_context_count = 0
    blocked_action_count = 0

    for record in records:
        patient = extract_patient_context(record["question"])
        pipeline.rule_engine.set_patient_context(patient)
        retrieved = pipeline._retrieve(record["question"])
        approved, rule_result = pipeline._filter_rule_approved_chunks(
            record["question"], retrieved, patient
        )
        if rule_result.trace:
            rules_fired_count += 1
        if len(approved) < len(retrieved):
            blocked_context_count += 1
        if rule_result.blocked:
            blocked_action_count += 1

    total = len(records)
    return {
        "total": total,
        "rules_fired_count": rules_fired_count,
        "blocked_context_count": blocked_context_count,
        "blocked_action_count": blocked_action_count,
        "rule_firing_rate": round(100.0 * rules_fired_count / total, 2) if total else 0.0,
        "context_block_rate": round(100.0 * blocked_context_count / total, 2) if total else 0.0,
    }


def print_question_report(result: dict[str, Any], index: int, total: int) -> None:
    print("\n" + "=" * 72)
    print(f" Targeted question {index}/{total} — {result['question_id']}")
    print("=" * 72)
    print(f"Keywords matched: {', '.join(result['matched_keywords'])}")
    print(f"\nQuestion:\n{result['question']}\n")

    if result["rules"]:
        print("Rules fired:")
        for rule in result["rules"]:
            print(f"  - {rule['rule_id']} [{rule['action']}]: {rule['message']}")
    else:
        print("Rules fired: (none)")

    if result["blocked_chunks"]:
        print("\nBlocked context:")
        for chunk in result["blocked_chunks"]:
            print(f"  - {chunk['chunk_id']} ({chunk['source_book']})")
            print(f"    {chunk['text_preview']}...")
    else:
        print("\nBlocked context: (none)")

    print("\nAnswers:")
    print(f"  PRAG         : {result['prag_answer_idx']} — {result['prag_answer']}")
    print(f"  Standard RAG : {result['standard_rag_answer_idx']} — {result['standard_rag_answer']}")
    print(f"  Correct      : {result['correct_answer_idx']} — {result['correct_answer']}")
    if result["answers_differ"]:
        print("  >> PRAG and RAG disagree on this sensitive question")


def print_summary(
    targeted_results: list[dict[str, Any]],
    targeted_stats: dict[str, Any],
    general_stats: dict[str, Any],
) -> None:
    matched = len(targeted_results)
    targeted_rules_fired = sum(1 for r in targeted_results if r["rules_fired"] > 0)
    targeted_blocked = sum(1 for r in targeted_results if r["blocked_chunk_count"] > 0)
    targeted_block_actions = sum(1 for r in targeted_results if r["rules_blocked"] > 0)
    prag_wins = sum(
        1 for r in targeted_results
        if r["prag_correct"] and not r["standard_rag_correct"]
    )
    rag_wins = sum(
        1 for r in targeted_results
        if r["standard_rag_correct"] and not r["prag_correct"]
    )
    disagreements = sum(1 for r in targeted_results if r["answers_differ"])

    print("\n" + "#" * 72)
    print(" TARGETED EVALUATION SUMMARY")
    print("#" * 72)
    print(f"  Questions matched (targeted)     : {matched}")
    print(f"  Questions with rules fired       : {targeted_rules_fired}")
    print(f"  Questions with blocked context   : {targeted_blocked}")
    print(f"  Questions with block actions     : {targeted_block_actions}")
    print(f"  PRAG vs RAG disagreements        : {disagreements}")
    print(f"  PRAG-only correct                : {prag_wins}")
    print(f"  RAG-only correct                 : {rag_wins}")
    print("-" * 72)
    print(f"  Targeted rule firing rate        : {targeted_stats['rule_firing_rate']:.2f}%")
    print(f"  General dev rule firing rate     : {general_stats['rule_firing_rate']:.2f}%")
    print(
        f"  Targeted context block rate      : {targeted_stats['context_block_rate']:.2f}%"
    )
    print(
        f"  General dev context block rate   : {general_stats['context_block_rate']:.2f}%"
    )
    print("#" * 72)


def run_targeted_eval(
    split: str = "dev",
    model: str = DEFAULT_ANSWERER_MODEL,
    output_path: Path | str = OUTPUT_PATH,
) -> dict[str, Any]:
    """Run targeted evaluation on all keyword-matching dev questions."""
    dataset = MedQADataset()
    dataset.load_split(split)  # type: ignore[arg-type]

    targeted_records: list[tuple[int, dict[str, Any], list[str]]] = []
    general_records: list[dict[str, Any]] = []

    for idx, record in enumerate(dataset.records):
        keywords = find_matched_keywords(record["question"])
        if keywords:
            targeted_records.append((idx, record, keywords))
        else:
            general_records.append(record)

    print(f"\n[TargetedEval] {len(targeted_records)} targeted / {len(general_records)} general "
          f"({split} split, model={model})")

    pipeline = PRAGPipeline(model=model)
    results: list[dict[str, Any]] = []

    for run_idx, (record_idx, record, keywords) in enumerate(targeted_records, start=1):
        qid = make_question_id(record, split, record_idx)
        result = run_targeted_question(pipeline, record, qid, keywords)
        results.append(result)
        print_question_report(result, run_idx, len(targeted_records))

    targeted_stats = measure_rule_stats(
        pipeline,
        [record for _, record, _ in targeted_records],
    )
    general_stats = measure_rule_stats(pipeline, general_records)

    summary = {
        "split": split,
        "model": model,
        "questions_matched": len(results),
        "questions_general": len(general_records),
        "targeted_rules_fired": sum(1 for r in results if r["rules_fired"] > 0),
        "targeted_blocked_context": sum(1 for r in results if r["blocked_chunk_count"] > 0),
        "targeted_block_actions": sum(1 for r in results if r["rules_blocked"] > 0),
        "targeted_prag_rag_disagreements": sum(1 for r in results if r["answers_differ"]),
        "targeted_prag_only_correct": sum(
            1 for r in results if r["prag_correct"] and not r["standard_rag_correct"]
        ),
        "targeted_rag_only_correct": sum(
            1 for r in results if r["standard_rag_correct"] and not r["prag_correct"]
        ),
        "targeted_rule_firing_rate": targeted_stats["rule_firing_rate"],
        "general_rule_firing_rate": general_stats["rule_firing_rate"],
        "targeted_context_block_rate": targeted_stats["context_block_rate"],
        "general_context_block_rate": general_stats["context_block_rate"],
        "keyword_groups": KEYWORD_GROUPS,
    }

    payload = {
        "summary": summary,
        "targeted_stats": targeted_stats,
        "general_stats": general_stats,
        "results": results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print_summary(results, targeted_stats, general_stats)
    print(f"\n[TargetedEval] Saved results to {output_path}")
    return payload


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Targeted PRAG evaluation on medically sensitive MedQA questions",
    )
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument(
        "--model",
        default=DEFAULT_ANSWERER_MODEL,
        choices=["flan-t5", "biomedbert"],
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Path to save targeted_results.json",
    )
    args = parser.parse_args()
    run_targeted_eval(split=args.split, model=args.model, output_path=args.output)


if __name__ == "__main__":
    _main()
