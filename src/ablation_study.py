"""
PRAG ablation study — Table 2 experiment for the research paper.

Runs four inference modes on the same 170 targeted safety questions to isolate
contributions of the base model, retrieval, rules, and full PRAG:

  Mode A — model only (no retrieval, no rules)
  Mode B — retrieval only (standard RAG, no rule filtering)
  Mode C — rules only (no retrieval; rule trace as context)
  Mode D — full PRAG (retrieval + Paninian rule filtering)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.medqa_loader import MedQADataset
from src.prag_pipeline import (
    DEFAULT_ANSWERER_MODEL,
    MCQAnswerer,
    PRAGPipeline,
    _format_options,
    extract_patient_context,
    make_question_id,
)
from src.targeted_eval import find_matched_keywords

OUTPUT_PATH = Path(r"d:\PRAG\outputs\ablation_results.json")
DEFAULT_SEED = 42


def _format_options_line(options: dict[str, str]) -> str:
    return _format_options(options)


def _answer_text(options: dict[str, str], letter: str) -> str:
    return options.get(letter, letter)


def _is_correct(letter: str, correct_idx: str) -> bool:
    return letter == correct_idx


def build_rule_trace_context(trace: list[dict[str, Any]]) -> str:
    """Summarize fired Paninian rules for Mode C (rules-only, no retrieval)."""
    if not trace:
        return "(No clinical safety rules fired.)"

    lines = ["Paninian clinical safety rules (no textbook retrieval):"]
    seen: set[str] = set()
    for entry in trace:
        rule_id = entry.get("rule_id", "?")
        if rule_id in seen:
            continue
        seen.add(rule_id)
        action = entry.get("action", "?")
        message = entry.get("message", "")
        principle = entry.get("principle", "")
        line = f"- {rule_id} [{action}]: {message}"
        if principle:
            line += f" (principle: {principle})"
        lines.append(line)
    return "\n".join(lines)


def build_mode_a_prompt(question: str, options: dict[str, str]) -> str:
    """Model-only prompt: no retrieval, no rules."""
    return (
        f"Answer this medical question: {question}\n"
        f"Options: {_format_options_line(options)}\n"
        "Answer with only the letter of the correct option."
    )


def run_ablation_question(
    pipeline: PRAGPipeline,
    record: dict[str, Any],
    question_id: str,
) -> dict[str, Any]:
    """Run all four ablation modes on one targeted MedQA record."""
    question = record["question"]
    options = record["options"]
    correct = MedQADataset.get_correct_answer(record)
    correct_idx = correct["answer_idx"]
    correct_text = correct["answer"]

    patient = extract_patient_context(question)
    pipeline.rule_engine.set_patient_context(patient)
    answerer = pipeline.answerer

    # Mode A — model only
    mode_a_prompt = build_mode_a_prompt(question, options)
    mode_a_letter = answerer.select_from_prompt(mode_a_prompt, options)

    # Mode B — retrieval only (standard RAG)
    retrieved = pipeline._retrieve(question)
    retrieved_context = pipeline._join_chunks(retrieved)
    mode_b_letter = answerer.select(question, options, retrieved_context)

    # Mode C — rules only (empty retrieval context)
    rule_result_empty = pipeline.rule_engine.evaluate(
        query=question,
        retrieved_context="",
        patient_context=patient,
    )
    rules_only_context = build_rule_trace_context(rule_result_empty.trace)
    mode_c_letter = answerer.select(question, options, rules_only_context)

    # Mode D — full PRAG (retrieval + rule filtering)
    approved, rule_result = pipeline._filter_rule_approved_chunks(
        question, retrieved, patient
    )
    approved_context = pipeline._join_chunks(approved)
    mode_d_letter = answerer.select(question, options, approved_context)

    mode_a_correct = _is_correct(mode_a_letter, correct_idx)
    mode_b_correct = _is_correct(mode_b_letter, correct_idx)
    mode_c_correct = _is_correct(mode_c_letter, correct_idx)
    mode_d_correct = _is_correct(mode_d_letter, correct_idx)

    chunks_blocked = len(retrieved) - len(approved)

    return {
        "question_id": question_id,
        "question": question[:100],
        "correct_answer": correct_text,
        "mode_A_answer": _answer_text(options, mode_a_letter),
        "mode_A_correct": mode_a_correct,
        "mode_B_answer": _answer_text(options, mode_b_letter),
        "mode_B_correct": mode_b_correct,
        "mode_C_answer": _answer_text(options, mode_c_letter),
        "mode_C_correct": mode_c_correct,
        "mode_D_answer": _answer_text(options, mode_d_letter),
        "mode_D_correct": mode_d_correct,
        "rules_fired": len(rule_result.trace),
        "chunks_blocked": chunks_blocked,
        "prag_unique_win": mode_d_correct and not mode_a_correct and not mode_b_correct and not mode_c_correct,
        "retrieval_helps": mode_b_correct and not mode_a_correct,
        "rules_help": mode_d_correct and not mode_b_correct,
    }


def collect_targeted_records(
    split: str = "dev",
) -> list[tuple[int, dict[str, Any], list[str]]]:
    """Return all keyword-matched records in stable dataset order."""
    dataset = MedQADataset()
    dataset.load_split(split)  # type: ignore[arg-type]

    targeted: list[tuple[int, dict[str, Any], list[str]]] = []
    for idx, record in enumerate(dataset.records):
        keywords = find_matched_keywords(record["question"])
        if keywords:
            targeted.append((idx, record, keywords))
    return targeted


def compute_summary(results: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "seed": seed,
            "mode_A_correct": 0,
            "mode_B_correct": 0,
            "mode_C_correct": 0,
            "mode_D_correct": 0,
            "mode_A_accuracy": 0.0,
            "mode_B_accuracy": 0.0,
            "mode_C_accuracy": 0.0,
            "mode_D_accuracy": 0.0,
            "only_D_correct": 0,
            "A_and_D_both_correct": 0,
            "B_and_D_both_correct": 0,
            "retrieval_helps_count": 0,
            "rules_help_count": 0,
            "rule_hierarchy_contribution_pct": 0.0,
        }

    mode_a = sum(1 for r in results if r["mode_A_correct"])
    mode_b = sum(1 for r in results if r["mode_B_correct"])
    mode_c = sum(1 for r in results if r["mode_C_correct"])
    mode_d = sum(1 for r in results if r["mode_D_correct"])

    only_d = sum(1 for r in results if r["prag_unique_win"])
    a_and_d = sum(1 for r in results if r["mode_A_correct"] and r["mode_D_correct"])
    b_and_d = sum(1 for r in results if r["mode_B_correct"] and r["mode_D_correct"])
    retrieval_helps = sum(1 for r in results if r["retrieval_helps"])
    rules_help = sum(1 for r in results if r["rules_help"])

    return {
        "total": total,
        "seed": seed,
        "mode_A_correct": mode_a,
        "mode_B_correct": mode_b,
        "mode_C_correct": mode_c,
        "mode_D_correct": mode_d,
        "mode_A_accuracy": round(100.0 * mode_a / total, 2),
        "mode_B_accuracy": round(100.0 * mode_b / total, 2),
        "mode_C_accuracy": round(100.0 * mode_c / total, 2),
        "mode_D_accuracy": round(100.0 * mode_d / total, 2),
        "only_D_correct": only_d,
        "A_and_D_both_correct": a_and_d,
        "B_and_D_both_correct": b_and_d,
        "retrieval_helps_count": retrieval_helps,
        "rules_help_count": rules_help,
        "rule_hierarchy_contribution_pct": round(100.0 * rules_help / total, 2),
    }


def print_summary_table(summary: dict[str, Any]) -> None:
    total = summary["total"]

    def pct(n: int) -> float:
        return round(100.0 * n / total, 1) if total else 0.0

    print("\n" + "=" * 48)
    print(f"PRAG ABLATION STUDY — {total} targeted questions")
    print("=" * 48)
    print(
        f"Mode A (model only)         : {summary['mode_A_correct']:>3}/{total}  "
        f"({pct(summary['mode_A_correct']):.1f}%)"
    )
    print(
        f"Mode B (RAG, no rules)      : {summary['mode_B_correct']:>3}/{total}  "
        f"({pct(summary['mode_B_correct']):.1f}%)"
    )
    print(
        f"Mode C (rules, no retrieval): {summary['mode_C_correct']:>3}/{total}  "
        f"({pct(summary['mode_C_correct']):.1f}%)"
    )
    print(
        f"Mode D (full PRAG)          : {summary['mode_D_correct']:>3}/{total}  "
        f"({pct(summary['mode_D_correct']):.1f}%)"
    )
    print("-" * 48)
    print(f"Questions where ONLY D correct : {summary['only_D_correct']:>3}  <- KEY NUMBER")
    print(f"Questions where A=D (model knew): {summary['A_and_D_both_correct']:>3}")
    print(f"Questions where B=D (rules add nothing): {summary['B_and_D_both_correct']:>3}")
    print("-" * 48)
    print(
        f"Rule hierarchy contribution    : {summary['rule_hierarchy_contribution_pct']:.1f}%  "
        "<- PAPER CLAIM"
    )
    print("=" * 48)


def run_ablation_study(
    split: str = "dev",
    seed: int = DEFAULT_SEED,
    model: str = DEFAULT_ANSWERER_MODEL,
    output_path: Path | str = OUTPUT_PATH,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the four-mode ablation on all targeted safety questions."""
    if model != "flan-t5":
        raise ValueError("Ablation study requires flan-t5 answerer (Mode A custom prompt).")

    random.seed(seed)

    targeted = collect_targeted_records(split)
    if limit is not None:
        targeted = targeted[:limit]

    print(
        f"\n[Ablation] {len(targeted)} targeted questions ({split}, seed={seed}, model={model})"
    )

    pipeline = PRAGPipeline(model=model)
    results: list[dict[str, Any]] = []

    for run_idx, (record_idx, record, keywords) in enumerate(targeted, start=1):
        qid = make_question_id(record, split, record_idx)
        result = run_ablation_question(pipeline, record, qid)
        result["matched_keywords"] = keywords
        results.append(result)

        def _mark(correct: bool) -> str:
            return "+" if correct else "-"

        flags = []
        if result["prag_unique_win"]:
            flags.append("ONLY-D")
        if result["retrieval_helps"]:
            flags.append("B>A")
        if result["rules_help"]:
            flags.append("D>B")
        flag_str = f"  [{' '.join(flags)}]" if flags else ""
        print(
            f"  [{run_idx}/{len(targeted)}] {qid}  "
            f"A{_mark(result['mode_A_correct'])} "
            f"B{_mark(result['mode_B_correct'])} "
            f"C{_mark(result['mode_C_correct'])} "
            f"D{_mark(result['mode_D_correct'])}{flag_str}"
        )

    summary = compute_summary(results, seed=seed)
    summary["split"] = split
    summary["model"] = model
    summary["answerer"] = pipeline.answerer._hf_model_name
    summary["retriever_type"] = type(pipeline.retriever).__name__

    payload = {
        "summary": summary,
        "modes": {
            "A": "model only (no retrieval, no rules)",
            "B": "retrieval only (standard RAG, top-5 chunks)",
            "C": "rules only (empty retrieval, rule trace as context)",
            "D": "full PRAG (retrieval + Paninian rule filtering)",
        },
        "results": results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print_summary_table(summary)
    print(f"\n[Ablation] Saved results to {output_path}")
    return payload


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="PRAG ablation study (Modes A–D) on targeted safety questions",
    )
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--model",
        default=DEFAULT_ANSWERER_MODEL,
        choices=["flan-t5"],
        help="Answerer backend (ablation requires flan-t5)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Path to save ablation_results.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of targeted questions (for smoke tests)",
    )
    args = parser.parse_args()
    run_ablation_study(
        split=args.split,
        seed=args.seed,
        model=args.model,
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    _main()
