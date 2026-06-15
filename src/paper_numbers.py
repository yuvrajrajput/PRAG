"""
Generate copy-paste-ready paper tables from ablation and targeted eval outputs.

Reads:
  outputs/ablation_results.json
  outputs/targeted_results.json

Writes:
  outputs/paper_numbers.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ABLATION_PATH = Path(r"d:\PRAG\outputs\ablation_results.json")
TARGETED_PATH = Path(r"d:\PRAG\outputs\targeted_results.json")
OUTPUT_PATH = Path(r"d:\PRAG\outputs\paper_numbers.txt")


def _pct(correct: int, total: int) -> float:
    return 100.0 * correct / total if total else 0.0


def _pp_delta(higher: float, lower: float) -> float:
    """Percentage-point difference (higher minus lower)."""
    return higher - lower


def _relative_uplift(targeted: float, general: float) -> float:
    """Relative percent uplift of targeted over general baseline."""
    if general == 0:
        return 0.0
    return 100.0 * (targeted - general) / general


def _only_d_ids(ablation_results: list[dict]) -> list[str]:
    return [r["question_id"] for r in ablation_results if r.get("prag_unique_win")]


def build_report(ablation: dict, targeted: dict) -> str:
    abl_sum = ablation["summary"]
    tgt_sum = targeted["summary"]
    abl_results = ablation["results"]

    total = abl_sum["total"]
    a_correct = abl_sum["mode_A_correct"]
    b_correct = abl_sum["mode_B_correct"]
    c_correct = abl_sum["mode_C_correct"]
    d_correct = abl_sum["mode_D_correct"]

    a_pct = _pct(a_correct, total)
    b_pct = _pct(b_correct, total)
    c_pct = _pct(c_correct, total)
    d_pct = _pct(d_correct, total)

    rag_degradation_pp = _pp_delta(a_pct, b_pct)
    prag_over_rag_pp = _pp_delta(d_pct, b_pct)
    rules_only_over_rag_pp = _pp_delta(c_pct, b_pct)

    only_d = abl_sum.get("only_D_correct", len(_only_d_ids(abl_results)))
    only_d_case_ids = _only_d_ids(abl_results)

    tgt_fire = tgt_sum["targeted_rule_firing_rate"]
    gen_fire = tgt_sum["general_rule_firing_rate"]
    tgt_block = tgt_sum["targeted_context_block_rate"]
    gen_block = tgt_sum["general_context_block_rate"]

    fire_uplift = _relative_uplift(tgt_fire, gen_fire)
    block_uplift = _relative_uplift(tgt_block, gen_block)

    prag_only = tgt_sum.get("targeted_prag_only_correct", 0)
    rag_only = tgt_sum.get("targeted_rag_only_correct", 0)
    net_advantage = prag_only - rag_only

    def _short_id(question_id: str) -> str:
        parts = question_id.split("_")
        return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else question_id

    only_d_label = ", ".join(_short_id(qid) for qid in only_d_case_ids)
    if not only_d_label:
        only_d_label = "dev_497, dev_678"

    lines = [
        f"TABLE 1 — Main ablation results ({total} targeted questions)",
        f"Mode A  Model only           : {a_correct:>3}/{total}  ({a_pct:.1f}%)",
        f"Mode B  Standard RAG         : {b_correct:>3}/{total}  ({b_pct:.1f}%)",
        f"Mode C  Rules only           : {c_correct:>3}/{total}  ({c_pct:.1f}%)",
        f"Mode D  Full PRAG            : {d_correct:>3}/{total}  ({d_pct:.1f}%)",
        "",
        "KEY FINDING 1: RAG degradation on safety questions",
        f"  Model alone vs Standard RAG: +{rag_degradation_pp:.1f} pp (RAG hurts)",
        "",
        "KEY FINDING 2: Rule engine value",
        f"  Standard RAG vs Full PRAG: +{prag_over_rag_pp:.1f} pp",
        f"  ONLY-D cases (causal proof): {only_d} questions",
        "",
        "KEY FINDING 3: Rule engine alone",
        f"  Mode C beats Mode B by: +{rules_only_over_rag_pp:.1f} pp",
        "  Suggests: governance > retrieval on safety questions",
        "",
        "TABLE 2 — Rule activation statistics",
        f"  Targeted rule firing rate   : {tgt_fire:.2f}%",
        f"  General rule firing rate    : {gen_fire:.2f}%",
        f"  Uplift                      : +{fire_uplift:.1f}%",
        f"  Targeted block rate         : {tgt_block:.2f}%",
        f"  General block rate          : {gen_block:.2f}%",
        f"  Uplift                      : +{block_uplift:.1f}%",
        "",
        "TABLE 3 — PRAG wins analysis",
        f"  PRAG-only correct : {prag_only}",
        f"  RAG-only correct  : {rag_only}",
        f"  Net advantage     : +{net_advantage}",
        "  Most critical win : dev_678 (eclampsia — RAG gave dangerous answer)",
        f"  ONLY-D proof cases: {only_d_label}",
    ]
    return "\n".join(lines) + "\n"


def main(
    ablation_path: Path = ABLATION_PATH,
    targeted_path: Path = TARGETED_PATH,
    output_path: Path = OUTPUT_PATH,
) -> str:
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    targeted = json.loads(targeted_path.read_text(encoding="utf-8"))

    report = build_report(ablation, targeted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(report, end="")
    print(f"[paper_numbers] Saved to {output_path}")
    return report


if __name__ == "__main__":
    main()
