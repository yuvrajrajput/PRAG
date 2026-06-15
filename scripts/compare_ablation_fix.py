"""Compare post-fix ablation vs pre-fix targeted eval."""
import json
from pathlib import Path

abl = json.loads(Path("outputs/ablation_results.json").read_text(encoding="utf-8"))
tgt = json.loads(Path("outputs/targeted_results.json").read_text(encoding="utf-8"))

abl_by_id = {r["question_id"]: r for r in abl["results"]}
tgt_by_id = {r["question_id"]: r for r in tgt["results"]}

preg_kw = ["pregnancy", "pregnant", "gestation"]
preg_qs = [
    r for r in abl["results"]
    if any(k in r.get("matched_keywords", []) for k in preg_kw)
]

print("=== ABLATION SUMMARY (post pregnancy fix) ===")
for k, v in abl["summary"].items():
    if k.startswith("mode_") or k in {
        "only_D_correct",
        "A_and_D_both_correct",
        "B_and_D_both_correct",
        "rule_hierarchy_contribution_pct",
        "total",
    }:
        print(f"  {k}: {v}")

print("\n=== TARGETED EVAL (pre-fix baseline) ===")
prag = sum(1 for r in tgt["results"] if r["prag_correct"])
rag = sum(1 for r in tgt["results"] if r["standard_rag_correct"])
prag_only = sum(
    1 for r in tgt["results"]
    if r["prag_correct"] and not r["standard_rag_correct"]
)
print(f"  PRAG: {prag}/170, RAG: {rag}/170, PRAG-only wins: {prag_only}")

print("\n=== KEY PREGNANCY CASES ===")
for qid in [
    "dev_822_87011fe2cb",
    "dev_678_4be418bdc9",
    "dev_497_644d6175e3",
    "dev_401_6aa7d992e6",
]:
    r = abl_by_id.get(qid)
    t = tgt_by_id.get(qid)
    if not r:
        continue
    print(f"\n{qid}:")
    print(
        f"  POST  A={r['mode_A_correct']} B={r['mode_B_correct']} "
        f"C={r['mode_C_correct']} D={r['mode_D_correct']}"
    )
    print(f"    D={r['mode_D_answer'][:55]}")
    print(f"    GT={r['correct_answer'][:55]}")
    print(f"    rules_fired={r['rules_fired']} chunks_blocked={r['chunks_blocked']}")
    if t:
        print(
            f"  PRE   PRAG={t['prag_correct']} RAG={t['standard_rag_correct']} "
            f"rules={t['rules_fired']} blocked={t['blocked_chunk_count']}"
        )

print("\n=== PREGNANCY-KEYWORD Qs: rule firing changed ===")
changed = []
for r in preg_qs:
    t = tgt_by_id.get(r["question_id"])
    if not t:
        continue
    if t["rules_fired"] != r["rules_fired"] or t["blocked_chunk_count"] != r["chunks_blocked"]:
        changed.append(r)

print(f"  Count: {len(changed)} / {len(preg_qs)} pregnancy-keyword questions")
for r in changed:
    t = tgt_by_id[r["question_id"]]
    d_delta = r["mode_D_correct"] != t["prag_correct"]
    print(
        f"  {r['question_id']}: rules {t['rules_fired']}->{r['rules_fired']}, "
        f"blocked {t['blocked_chunk_count']}->{r['chunks_blocked']}, "
        f"D correct {r['mode_D_correct']} (was PRAG {t['prag_correct']})"
        + (" *** ANSWER CHANGED ***" if d_delta else "")
    )

print("\n=== ONLY-D wins (post-fix) ===")
for r in abl["results"]:
    if r["prag_unique_win"]:
        print(f"  {r['question_id']}")
        print(f"    correct: {r['correct_answer'][:60]}")
        print(f"    A={r['mode_A_answer'][:40]}")
        print(f"    B={r['mode_B_answer'][:40]}")
        print(f"    D={r['mode_D_answer'][:40]}")

print("\n=== D>B wins (rules help, post-fix) ===")
d_b = [r for r in abl["results"] if r["rules_help"]]
print(f"  Count: {len(d_b)}")
for r in d_b:
    print(f"  {r['question_id']}: B={r['mode_B_answer'][:35]} -> D={r['mode_D_answer'][:35]}")
