"""Print PRAG-only wins from targeted_results.json in terminal report format."""
import json
from pathlib import Path

LETTERS = "ABCDEFGHI"


def fmt_option(idx, options, fallback: str) -> str:
    if idx is None:
        return "?"
    text = options.get(str(idx), options.get(idx, fallback))
    return f"{LETTERS[idx]} — {text}"


def main() -> None:
    data = json.loads(
        Path("outputs/targeted_results.json").read_text(encoding="utf-8")
    )
    wins = [
        x
        for x in data["results"]
        if x["prag_correct"] and not x["standard_rag_correct"]
    ]

    for i, x in enumerate(wins, 1):
        print("=" * 72)
        print(f" Targeted question PRAG-only win {i}/{len(wins)} — {x['question_id']}")
        print("=" * 72)
        print("Keywords matched:", ", ".join(x["matched_keywords"]))
        print()
        print("Question:")
        print(x["question"])
        print()
        print("Rules fired:")
        rules_detail = x.get("rules_detail") or []
        if rules_detail:
            for rd in rules_detail:
                rid = rd.get("rule_id", "?")
                action = rd.get("action", "?")
                msg = rd.get("message", "")
                print(f"  - {rid} [{action}]: {msg}")
        elif x.get("rules_fired"):
            print(f"  ({x['rules_fired']} rules fired — detail not stored)")
        else:
            print("  (none)")
        print()
        print("Blocked context:")
        blocked = x.get("blocked_chunks") or []
        if blocked:
            for bc in blocked[:3]:
                cid = bc.get("chunk_id", "?")
                src = bc.get("source", bc.get("source_book", "?"))
                txt = (bc.get("text") or "")[:250].replace("\n", " ")
                print(f"  - {cid} ({src})")
                print(f"    {txt}...")
            if len(blocked) > 3:
                print(f"  ... and {len(blocked) - 3} more blocked chunk(s)")
        else:
            print("  (none)")
        print()
        opts = x.get("options", {})
        print("Answers:")
        print(f"  PRAG         : {fmt_option(x['prag_answer_idx'], opts, x['prag_answer'])}")
        print(
            f"  Standard RAG : {fmt_option(x['standard_rag_answer_idx'], opts, x['standard_rag_answer'])}"
        )
        print(
            f"  Correct      : {fmt_option(x['correct_answer_idx'], opts, x['correct_answer'])}"
        )
        print()


if __name__ == "__main__":
    main()
