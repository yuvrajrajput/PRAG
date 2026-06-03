"""
MedQA US split dataloader for the PRAG research project.

Loads JSONL question files from the official MedQA data_clean release.
Uses only the Python standard library (json, pathlib, random).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Literal

SplitName = Literal["train", "dev", "test"]

DEFAULT_DATA_ROOT = Path(r"d:\PRAG\MedQA\data\data_clean")
DEFAULT_QUESTIONS_DIR = DEFAULT_DATA_ROOT / "questions" / "US"
OPTION_ORDER = ("A", "B", "C", "D", "E")


def _normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a clean record dict with normalized options."""
    options = raw.get("options", {})
    if isinstance(options, str):
        options = json.loads(options.replace("'", '"'))
    if not isinstance(options, dict):
        raise ValueError(f"Expected options dict, got {type(options).__name__}")

    return {
        "question": str(raw["question"]).strip(),
        "answer": str(raw["answer"]).strip(),
        "options": {str(k): str(v).strip() for k, v in options.items()},
        "meta_info": str(raw.get("meta_info", "")).strip(),
        "answer_idx": str(raw.get("answer_idx", "")).strip().upper(),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
            records.append(_normalize_record(raw))
    return records


def _print_stats(split: str, records: list[dict[str, Any]]) -> None:
    meta_counts = Counter(r["meta_info"] for r in records)
    print(f"[MedQA] Loaded split={split!r}: {len(records)} records")
    print("[MedQA] meta_info distribution:")
    for meta, count in sorted(meta_counts.items(), key=lambda item: (-item[1], item[0])):
        pct = 100.0 * count / len(records) if records else 0.0
        print(f"  {meta!r}: {count} ({pct:.1f}%)")


class MedQADataset:
    """US MedQA multiple-choice exam questions."""

    def __init__(
        self,
        questions_dir: Path | str | None = None,
        split: SplitName | None = None,
    ) -> None:
        self.questions_dir = Path(questions_dir or DEFAULT_QUESTIONS_DIR)
        self.split: SplitName | None = None
        self.records: list[dict[str, Any]] = []

        if split is not None:
            self.load_split(split)

    def load_split(self, split: SplitName = "train") -> list[dict[str, Any]]:
        """Load train, dev, or test JSONL and print dataset statistics."""
        if split not in ("train", "dev", "test"):
            raise ValueError(f"split must be 'train', 'dev', or 'test', got {split!r}")

        path = self.questions_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"MedQA split file not found: {path}")

        self.split = split
        self.records = _load_jsonl(path)
        _print_stats(split, self.records)
        return self.records

    @staticmethod
    def get_question_text(record: dict[str, Any]) -> str:
        """Format the stem and all answer options as a single prompt string."""
        lines = [record["question"], ""]
        for key in OPTION_ORDER:
            if key in record["options"]:
                lines.append(f"{key}. {record['options'][key]}")
        return "\n".join(lines)

    @staticmethod
    def get_correct_answer(record: dict[str, Any]) -> dict[str, str]:
        """Return correct answer letter and text."""
        answer_idx = record["answer_idx"]
        answer_text = record["answer"]
        if answer_idx in record["options"]:
            answer_text = record["options"][answer_idx]
        return {"answer_idx": answer_idx, "answer": answer_text}

    def filter_by_meta(self, meta_info: str) -> MedQADataset:
        """Return a new dataset containing only records with the given meta_info."""
        filtered = MedQADataset(questions_dir=self.questions_dir)
        filtered.split = self.split
        filtered.records = [
            record for record in self.records if record["meta_info"] == meta_info
        ]
        return filtered

    def sample(self, n: int, seed: int | None = None) -> list[dict[str, Any]]:
        """Return n random records (without replacement)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return []
        if n > len(self.records):
            raise ValueError(
                f"Requested {n} samples but only {len(self.records)} records loaded"
            )
        rng = random.Random(seed)
        return rng.sample(self.records, n)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


def _record_for_export(record: dict[str, Any]) -> dict[str, Any]:
    """Build an inspection-friendly dict for JSON export."""
    correct = MedQADataset.get_correct_answer(record)
    return {
        **record,
        "question_text": MedQADataset.get_question_text(record),
        "correct_answer_idx": correct["answer_idx"],
        "correct_answer_text": correct["answer"],
    }


def _main() -> None:
    output_path = Path(r"d:\PRAG\outputs\sample_questions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = MedQADataset()
    dataset.load_split("train")

    samples = [_record_for_export(r) for r in dataset.sample(5, seed=42)]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"[MedQA] Wrote {len(samples)} sample records to {output_path}")


if __name__ == "__main__":
    _main()
