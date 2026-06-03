# PRAG — Parametric Retrieval-Augmented Generation (MedQA)

Research codebase for medical multiple-choice QA with textbook retrieval over the [MedQA](https://github.com/jind11/MedQA) dataset (Jin et al., 2020).

## Quick start

```powershell
pip install -r requirements.txt

# Download MedQA data (not in git) — see "Re-download data" below
python src/data/medqa_loader.py

# Build vector index (first run downloads the embedding model)
python src/knowledge/textbook_store.py
```

## Source code

| Module | Path | Purpose |
|--------|------|---------|
| Question loader | `src/data/medqa_loader.py` | US train/dev/test JSONL |
| Textbook store | `src/knowledge/textbook_store.py` | Chunk, embed, FAISS retrieve |

## Layout

| Path | Contents |
|------|----------|
| `MedQA/` | Official repo (IR baseline code) |
| `MedQA/data/data_clean/questions/` | QA splits: `US/`, `Mainland/`, `Taiwan/` |
| `MedQA/data/data_clean/textbooks/` | English + Chinese textbook corpora |

## Question data (JSONL)

Each line is one multiple-choice exam question, e.g. under `MedQA/data/data_clean/questions/US/`:

- `train.jsonl`, `dev.jsonl`, `test.jsonl` — official random split
- `US_qbank.jsonl` — full US bank
- `4_options/` — 4-option variants (US and Mainland)

## Textbooks

- `textbooks/en/` — 18 English medical textbooks (one `.txt` per book)
- `textbooks/zh_sentence/` and `zh_paragraph/` — Chinese corpus (sentence vs paragraph splits)

## Re-download data

The GitHub repo does not ship the data. Download from [Google Drive](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing):

```powershell
pip install gdown
cd MedQA
mkdir data
gdown "https://drive.google.com/uc?id=1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw" -O "data\medqa_data.zip"
Expand-Archive -Path "data\medqa_data.zip" -DestinationPath "data" -Force
```

## Citation

```bibtex
@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}
```
