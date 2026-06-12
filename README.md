# PRAG — Paninian Retrieval-Augmented Generation 

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Dataset: MedQA](https://img.shields.io/badge/dataset-MedQA-green.svg)](https://github.com/jind11/MedQA)

**PRAG** is a research codebase that combines **retrieval-augmented generation (RAG)** over medical textbooks with a **Paninian rule engine** inspired by classical Sanskrit grammar (utsarga-apavada, anuvrtti, paribhasha, nitya-anitya, antaranga-bahiranga). Every answer includes a full **auditable rule trace** — the main research contribution.

Built on the [MedQA](https://github.com/jind11/MedQA) USMLE-style multiple-choice dataset (Jin et al., 2020).

---

## Why PRAG?

| Approach | What it does |
|----------|----------------|
| **Standard RAG** | Retrieve textbook chunks → answer |
| **PRAG** | Retrieve → **apply Paninian clinical rules** → answer using **rule-approved context only** |

Rules govern drug contraindications, pregnancy safety, dosage limits, diagnostic red flags, and guideline conflicts — with explainable traces for every decision.

---

## Architecture

```
MedQA Question
      │
      ▼
TextbookStore (FAISS + sentence-transformers)  ──► top-k chunks
      │
      ▼
PaniniRuleEngine (32 medical rules)            ──► filter / block / warn
      │
      ▼
MCQ Answerer (BiomedBERT or keyword fallback)  ──► PRAG answer + rule trace
```

| Module | Path | Purpose |
|--------|------|---------|
| Question loader | `src/data/medqa_loader.py` | US train/dev/test JSONL |
| Textbook store | `src/knowledge/textbook_store.py` | Chunk, embed, FAISS retrieve |
| Rule engine | `src/rules/paninian_rule_engine.py` | 32 Paninian-governed clinical rules |
| Pipeline | `src/prag_pipeline.py` | End-to-end PRAG vs standard RAG |

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/yuvrajrajput/PRAG.git
cd PRAG
git checkout development   # active dev branch
pip install -r requirements.txt
```

### 2. Download MedQA data (not in git)

Download from [Google Drive](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing) and extract to `MedQA/data/data_clean/`.

```bash
pip install gdown
mkdir -p MedQA/data
gdown "https://drive.google.com/uc?id=1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw" -O "MedQA/data/medqa_data.zip"
unzip MedQA/data/medqa_data.zip -d MedQA/data/
```

### 3. Build vector index (~60 min on CPU)

```bash
python src/knowledge/textbook_store.py
```

Saves to `data/vector_store/` (18 textbooks, ~51k chunks).

### 4. Run pipeline

```bash
# Single question demo
python src/prag_pipeline.py

# Benchmark PRAG vs standard RAG (50 dev questions)
python src/prag_pipeline.py --compare 50 --split dev
```

Results saved to `outputs/benchmark_results.json`.

---

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases |
| `development` | Active research (use this for contributions) |
| `PRAG` | Legacy initial branch (not updated) |

---

## Benchmark snapshot (50 dev questions)

| System | Accuracy |
|--------|----------|
| Standard RAG | 16% |
| PRAG | 16% |

Same accuracy on this baseline because BiomedBERT embedding MCQ scoring is the bottleneck; PRAG adds **rule traces and chunk governance** without changing answers yet. See `outputs/benchmark_results.json` for per-question `rule_trace`.

---

## Keywords / topics

`medical-qa` `medqa` `rag` `retrieval-augmented-generation` `clinical-decision-support` `paninian-grammar` `rule-engine` `faiss` `usmle` `healthcare-ai` `nlp` `explainable-ai`

---

## Citation

If you use this codebase, cite MedQA:

```bibtex
@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}
```

---

## License

GPL-3.0 — see [LICENSE](LICENSE).
