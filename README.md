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
MCQ Answerer (FLAN-T5 default, or BiomedBERT)  ──► PRAG answer + rule trace
```

| Module | Path | Purpose |
|--------|------|---------|
| Question loader | `src/data/medqa_loader.py` | US train/dev/test JSONL |
| Textbook store | `src/knowledge/textbook_store.py` | Chunk, embed, FAISS retrieve |
| Rule engine | `src/rules/paninian_rule_engine.py` | 32 Paninian-governed clinical rules |
| Pipeline | `src/prag_pipeline.py` | End-to-end PRAG vs standard RAG |
| Ablation study | `src/ablation_study.py` | Four-mode ablation (A–D) on 170 safety questions |
| Targeted eval | `src/targeted_eval.py` | Keyword-filtered safety evaluation |
| Paper numbers | `src/paper_numbers.py` | Export tables from experiment JSON |

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

# Benchmark PRAG vs standard RAG (50 dev questions, FLAN-T5)
python src/prag_pipeline.py --compare 50 --split dev --model flan-t5
```

Results saved to `outputs/benchmark_results.json`.

```bash
# Four-mode ablation on 170 safety-critical questions
python src/ablation_study.py

# Targeted safety evaluation
python src/targeted_eval.py --split dev --model flan-t5

# Export paper tables from results JSON
python src/paper_numbers.py
```

---

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases |
| `development` | Active research (use this for contributions) |
| `PRAG` | Legacy initial branch (not updated) |

---

## Key findings

### Finding 1 — Standard RAG hurts on safety-critical questions

On 170 safety-critical questions (pregnancy, renal failure, NSAIDs,
anticoagulants, paediatric contraindications), standard RAG retrieval
**actively degrades** performance compared to the base model alone:

| Mode | Description | Accuracy |
|------|-------------|----------|
| A — model only | No retrieval, no rules | 24.7% (42/170) |
| B — standard RAG | Retrieval, no rules | 17.6% (30/170) |
| C — rules only | Rules, no retrieval | 25.9% (44/170) |
| D — full PRAG | Retrieval + rules | **18.8% (32/170)** |

RAG drops accuracy by **7.1 percentage points** versus the base model on
safety-critical cases. Full PRAG improves over standard RAG by **+1.2 pp**
and prevents dangerous answers in clinically critical cases (see Finding 3).

### Finding 2 — The rule engine is discriminative, not noisy

| Metric | Safety-critical questions | General questions | Uplift |
|--------|--------------------------|-------------------|--------|
| Rule firing rate | 57.65% | 37.75% | **+52.7%** |
| Context block rate | 37.65% | 17.79% | **+111.6%** |

Rules fire and block significantly more on exactly the questions where
mistakes are clinically dangerous — not uniformly across all questions.

### Finding 3 — Causal proof via ablation

Two questions (`dev_497`, `dev_678`) were answered correctly **only in
Mode D** — wrong in model-alone, wrong in standard RAG, wrong in
rules-alone, correct only in full PRAG. This isolates the rule hierarchy
itself as the contributing factor, not retrieval or model prior knowledge.

Five additional questions show PRAG correct where standard RAG was wrong
(`dev_401`, `dev_497`, `dev_678`, `dev_695`, `dev_822`).

The most critical case (`dev_678`, eclampsia):

```
Query   : 30-week pregnant woman, seizures, BP 170/102, hyperreflexia
RAG     : Calcium gluconate  ✗  (treatment for hypocalcaemia)
PRAG    : Magnesium sulfate  ✓  (correct first-line eclampsia treatment)

Rules fired:
  RULE_P005 [block] — pregnancy-safety context filtered (Nitya, Antaranga)
  Pāṇinian principle: Utsarga-Apavāda — exception overrides general rule
```

> In a real clinical setting, the RAG answer could contribute to a patient
> not receiving the correct treatment for a life-threatening emergency.

Pre-computed results for all 170 questions are in `outputs/ablation_results.json`
and `outputs/targeted_results.json`. Run `python src/paper_numbers.py` to
regenerate copy-paste tables for the paper.

---

## Keywords / topics

`medical-qa` `medqa` `rag` `retrieval-augmented-generation` `clinical-decision-support` `paninian-grammar` `rule-engine` `faiss` `usmle` `healthcare-ai` `nlp` `explainable-ai`

---

## Citation

If you use PRAG in your research, please cite:

```bibtex
@software{rajput2026prag,
  author    = {Rajput, Yuvraj},
  title     = {{PRAG}: {P}aninian Retrieval-Augmented Generation
               for Safety-Critical Medical {AI}},
  year      = {2026},
  url       = {https://github.com/yuvrajrajput/PRAG},
  version   = {1.0.0}
}
```

Also cite the MedQA dataset:

```bibtex
@article{jin2020disease,
  title   = {What Disease does this Patient Have? A Large-scale Open Domain
             Question Answering Dataset from Medical Exams},
  author  = {Jin, Di and Pan, Eileen and Oufattole, Nassim and
             Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal = {arXiv preprint arXiv:2009.13081},
  year    = {2020}
}
```

GitHub also reads citation metadata from [`CITATION.cff`](CITATION.cff).

---

## License

GPL-3.0 — see [LICENSE](LICENSE).
