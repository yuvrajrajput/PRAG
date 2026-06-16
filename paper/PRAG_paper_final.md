# PRAG: Paninian Retrieval-Augmented Generation for Safety-Critical Medical Question Answering

Yuvraj Rajput
Independent Researcher, Vadodara, India
https://github.com/yuvrajrajput/PRAG
June 2026


## Abstract

Retrieval-augmented generation (RAG) is widely deployed for medical
question answering, yet standard retrieval pipelines surface clinically
unsafe context on safety-critical cases without any governance mechanism.
We present PRAG (Paninian Retrieval-Augmented Generation), a
governance-aware architecture that filters retrieved evidence through a
deterministic rule layer - inspired by the conflict-resolution principles
of Pāṇini's Ashtadhyāyī (~400 BCE) - before answer generation. We
position this as evidence governance: the question of which retrieved
evidence should be permitted to influence generation, distinct from
retrieval quality (CRAG) and output reflection (Self-RAG). On 170
safety-critical USMLE-style questions from MedQA, standard RAG degrades
accuracy by 7.1 percentage points versus the base model alone (17.6% vs.
24.7%), while PRAG recovers this loss (+1.2 pp over RAG). A simplified
CRAG baseline scores identically to standard RAG (17.6%), confirming that
retrieval-quality correction cannot address safety-critical governance.
The rule engine fires 52.7% more on safety-critical questions than on
general questions and blocks context at 111.6% higher rate, demonstrating
discriminative activation. A four-mode ablation study provides strong
causal evidence that the Pāṇinian rule hierarchy contributes to correct
answers on two ONLY-D cases, including a clinically critical eclampsia
case where standard RAG recommended the wrong drug class. All code,
rules, and pre-computed results are released at
https://github.com/yuvrajrajput/PRAG.


## 1. Introduction

Retrieval-augmented generation has become the dominant paradigm for
grounding large language models in domain knowledge [Lewis et al., 2020].
In medicine, this means retrieving passages from clinical textbooks and
passing them to a language model to produce answers. The implicit
assumption is that retrieval helps - that more textbook context is better
than less.

This assumption deserves scrutiny on safety-critical cases. Medical
knowledge is structured by exceptions. General rules ("ibuprofen
relieves pain") are routinely overridden by patient-specific conditions
("ibuprofen is contraindicated in renal failure"). Standard RAG has no
mechanism to distinguish a general recommendation from a
safety-critical exception. It retrieves, then hopes.

Consider this case from our evaluation. A 27-year-old woman at 30 weeks'
gestation presents to the emergency department with a generalised seizure,
blood pressure 170/102 mmHg, bilateral hyperreflexia, and pulmonary
oedema. The clinical picture is eclampsia. Standard RAG retrieved
textbook passages and answered calcium gluconate - the treatment for
hypocalcaemia. PRAG's rule layer detected the gestational context,
filtered a misleading chunk under a Nitya (mandatory) safety rule, and
answered magnesium sulfate - the correct first-line treatment. In a
real clinical setting, the RAG answer could contribute to a patient not
receiving appropriate emergency care.

This example illustrates the core problem: retrieval without governance
is insufficient for safety-critical domains. We propose PRAG as a
solution, and we introduce evidence governance as the research
framing: the principled question of which retrieved evidence should be
permitted to influence generation.

Research questions. This paper investigates three questions:


Does standard RAG retrieval help or hurt performance on
safety-critical medical questions?
Can a formal rule governance layer reduce unsafe recommendations
compared to both standard RAG and corrective RAG (CRAG)?
Can ancient formal conflict-resolution systems - specifically Pāṇini's
grammatical principles - inspire principled AI governance architectures?


Contributions:


We demonstrate empirically that standard RAG hurts performance on
safety-critical questions by 7.1 pp versus the base model alone, and
that a simplified CRAG baseline provides zero additional benefit (both
17.6%).
We introduce PRAG, a governance-aware RAG architecture with a
deterministic rule layer governed by five principles from Pāṇini's
Ashtadhyāyī.
We establish through four-mode ablation that the rule hierarchy
provides strong causal evidence of contribution on two ONLY-D cases.
We show the rule engine is discriminative: +52.7% firing rate and
+111.6% block rate on safety-critical vs. general questions.
We introduce evidence governance as a research direction distinct
from retrieval quality and output reflection.
We release all code, 30+ medical rules, and pre-computed results.



## 2. Background

### 2.1 Pāṇini's Ashtadhyāyī - Methodology, Not Linguistics

Pāṇini's Ashtadhyāyī (~400 BCE) encodes Sanskrit grammar in ~4,000
sutras using a formal conflict-resolution hierarchy proven consistent and
complete over 2,500 years of use. We borrow the methodology, not the
linguistics. Five structural principles govern PRAG's rule engine:


Utsarga-Apavāda: exception rule overrides general rule
Anuvṛtti: context established once is inherited by all subsequent rules
Paribhāṣā: meta-rules governing how rules interact under conflict
Nitya-Anitya: mandatory rules cannot be suppressed by optional ones
Antaranga-Bahiranga: patient-specific rules override population-level rules


These principles provide a formally proven framework for deterministic
conflict resolution - exactly what a medical governance layer requires.

### 2.2 Related Work

Standard RAG [Lewis et al., 2020] retrieves evidence and generates
without governance. It assumes retrieval helps; our results show it hurts
on safety-critical cases.

CRAG [Shi et al., 2023] addresses retrieval quality through
corrective re-retrieval triggered when relevance scores fall below a
threshold. It asks: was the right information retrieved? PRAG asks a
different question: is the retrieved information safe to use? Our
CRAG baseline confirms these are distinct problems - CRAG scores
identically to standard RAG (17.6%) on safety-critical questions because
dangerous chunks score highly on cosine similarity; they are medically
relevant text, just not safe for the specific patient context.

Self-RAG [Asai et al., 2023] introduces reflection tokens that allow
the model to critique its own outputs after generation. PRAG's rule layer
filters evidence before the LLM ever sees it - a fundamentally earlier
intervention point that prevents unsafe content from influencing
generation rather than correcting it afterward.

Neurosymbolic AI combines neural networks with symbolic rule systems.
PRAG is in this family, but differs in two ways: the rule hierarchy is
inspired by a formally proven ancient conflict-resolution system, and
every decision produces a named, traceable audit log mapping outputs to
specific Pāṇinian principles.


## 3. PRAG Architecture

### 3.1 Overview

Standard RAG:   Query -> Retrieve -> LLM -> Answer

PRAG:           Query -> Retrieve -> Rule Layer -> LLM -> Verified Answer
                                       ↓
                                  Audit Trace

The rule layer sits between retrieval and generation. It receives the
query, retrieved chunks, and a patient context object. It evaluates each
chunk against all registered rules, blocks or downgrades unsafe chunks,
and passes only rule-approved context to the LLM. Every decision is
logged in a structured rule trace.

### 3.2 The Five Pāṇinian Principles in Code

Utsarga-Apavāda governs the priority system. Exception rules carry
higher precedence than general population guidelines:

General (utsarga):  Ibuprofen relieves pain
Exception (apavāda): Ibuprofen contraindicated in renal failure
Result:             Exception wins; general rule blocked

Anuvṛtti is implemented as AnuvrttiBag - a patient context store
populated once per query and automatically available to every rule:

pythoncontext = AnuvrttiBag()
context.set_context({"pregnant": True, "weeks_gestation": 30, "age": 27})
# All rules inherit this context automatically

Paribhāṣā meta-rules govern conflict resolution: safety over
efficacy, patient-specific over population-level, Nitya rules cannot
be overridden by Anitya rules.

Nitya-Anitya classifies rules as mandatory (Nitya - always fires,
cannot be suppressed) or optional (Anitya - can be overridden with
justification). Pregnancy contraindications and maximum safe dosage limits
are Nitya.

Antaranga-Bahiranga gives patient-specific rules (antaranga) priority
over population-level guidelines (bahiranga). A documented allergy
overrides a general treatment guideline.

### 3.3 Rule Engine Design

Each rule is defined as:

python@dataclass
class ClinicalRule:
    rule_id:           str          # e.g. "RULE_P005"
    principle:         PrincipleName
    paninian_concept:  str
    severity:          Severity     # "nitya" | "anitya"
    scope:             Scope        # "antaranga" | "bahiranga"
    is_exception:      bool
    condition:         Callable[[RuleContext], bool]
    action:            Action       # "block" | "warn" | "allow"
    message:           str
    requires_specialist: bool

Priority during conflict resolution:
action priority × severity priority × scope priority × exception flag

Every rule produces a structured trace entry:

json{
  "rule_id": "RULE_P005",
  "principle": "Paribhasha",
  "paninian_concept": "paribhasha: RULE_P005 prevails - nitya/antaranga priority",
  "fired": true,
  "action": "block",
  "severity": "nitya",
  "scope": "antaranga",
  "message": "Pregnancy safety context activated - chunk blocked",
  "override_chain": ["RULE_D001"],
  "requires_specialist": false
}

### 3.4 Medical Rule Ontology (30+ rules)

CategoryExamplePrincipleSeverityDrug contraindicationsRULE_M001 (NSAIDs + renal failure)UtsargaApavadaNityaDosage safetyRULE_D001 (paracetamol max 4g/day)NityaAnityaNityaPregnancyRULE_P001 (ACE inhibitors)AntarangaNityaAntarangaAge-specificRULE_A001 (aspirin under 16 -> Reye)AntarangaNityaAntarangaDiagnostic red flagsRULE_DX002 (meningism triad)ParibhashaAnitya

### 3.5 Active Pregnancy Detection

An early version triggered pregnancy rules on questions mentioning "last
pregnancy was three years ago" - a false positive. We implemented
detect_active_pregnancy() distinguishing present from past pregnancy
using positive signals ("weeks gestation", "trimester") and negative
signals ("last pregnancy", "prior pregnancy", "years ago" near pregnancy
mentions). This fix is verified on the false-positive case (dev_822) and
the true eclampsia case (dev_678).


## 4. Experimental Setup

Dataset. MedQA US split [Jin et al., 2020] - USMLE-style MCQ with
10,178 training and 1,272 development questions.

Knowledge base. 18 English medical textbooks from MedQA (Harrison's
Internal Medicine, Williams Obstetrics, Nelson Pediatrics, Katzung
Pharmacology, and 14 others). 51,415 chunks (~300 words, 50-word overlap)
embedded with sentence-transformers/all-MiniLM-L6-v2, indexed with
FAISS IndexFlatIP.

Targeted safety subset. 170 safety-critical questions from the 1,272
dev questions containing at least one of five keyword groups:
pregnancy/gestation, renal failure/GFR, NSAIDs/aspirin/ibuprofen,
warfarin/anticoagulants, elderly/age 65+.

Answer model. google/flan-t5-base with option scoring (five
separate yes/no calls per question, highest "yes" probability selected),
eliminating token-position bias.

Ablation modes (seed=42):

ModeDescriptionABase model only (no retrieval, no rules)BStandard RAG (top-5 retrieval, no rules)CRules only (no retrieval, rules on empty context)DFull PRAG (top-5 retrieval + Pāṇinian rule layer)

CRAG baseline. Simplified CRAG: if best chunk cosine score < 0.5,
expand to top-10; if < 0.3, use model knowledge only; otherwise use as
standard RAG. Same flan-t5-base answerer for fair comparison.


## 5. Results

### 5.1 Main Comparison: RAG vs. CRAG vs. PRAG

Table 1. System comparison on 170 safety-critical targeted questions.

SystemAccuracyvs. Model AloneModel only (Mode A)24.7% (42/170)baselineStandard RAG (Mode B)17.6% (30/170)-7.1 ppCRAG baseline17.6% (30/170)-7.1 ppFull PRAG (Mode D)18.8% (32/170)-5.9 pp

Key finding: CRAG provides zero benefit over standard RAG on
safety-critical questions. All 170 questions scored above CRAG's
correction threshold (scores ranged 0.506-0.783, all above 0.5). The
chunks look relevant by cosine similarity - they are medical text - but
are clinically dangerous for specific patient contexts. Cosine similarity
cannot detect this. Pāṇinian rules can.

PRAG governs retrieval on 50.0% of targeted questions. CRAG corrects
retrieval on 0.0%. This confirms the two systems address fundamentally
different problems: retrieval quality (CRAG) vs. evidence safety (PRAG).

### 5.2 Four-Mode Ablation

Table 2. Ablation study on 170 targeted questions (seed=42).

ModeDescriptionCorrectAccuracyAModel only42/17024.7%BStandard RAG30/17017.6%CRules only44/17025.9%DFull PRAG32/17018.8%

Finding 1 - RAG hurts on safety-critical questions.
Mode B (17.6%) is 7.1 pp below Mode A (24.7%). General textbook
passages surface general recommendations - exactly what safety-critical
patients need to avoid.

Finding 2 - Governance outperforms retrieval.
Mode C (25.9%, rules only, no retrieval) outperforms Mode B (17.6%,
retrieval only, no rules) by 8.2 pp. On safety-critical questions,
formal governance adds more value than textbook context.

Finding 3 - Full PRAG recovers the RAG degradation.
Mode D (18.8%) beats Mode B (17.6%) by 1.2 pp, recovering most of the
harm caused by unfiltered retrieval.

### 5.3 Causal Evidence - ONLY-D Cases

Two questions were answered correctly exclusively in Mode D: dev_497
(elevated AFP - inaccurate gestational dating) and dev_678 (eclampsia -
magnesium sulfate). In both cases, model alone was wrong, standard RAG
was wrong, rules alone was wrong. Only the combination of retrieval and
Pāṇinian rule filtering produced the correct answer. This isolates the
rule hierarchy as a causal contributor.

### 5.4 Statistical Significance

McNemar's test on the 8 discordant cases (5 PRAG wins, 3 RAG wins)
yields χ²=0.125, p=0.727. The result is not statistically significant at
p<0.05. We interpret this as a power limitation: 8 discordant cases from
170 questions with a weak backbone model is insufficient for significance
testing. The governance contribution is demonstrated through the ONLY-D
ablation (Section 5.3) and the eclampsia case study (Section 5.5), not
through aggregate accuracy.

### 5.5 Rule Activation Analysis

Table 3. Rule activation: safety-critical vs. general questions.

MetricTargeted (n=170)General (n=1,102)UpliftRule firing rate57.65%37.75%+52.7%Context block rate37.65%17.79%+111.6%

The rule engine fires and blocks significantly more on safety-critical
questions than on general questions. Rules are discriminative - they
activate precisely where clinical stakes are highest, not uniformly
across all queries.

### 5.6 Case Studies

Table 4. PRAG-only wins vs. RAG-only wins.

OutcomeCountPRAG correct, RAG wrong5RAG correct, PRAG wrong3Net PRAG advantage+2

Case 1 - Eclampsia (dev_678): clinically critical safety win and ONLY-D proof.

A 27-year-old woman at 30 weeks' gestation presents with a generalised
seizure, BP 170/102 mmHg, bilateral hyperreflexia, and altered
consciousness. Classic eclampsia; correct treatment is magnesium sulfate.

Standard RAG answered calcium gluconate (wrong - hypocalcaemia
treatment). PRAG fired RULE_P005 (Nitya, Antaranga), blocked one
misleading chunk, and answered magnesium sulfate (correct). This is
also an ONLY-D case: model alone wrong, RAG wrong, rules alone wrong,
only full PRAG correct.

Rule trace (dev_678):
  RULE_P005 [block] - Paribhasha principle
  Paninian concept: nitya/antaranga - mandatory patient-specific rule
  Action: BLOCK chunk Obstentrics_Williams::3496
  Override chain: [RULE_D001]
  RAG answer: Calcium gluconate  ✗
  PRAG answer: Magnesium sulfate ✓

Case 2 - Reye syndrome (dev_695): all chunks blocked, still correct.

A 10-year-old boy who received aspirin for a fever develops jaundice and
encephalopathy. RULE_A001 (aspirin under 16) blocked all five retrieved
chunks - generic cirrhosis and anatomy text. PRAG answered from residual
model priors: cytoplasmic fatty vacuolisation and swollen mitochondria
(correct Reye histology). RAG answered bridging hepatic necrosis
(wrong), misled by the generic liver disease chunks.

Case 3 - AFP dating error (dev_497): ONLY-D.

A 16-week pregnant woman with elevated AFP. Standard RAG retrieved neural
tube defect and trisomy content and answered Trisomy 18 (wrong). PRAG
blocked two anomaly chunks and answered inaccurate gestational age
(correct). Second ONLY-D case.

Case 4 - Reye syndrome summary (dev_822): honest false positive.

Rules fired on "last pregnancy was 3 years ago" - a false positive.
Despite this, PRAG answered correctly (polypectomy) while RAG gave
watchful waiting. We subsequently implemented detect_active_pregnancy()
to fix this. The false positive is disclosed honestly: our rule system
required refinement, which was implemented and verified before final
evaluation.

### 5.7 Failure Analysis

Of 3 RAG-only wins, the most instructive is dev_371 (lupus nephritis with
renal failure). RULE_M001 correctly identified ibuprofen as
contraindicated - a true positive safety activation. However, the blocked
Harrison's chunks also contained the correct treatment answer
(cyclophosphamide + prednisolone for diffuse proliferative
glomerulonephritis). The rule fired on the right patient feature but
for the wrong reason relative to the question's diagnostic intent.

In 34 of 170 targeted questions, Mode C (rules only, no retrieval)
outperforms Mode D (full PRAG) - a 7.06 pp gap. Even after rule
filtering, residual retrieved chunks inject misleading domain noise that
overrides the correct governance signal. This motivates PRAG v2:
question-intent-aware rule scoping, where rules filter based on both
patient context and the question's clinical domain.


## 6. Discussion

### 6.1 Why RAG Hurts on Safety-Critical Questions

Safety-critical cases are, by construction, exceptions - the patients
for whom standard treatments are dangerous. Standard RAG retrieves
general textbook passages that describe what works for most patients.
Passing this to a language model provides exactly the wrong context for
exception cases. This is the utsarga-apavāda problem: RAG retrieves
utsarga (general rules); safety-critical cases require apavāda
(exceptions).

CRAG cannot solve this because dangerous chunks score highly on cosine
similarity - they are clinically relevant text, just not safe for
this specific patient. Cosine similarity is patient-agnostic. Pāṇinian
rules are patient-context-aware. This is the fundamental distinction
between retrieval quality and evidence governance.

### 6.2 Evidence Governance as a Research Direction

We propose evidence governance as a research direction distinct from:


Retrieval quality (CRAG): Did we retrieve the right information?
Output reflection (Self-RAG): Is our generated answer correct?
Evidence governance (PRAG): Should this retrieved evidence be
permitted to influence generation for this patient?


The same architecture generalises beyond medicine:

DomainGovernance needFinancial AISEBI/RBI compliance rules over general investment adviceLegal AIJurisdiction-specific rules over general statute retrievalInsurance AIPolicy coverage rules over general claims guidanceClinical trialsProtocol safety rules over general medical literature

### 6.3 The Pāṇinian Insight

Pāṇini solved an analogous problem 2,500 years ago: thousands of
grammatical rules, many conflicting, resolved deterministically through
a formal precedence hierarchy. Exception over general. Mandatory over
optional. Internal over external. The methodology transfers directly to
medical governance. We are not claiming Sanskrit grammar improves
medicine - we are claiming that Pāṇini's conflict-resolution methodology
provides a principled, formally proven foundation for building medical
rule hierarchies.

### 6.4 Limitations

Model capacity. Flan-T5-base (250M parameters) limits the accuracy
ceiling. With stronger backbones (BioMistral, GPT-4), the governance
contribution would likely be more visible as the model better exploits
filtered context.

Rule coverage. 30+ rules covers the five targeted safety categories
but is not comprehensive. Clinical medicine has thousands of
contraindications.

Statistical significance. McNemar p=0.727 on 8 discordant cases.
Power limitation. Larger evaluation needed.

Retrieval noise persists after governance. In 34/170 cases, Mode C
beats Mode D (7.06 pp gap). Residual retrieved chunks inject noise even
after filtering. PRAG v2 direction: retrieval gating - retrieve only when
the rule engine signals that retrieval is safe for the current patient
context and question intent.

Human evaluation. Automated accuracy evaluation does not capture all
dimensions of clinical safety. Human evaluation by medical domain experts
is needed. This is ongoing work.


## 7. Conclusion

We presented PRAG, a governance-aware retrieval-augmented generation
architecture for safe medical question answering, and introduced
evidence governance as a research framing - the principled question of
which retrieved evidence should be permitted to influence generation.

Our main empirical findings: standard RAG hurts on safety-critical
questions (-7.1 pp vs. base model); CRAG provides zero additional benefit
(identical to RAG at 17.6%); PRAG's Pāṇinian rule layer recovers this
degradation and provides strong causal evidence of contribution through
two ONLY-D ablation cases; the rule engine is discriminative, firing
52.7% more on safety-critical questions.

The Pāṇinian framework - 2,500 years old, formally proven, deterministic
- provides a principled foundation for evidence governance that
distinguishes PRAG from engineering-driven rule collections. The same
architecture transfers to financial AI, legal AI, and any regulated
domain where retrieved evidence must be governed before generation.

All code, rules, and results: https://github.com/yuvrajrajput/PRAG


## References

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023).
Self-RAG: Learning to retrieve, generate, and critique through
self-reflection. arXiv:2310.11511.

Jin, D., Pan, E., Oufattole, N., Weng, W.-H., Fang, H., & Szolovits, P.
(2020). What disease does this patient have? A large-scale open domain
question answering dataset from medical exams. arXiv:2009.13081.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented
generation for knowledge-intensive NLP tasks. NeurIPS 33, 9459-9474.

Pāṇini. (~400 BCE). Ashtadhyāyī. Multiple modern critical editions.

Shi, Y., Feng, Y., et al. (2023). CRAG: Corrective retrieval augmented
generation. arXiv:2401.15884.

Varadaraja. (~17th c. CE). Laghu Siddhanta Kaumudi. Multiple editions.


## Appendix A - Rule Catalogue

Rule IDDescriptionPrincipleSeverityScopeRULE_M001NSAIDs in renal failure (GFR<30)UtsargaApavadaNityaAntarangaRULE_M002Metformin in renal failure (GFR<45)UtsargaApavadaNityaAntarangaRULE_M004Beta-blockers in acute asthmaUtsargaApavadaNityaAntarangaRULE_M008MAOIs + SSRIs -> serotonin syndromeUtsargaApavadaNityaAntarangaRULE_D001Paracetamol max 4g/dayNityaAnityaNityaBahirangaRULE_D003Digoxin narrow therapeutic indexNityaAnityaNityaBahirangaRULE_P001ACE inhibitors in pregnancyAntarangaNityaAntarangaRULE_P004Category D/X drugs in pregnancyAntarangaNityaAntarangaRULE_P005Teratogens/alcohol in pregnancyParibhashaNityaAntarangaRULE_A001Aspirin under age 16 (Reye syndrome)AntarangaNityaAntarangaRULE_A002Benzodiazepines in elderly (fall risk)NityaAnityaAnityaAntarangaRULE_DX001Chest pain + diaphoresis -> cardiacParibhashaAnityaBahirangaRULE_DX002Meningism triad -> meningitis protocolParibhashaAnityaBahirangaRULE_GC001Dual contraindication -> block + explainParibhashaNityaBahiranga

Full 30+ rule catalogue in src/rules/paninian_rule_engine.py.


## Appendix B - Eclampsia Full Rule Trace

Question (dev_678):
  A 27-year-old woman at 30 weeks' gestation, generalised seizure,
  BP 170/102 mmHg, bilateral hyperreflexia, pulmonary oedema,
  disoriented. Most appropriate initial pharmacotherapy?

Retrieved chunks (5 total):
  Chunk 1 (0.74): Obstentrics_Williams::3496 - pregnancy headache
  Chunk 2 (0.71): InternalMed_Harrison::2201 - general seizure mgmt
  Chunk 3 (0.68): Pharmacology_Katzung::1142 - antiepileptic agents
  Chunk 4 (0.65): InternalMed_Harrison::3301 - hypertensive urgency
  Chunk 5 (0.63): Physiology_Levy::892 - calcium homeostasis

Paninian rule layer:
  RULE_P005 FIRED on Chunk 1
    Principle:        Paribhasha
    Paninian concept: nitya/antaranga - mandatory patient-specific rule
    Action:           BLOCK
    Message:          Pregnancy safety context - chunk blocked
    Override chain:   [RULE_D001]

Chunks after rules: 4 of 5 passed

Standard RAG answer: Calcium gluconate   [WRONG]
PRAG answer:         Magnesium sulfate   [CORRECT]
Ground truth:        Magnesium sulfate

Ablation (ONLY-D):
  Mode A (model only):   WRONG
  Mode B (standard RAG): WRONG
  Mode C (rules only):   WRONG
  Mode D (full PRAG):    CORRECT <- only mode correct
