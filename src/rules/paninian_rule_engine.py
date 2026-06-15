"""
Paninian Rule Engine for PRAG medical AI research.

Maps five principles from Panini's Ashtadhyayi / Laghu Siddhanta Kaumudi onto
clinical rule evaluation: general-vs-exception (utsarga-apavada), context
inheritance (anuvrtti), meta-rules (paribhasha), mandatory-vs-optional
(nitya-anitya), and internal-vs-external scope (antaranga-bahiranga).

The trace log produced by :class:`PaniniRuleEngine` is the primary auditable
artifact for research: every PRAG answer can be tied to which principle fired
and why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

Severity = Literal["nitya", "anitya"]
Scope = Literal["antaranga", "bahiranga"]
Action = Literal["allow", "block", "warn"]
PrincipleName = Literal[
    "UtsargaApavada",
    "Anuvrtti",
    "Paribhasha",
    "NityaAnitya",
    "AntarangaBahiranga",
]

ACTION_PRIORITY: dict[Action, int] = {"block": 3, "warn": 2, "allow": 1}
SEVERITY_PRIORITY: dict[Severity, int] = {"nitya": 2, "anitya": 1}
SCOPE_PRIORITY: dict[Scope, int] = {"antaranga": 2, "bahiranga": 1}


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------


@dataclass
class RuleContext:
    """Unified evaluation context passed to every rule condition."""

    query: str
    retrieved_context: str
    patient: dict[str, Any]
    combined_text: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.combined_text = " ".join(
            [self.query, self.retrieved_context, _patient_text(self.patient)]
        ).lower()


def _patient_text(patient: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in patient.items():
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts)


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term.lower() in text for term in terms)


def _mentions_drug(text: str, drug_terms: list[str]) -> bool:
    return _contains_any(text, drug_terms)


NSAID_TERMS = [
    "nsaid", "ibuprofen", "naproxen", "diclofenac", "ketorolac",
    "indomethacin", "celecoxib", "meloxicam", "aspirin",
]
ACE_TERMS = [
    "ace inhibitor", "lisinopril", "enalapril", "ramipril", "captopril",
    "benazepril", "perindopril",
]
BETA_BLOCKER_TERMS = [
    "beta blocker", "beta-blocker", "propranolol", "metoprolol",
    "atenolol", "bisoprolol", "carvedilol",
]
FLUOROQUINOLONE_TERMS = [
    "fluoroquinolone", "ciprofloxacin", "levofloxacin", "moxifloxacin",
    "ofloxacin",
]
TETRACYCLINE_TERMS = [
    "tetracycline", "doxycycline", "minocycline", "tigecycline",
]
SSRI_TERMS = [
    "ssri", "fluoxetine", "sertraline", "paroxetine", "citalopram",
    "escitalopram",
]
MAO_TERMS = [
    "mao inhibitor", "maoi", "phenelzine", "tranylcypromine",
    "isocarboxazid", "selegiline",
]
STATIN_TERMS = [
    "statin", "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
]
BENZO_TERMS = [
    "benzodiazepine", "diazepam", "lorazepam", "alprazolam", "clonazepam",
    "midazolam",
]
AMINOGLYCOSIDE_TERMS = [
    "aminoglycoside", "gentamicin", "tobramycin", "amikacin", "streptomycin",
]
PENICILLIN_TERMS = [
    "penicillin", "amoxicillin", "ampicillin", "piperacillin",
    "nafcillin", "oxacillin",
]
CATEGORY_DX_TERMS = [
    "isotretinoin", "methotrexate", "valproate", "valproic acid",
    "lithium", "phenytoin", "carbamazepine", "thalidomide",
]


# ---------------------------------------------------------------------------
# Principle 1 — UtsargaApavada (general vs exception)
# ---------------------------------------------------------------------------


@dataclass
class UtsargaApavadaRule:
    """
    Utsarga-apavada: a general rule (utsarga) yields to a more specific
    exception rule (apavada). Exception rules override any rule_ids listed
    in ``overrides``.
    """

    rule_id: str
    description: str
    severity: Severity
    scope: Scope
    condition: Callable[[RuleContext], bool]
    action: Action
    message: str
    is_exception: bool = False
    overrides: list[str] = field(default_factory=list)
    principle: PrincipleName = "UtsargaApavada"
    paninian_concept: str = "utsarga (general rule)"
    requires_specialist: bool = False

    def evaluate(self, context: RuleContext) -> dict[str, Any] | None:
        """Return a trace dict if the rule fires, else None."""
        if not self.condition(context):
            return None
        return {
            "rule_id": self.rule_id,
            "principle": self.principle,
            "paninian_concept": self.paninian_concept,
            "fired": True,
            "action": self.action,
            "severity": self.severity,
            "scope": self.scope,
            "message": self.message,
            "override_chain": [],
            "requires_specialist": self.requires_specialist,
            "description": self.description,
            "is_exception": self.is_exception,
            "overrides": list(self.overrides),
        }


# ---------------------------------------------------------------------------
# Principle 2 — Anuvrtti (context inheritance)
# ---------------------------------------------------------------------------


class AnuvrttiBag:
    """
    Anuvrtti: contextual conditions are stated once and inherited by all
    subsequent rules, mirroring Panini's ellipsis of repeated terms across
    a rule block.
    """

    def __init__(self) -> None:
        self._context: dict[str, Any] = {"active_pregnancy": False}

    def set_context(self, patient_dict: dict[str, Any]) -> None:
        """Store patient context for automatic inheritance by all rules."""
        self._context = dict(patient_dict)
        self._context.setdefault("active_pregnancy", False)

    def get_context(self) -> dict[str, Any]:
        """Return the inherited patient context."""
        return dict(self._context)

    def clear(self) -> None:
        """Reset inherited context."""
        self._context = {"active_pregnancy": False}

    def merge(self, patient_dict: dict[str, Any]) -> dict[str, Any]:
        """Merge call-time context over inherited anuvrtti defaults."""
        merged = dict(self._context)
        merged.update(patient_dict)
        merged.setdefault("active_pregnancy", False)
        return merged


# ---------------------------------------------------------------------------
# Principle 3 — Paribhasha (meta-rules)
# ---------------------------------------------------------------------------


class Paribhasha:
    """
    Paribhasha: meta-rules that govern how other rules interact — analogous
    to Panini's paribhasha sutras that resolve ambiguity among vartikas.
    """

    @staticmethod
    def resolve_conflict(
        rule_a: dict[str, Any],
        rule_b: dict[str, Any],
    ) -> dict[str, Any]:
        """
        When two rules conflict, the safer / higher-priority rule wins.

        Priority order:
        1. Stronger action (block > warn > allow)
        2. Nitya over anitya (nitya cannot be overridden by anitya)
        3. Antaranga over bahiranga (patient-specific over population)
        4. Exception (apavada) over general (utsarga)
        """
        winner, loser = rule_a, rule_b

        if ACTION_PRIORITY[rule_b["action"]] > ACTION_PRIORITY[rule_a["action"]]:
            winner, loser = rule_b, rule_a
        elif (
            ACTION_PRIORITY[rule_b["action"]] == ACTION_PRIORITY[rule_a["action"]]
            and SEVERITY_PRIORITY[rule_b["severity"]] > SEVERITY_PRIORITY[rule_a["severity"]]
        ):
            winner, loser = rule_b, rule_a
        elif (
            ACTION_PRIORITY[rule_b["action"]] == ACTION_PRIORITY[rule_a["action"]]
            and SEVERITY_PRIORITY[rule_b["severity"]] == SEVERITY_PRIORITY[rule_a["severity"]]
            and SCOPE_PRIORITY[rule_b["scope"]] > SCOPE_PRIORITY[rule_a["scope"]]
        ):
            winner, loser = rule_b, rule_a
        elif rule_b.get("is_exception") and not rule_a.get("is_exception"):
            winner, loser = rule_b, rule_a

        winner = dict(winner)
        chain = list(winner.get("override_chain", []))
        chain.append(loser["rule_id"])
        winner["override_chain"] = chain
        winner["paninian_concept"] = (
            f"paribhasha: {winner['rule_id']} prevails over {loser['rule_id']} "
            f"(safer / nitya / antaranga / apavada priority)"
        )
        winner["principle"] = "Paribhasha"
        return winner

    @staticmethod
    def dosage_vs_age(rule_dosage: dict[str, Any], rule_age: dict[str, Any]) -> dict[str, Any]:
        """When dosage conflicts with age rule, age rule takes priority."""
        winner = dict(rule_age)
        chain = list(winner.get("override_chain", []))
        chain.append(rule_dosage["rule_id"])
        winner["override_chain"] = chain
        winner["principle"] = "Paribhasha"
        winner["paninian_concept"] = (
            "paribhasha: age-specific rule (antaranga) overrides dosage guideline"
        )
        return winner

    @staticmethod
    def nitya_blocks_anitya_override(
        nitya_rule: dict[str, Any],
        anitya_rule: dict[str, Any],
    ) -> dict[str, Any]:
        """Nitya (mandatory) rules cannot be suppressed by anitya rules."""
        if (
            nitya_rule["severity"] == "nitya"
            and anitya_rule["severity"] == "anitya"
            and nitya_rule["action"] == "block"
        ):
            result = dict(nitya_rule)
            chain = list(result.get("override_chain", []))
            chain.append(anitya_rule["rule_id"])
            result["override_chain"] = chain
            result["principle"] = "Paribhasha"
            result["paninian_concept"] = (
                "paribhasha: nitya rule cannot be overridden by anitya rule"
            )
            return result
        return nitya_rule


# ---------------------------------------------------------------------------
# Principle 4 — NityaAnityaClassifier
# ---------------------------------------------------------------------------


class NityaAnityaClassifier:
    """
    Nitya-anitya: classifies rules as mandatory (nitya, always binding) or
    contextual (anitya, defeasible with justification).
    """

    @staticmethod
    def is_nitya(rule: dict[str, Any] | UtsargaApavadaRule) -> bool:
        severity = rule["severity"] if isinstance(rule, dict) else rule.severity
        return severity == "nitya"

    @staticmethod
    def is_anitya(rule: dict[str, Any] | UtsargaApavadaRule) -> bool:
        return not NityaAnityaClassifier.is_nitya(rule)

    @staticmethod
    def filter_suppressed(
        fired: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove anitya blocks overridden by nitya blocks via paribhasha."""
        nitya_blocks = {
            r["rule_id"]
            for r in fired
            if r["severity"] == "nitya" and r["action"] == "block"
        }
        kept: list[dict[str, Any]] = []
        for rule in fired:
            if (
                rule["severity"] == "anitya"
                and rule["action"] == "block"
                and any(overridden in nitya_blocks for overridden in rule.get("overrides", []))
            ):
                continue
            kept.append(rule)
        return kept


# ---------------------------------------------------------------------------
# Principle 5 — AntarangaBahirangaResolver
# ---------------------------------------------------------------------------


class AntarangaBahirangaResolver:
    """
    Antaranga-bahiranga: internal (patient-specific, antaranga) rules take
  precedence over external (population-level, bahiranga) guidelines.
    """

    @staticmethod
    def resolve(
        fired: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Suppress bahiranga rules when a conflicting antaranga rule fires."""
        antaranga_ids = {
            r["rule_id"]
            for r in fired
            if r["scope"] == "antaranga" and r["action"] in ("block", "warn")
        }
        resolved: list[dict[str, Any]] = []
        for rule in fired:
            if (
                rule["scope"] == "bahiranga"
                and rule["action"] == "allow"
                and any(
                    antaranga_id in rule.get("overrides", []) or True
                    for antaranga_id in antaranga_ids
                )
            ):
                # Check if any antaranga rule explicitly conflicts on same drug topic
                conflicting = [
                    a for a in fired
                    if a["scope"] == "antaranga"
                    and a["action"] == "block"
                    and a["rule_id"] != rule["rule_id"]
                ]
                if conflicting:
                    rule = dict(rule)
                    rule["action"] = "warn"
                    rule["principle"] = "AntarangaBahiranga"
                    rule["paninian_concept"] = (
                        "antaranga (patient-specific) overrides bahiranga (population) guideline"
                    )
                    chain = list(rule.get("override_chain", []))
                    chain.extend(c["rule_id"] for c in conflicting)
                    rule["override_chain"] = chain
            resolved.append(rule)
        return resolved

    @staticmethod
    def pick_winner(
        antaranga: dict[str, Any],
        bahiranga: dict[str, Any],
    ) -> dict[str, Any]:
        """Antaranga always wins over bahiranga at equal action level."""
        winner = dict(antaranga)
        chain = list(winner.get("override_chain", []))
        chain.append(bahiranga["rule_id"])
        winner["override_chain"] = chain
        winner["principle"] = "AntarangaBahiranga"
        winner["paninian_concept"] = (
            "antaranga (internal/patient) overrides bahiranga (external/population)"
        )
        return winner


# ---------------------------------------------------------------------------
# RuleResult
# ---------------------------------------------------------------------------


@dataclass
class RuleResult:
    """Aggregated outcome of a full Paninian rule evaluation pass."""

    allowed: bool
    action: Action
    trace: list[dict[str, Any]]
    blocked: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    messages: list[str]
    requires_specialist: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "trace": self.trace,
            "blocked": self.blocked,
            "warnings": self.warnings,
            "messages": self.messages,
            "requires_specialist": self.requires_specialist,
        }


# ---------------------------------------------------------------------------
# Helper predicates for medical rules
# ---------------------------------------------------------------------------


def _gfr(patient: dict[str, Any]) -> float | None:
    value = patient.get("gfr") or patient.get("egfr")
    if value is None:
        return None
    return float(value)


def _age(patient: dict[str, Any], text: str) -> int | None:
    if "age" in patient and patient["age"] is not None:
        return int(patient["age"])
    match = re.search(r"\b(\d{1,3})\s*(?:-|\s)?year[- ]old\b", text)
    if match:
        return int(match.group(1))
    return None


_ACTIVE_PREGNANCY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d+\s+weeks?\s*['']?\s*(?:pregnant|gestation)\b", re.IGNORECASE),
    re.compile(r"\b(?:first|second|third|1st|2nd|3rd)\s+trimester\b", re.IGNORECASE),
    re.compile(r"\btrimester\b", re.IGNORECASE),
    re.compile(r"\bcurrently\s+pregnant\b", re.IGNORECASE),
    re.compile(r"\bis\s+pregnant\b", re.IGNORECASE),
    re.compile(
        r"\bgravida\b[^.]{0,100}\b(?:\d+\s+weeks?\s*['']?\s*(?:gestation|pregnant)|trimester)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:\d+\s+weeks?\s*['']?\s*(?:gestation|pregnant)|trimester)\b[^.]{0,100}\bgravida\b",
        re.IGNORECASE,
    ),
)

_HISTORICAL_PREGNANCY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:last|previous|prior|past)\s+pregnancy\b", re.IGNORECASE),
    re.compile(r"\bhistory\s+of\s+pregnancy\b", re.IGNORECASE),
    re.compile(
        r"(?:pregnancy|pregnant|gestation).{0,50}years?\s+ago"
        r"|years?\s+ago.{0,50}(?:pregnancy|pregnant|gestation)",
        re.IGNORECASE | re.DOTALL,
    ),
)

_GP_NOTATION_RE = re.compile(r"\bG\s*\d+\s*P\s*\d+\b", re.IGNORECASE)
_GP_SHORTHAND_RE = re.compile(r"\bGP\d+\b", re.IGNORECASE)


def detect_active_pregnancy(question_text: str) -> bool:
    """
    Detect whether the vignette describes a currently pregnant patient.

    Distinguishes active pregnancy (RULE_P001–P005 may fire) from historical
    mentions such as "last pregnancy was 3 years ago".
    """
    if not question_text or not question_text.strip():
        return False

    text = question_text.strip()

    for pattern in _ACTIVE_PREGNANCY_PATTERNS:
        if pattern.search(text):
            return True

    for sentence in re.split(r"[.!?]+", text):
        if (
            re.search(r"\b\d{1,3}[- ]year[- ]old\b", sentence, re.IGNORECASE)
            and re.search(r"\bwoman\b", sentence, re.IGNORECASE)
            and re.search(r"\bpregnant\b", sentence, re.IGNORECASE)
            and not re.search(
                r"\b(?:last|previous|prior|past)\s+pregnancy\b", sentence, re.IGNORECASE
            )
        ):
            return True

    for pattern in _HISTORICAL_PREGNANCY_PATTERNS:
        if pattern.search(text):
            return False

    if _GP_NOTATION_RE.search(text) or _GP_SHORTHAND_RE.search(text):
        return False

    if re.search(r"\b(?:pregnant|pregnancy|gestation|gravid)\b", text, re.IGNORECASE):
        return False

    return False


def _has_active_pregnancy(patient: dict[str, Any]) -> bool:
    """True when anuvrtti patient context marks an active pregnancy."""
    return patient.get("active_pregnancy") is True


def _is_pregnant(patient: dict[str, Any], text: str) -> bool:
    """Pregnancy-sensitive rules use active_pregnancy from vignette, not chunk keywords."""
    return _has_active_pregnancy(patient)


def _has_condition(patient: dict[str, Any], text: str, terms: list[str]) -> bool:
    conditions = patient.get("conditions", [])
    if isinstance(conditions, str):
        conditions = [conditions]
    condition_text = " ".join(str(c) for c in conditions).lower()
    return _contains_any(condition_text, terms) or _contains_any(text, terms)


def _has_medication(patient: dict[str, Any], text: str, terms: list[str]) -> bool:
    medications = patient.get("medications", [])
    if isinstance(medications, str):
        medications = [medications]
    med_text = " ".join(str(m) for m in medications).lower()
    return _mentions_drug(med_text, terms) or _mentions_drug(text, terms)


def _has_allergy(patient: dict[str, Any], text: str, terms: list[str]) -> bool:
    allergies = patient.get("allergies", [])
    if isinstance(allergies, str):
        allergies = [allergies]
    allergy_text = " ".join(str(a) for a in allergies).lower()
    return _contains_any(allergy_text, terms) or _contains_any(text, terms)


# ---------------------------------------------------------------------------
# 30+ Medical rules
# ---------------------------------------------------------------------------


def _build_medical_rules() -> list[UtsargaApavadaRule]:
    """Construct the canonical MEDICAL_RULES registry."""

    def renal_failure(ctx: RuleContext) -> bool:
        gfr = _gfr(ctx.patient)
        return (
            (gfr is not None and gfr < 30)
            or _has_condition(ctx.patient, ctx.combined_text, ["renal failure", "ckd stage 4", "ckd stage 5", "esrd"])
        )

    def renal_impairment_moderate(ctx: RuleContext) -> bool:
        gfr = _gfr(ctx.patient)
        return (
            (gfr is not None and gfr < 45)
            or _has_condition(ctx.patient, ctx.combined_text, ["renal impairment", "ckd", "renal failure"])
        )

    rules: list[UtsargaApavadaRule] = [
        # --- Drug contraindications (Nitya, Antaranga) ---
        UtsargaApavadaRule(
            rule_id="RULE_M001",
            description="NSAIDs contraindicated in renal failure (GFR < 30)",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: _mentions_drug(ctx.combined_text, NSAID_TERMS) and renal_failure(ctx),
            action="block",
            message="NSAIDs contraindicated: patient has renal failure (GFR < 30)",
            paninian_concept="nitya antaranga apavada (mandatory patient-specific exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_M002",
            description="Metformin contraindicated in renal failure (GFR < 45)",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: _mentions_drug(ctx.combined_text, ["metformin"]) and renal_impairment_moderate(ctx),
            action="block",
            message="Metformin contraindicated: renal impairment (GFR < 45)",
            paninian_concept="nitya antaranga apavada (renal safety exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_M003",
            description="ACE inhibitors contraindicated in bilateral renal artery stenosis",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ACE_TERMS)
                and _has_condition(
                    ctx.patient, ctx.combined_text,
                    ["bilateral renal artery stenosis", "renal artery stenosis"],
                )
            ),
            action="block",
            message="ACE inhibitors contraindicated: bilateral renal artery stenosis",
            paninian_concept="nitya antaranga apavada (anatomical contraindication)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_M004",
            description="Beta blockers contraindicated in acute decompensated asthma",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, BETA_BLOCKER_TERMS)
                and _has_condition(
                    ctx.patient, ctx.combined_text,
                    ["acute asthma", "decompensated asthma", "status asthmaticus", "severe asthma exacerbation"],
                )
            ),
            action="block",
            message="Beta blockers contraindicated: acute decompensated asthma",
            paninian_concept="nitya antaranga apavada (respiratory contraindication)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_M005",
            description="Fluoroquinolones contraindicated under age 18",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, FLUOROQUINOLONE_TERMS)
                and (_age(ctx.patient, ctx.combined_text) is not None and _age(ctx.patient, ctx.combined_text) < 18)
            ),
            action="block",
            message="Fluoroquinolones contraindicated: patient under 18 (cartilage damage risk)",
            paninian_concept="nitya antaranga apavada (pediatric exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_M006",
            description="Tetracyclines contraindicated in pregnancy",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, TETRACYCLINE_TERMS)
                and _is_pregnant(ctx.patient, ctx.combined_text)
            ),
            action="block",
            message="Tetracyclines contraindicated in pregnancy (fetal bone/teeth toxicity)",
            paninian_concept="nitya antaranga apavada (pregnancy exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_M007",
            description="Warfarin — flag NSAID co-prescription (bleeding risk)",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ["warfarin"])
                and (
                    _has_medication(ctx.patient, ctx.combined_text, NSAID_TERMS)
                    or _mentions_drug(ctx.combined_text, NSAID_TERMS)
                )
            ),
            action="warn",
            message="Warfarin with NSAIDs: elevated bleeding risk — review co-prescription",
            paninian_concept="nitya antaranga (drug-drug interaction exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_M008",
            description="MAO inhibitors block SSRIs (serotonin syndrome)",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                (_has_medication(ctx.patient, ctx.combined_text, MAO_TERMS) and _mentions_drug(ctx.combined_text, SSRI_TERMS))
                or (_mentions_drug(ctx.combined_text, MAO_TERMS) and _has_medication(ctx.patient, ctx.combined_text, SSRI_TERMS))
            ),
            action="block",
            message="MAO inhibitor + SSRI combination blocked: serotonin syndrome risk",
            paninian_concept="nitya antaranga apavada (lethal interaction exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_M009",
            description="Penicillin contraindicated with documented penicillin allergy",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, PENICILLIN_TERMS)
                and _has_allergy(ctx.patient, ctx.combined_text, ["penicillin", "beta-lactam", "amoxicillin"])
            ),
            action="block",
            message="Penicillin contraindicated: documented penicillin allergy",
            paninian_concept="nitya antaranga apavada (allergy exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_M010",
            description="Potassium-sparing diuretics contraindicated in hyperkalemia",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ["spironolactone", "eplerenone", "amiloride", "triamterene"])
                and _has_condition(ctx.patient, ctx.combined_text, ["hyperkalemia", "high potassium"])
            ),
            action="block",
            message="Potassium-sparing diuretics contraindicated: hyperkalemia present",
            paninian_concept="nitya antaranga apavada (electrolyte exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_M011",
            description="Potassium supplements with ACE inhibitors — hyperkalemia risk",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ["potassium", "kcl", "potassium chloride"])
                and (
                    _has_medication(ctx.patient, ctx.combined_text, ACE_TERMS)
                    or _mentions_drug(ctx.combined_text, ACE_TERMS)
                )
            ),
            action="warn",
            message="Potassium supplementation with ACE inhibitor: monitor for hyperkalemia",
            paninian_concept="nitya antaranga (interaction exception)",
        ),

        # --- Dosage safety (Nitya, Bahiranga) ---
        UtsargaApavadaRule(
            rule_id="RULE_D001",
            description="Paracetamol max 4g/day adults; 2g/day if liver disease",
            severity="nitya",
            scope="bahiranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ["paracetamol", "acetaminophen"])
                and (
                    _contains_any(ctx.combined_text, ["5g", "6g", "4000mg", "4 g/day", "exceed", "overdose"])
                    or _has_condition(ctx.patient, ctx.combined_text, ["liver disease", "cirrhosis", "hepatic"])
                )
            ),
            action="warn",
            message="Paracetamol dosing: max 4g/day (2g/day if liver disease)",
            paninian_concept="nitya bahiranga utsarga (population dosing limit)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_D002",
            description="Ibuprofen max 2.4g/day; reduce in elderly (>65)",
            severity="nitya",
            scope="bahiranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ["ibuprofen"])
                and (
                    _contains_any(ctx.combined_text, ["3g", "3200mg", "exceed", "high dose"])
                    or (_age(ctx.patient, ctx.combined_text) is not None and _age(ctx.patient, ctx.combined_text) > 65)
                )
            ),
            action="warn",
            message="Ibuprofen: max 2.4g/day; reduce dose in elderly (>65)",
            paninian_concept="nitya bahiranga utsarga (age-adjusted dosing)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_D003",
            description="Digoxin — narrow therapeutic index; check renal function",
            severity="nitya",
            scope="bahiranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ["digoxin"])
                and (
                    renal_impairment_moderate(ctx)
                    or _contains_any(ctx.combined_text, ["loading dose", "high dose", "toxicity"])
                )
            ),
            action="warn",
            message="Digoxin: narrow therapeutic index — verify renal function and levels",
            paninian_concept="nitya bahiranga utsarga (therapeutic index guideline)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_D004",
            description="Lithium — flag dehydration, NSAIDs, ACE inhibitor combinations",
            severity="nitya",
            scope="bahiranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ["lithium"])
                and (
                    _has_medication(ctx.patient, ctx.combined_text, NSAID_TERMS + ACE_TERMS)
                    or _contains_any(ctx.combined_text, ["dehydration", "nsaid", "ace inhibitor"])
                )
            ),
            action="warn",
            message="Lithium interaction risk: dehydration, NSAIDs, or ACE inhibitors increase toxicity",
            paninian_concept="nitya bahiranga utsarga (interaction guideline)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_D005",
            description="Opioids — caution in respiratory depression / COPD",
            severity="nitya",
            scope="bahiranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ["morphine", "oxycodone", "fentanyl", "hydromorphone", "opioid"])
                and _has_condition(ctx.patient, ctx.combined_text, ["copd", "respiratory depression", "sleep apnea"])
            ),
            action="warn",
            message="Opioids: use caution in COPD or respiratory depression",
            paninian_concept="nitya bahiranga utsarga (respiratory safety guideline)",
        ),

        # --- Pregnancy rules (Nitya, Antaranga) ---
        UtsargaApavadaRule(
            rule_id="RULE_P001",
            description="No ACE inhibitors in pregnancy (teratogenic)",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ACE_TERMS)
                and _has_active_pregnancy(ctx.patient)
            ),
            action="block",
            message="ACE inhibitors contraindicated in pregnancy (teratogenic)",
            paninian_concept="nitya antaranga apavada (pregnancy overrides all bahiranga indications)",
            is_exception=True,
            overrides=["RULE_D003"],
        ),
        UtsargaApavadaRule(
            rule_id="RULE_P002",
            description="No statins in pregnancy (teratogenic)",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, STATIN_TERMS)
                and _has_active_pregnancy(ctx.patient)
            ),
            action="block",
            message="Statins contraindicated in pregnancy (teratogenic)",
            paninian_concept="nitya antaranga apavada (pregnancy exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_P003",
            description="No warfarin in first trimester",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ["warfarin"])
                and _has_active_pregnancy(ctx.patient)
                and _contains_any(ctx.combined_text, ["first trimester", "1st trimester", "week 8", "week 10"])
            ),
            action="block",
            message="Warfarin contraindicated in first trimester (teratogenic)",
            paninian_concept="nitya antaranga apavada (trimester-specific exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_P004",
            description="Flag category D/X drugs if pregnancy mentioned",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _has_active_pregnancy(ctx.patient)
                and _mentions_drug(ctx.combined_text, CATEGORY_DX_TERMS)
            ),
            action="block",
            message="Category D/X drug flagged: pregnancy present — seek safer alternative",
            paninian_concept="nitya antaranga apavada (FDA pregnancy category exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_P005",
            description="Alcohol contraindicated in pregnancy",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _has_active_pregnancy(ctx.patient)
                and _contains_any(ctx.combined_text, ["alcohol", "ethanol", "drink"])
            ),
            action="block",
            message="Alcohol contraindicated in pregnancy (fetal alcohol spectrum risk)",
            paninian_concept="nitya antaranga apavada (pregnancy behavioral exception)",
        ),

        # --- Age-specific rules (Nitya, Antaranga) ---
        UtsargaApavadaRule(
            rule_id="RULE_A001",
            description="Aspirin contraindicated under 16 (Reye syndrome)",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ["aspirin"])
                and (_age(ctx.patient, ctx.combined_text) is not None and _age(ctx.patient, ctx.combined_text) < 16)
            ),
            action="block",
            message="Aspirin contraindicated under age 16 (Reye syndrome risk)",
            paninian_concept="nitya antaranga apavada (pediatric exception)",
            is_exception=True,
            overrides=["RULE_D002"],
        ),
        UtsargaApavadaRule(
            rule_id="RULE_A002",
            description="Benzodiazepines — flag in elderly (fall risk, cognitive)",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, BENZO_TERMS)
                and (_age(ctx.patient, ctx.combined_text) is not None and _age(ctx.patient, ctx.combined_text) > 65)
            ),
            action="warn",
            message="Benzodiazepines in elderly: elevated fall risk and cognitive impairment",
            paninian_concept="nitya antaranga (geriatric safety exception)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_A003",
            description="Aminoglycosides — reduce dose if age > 65",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, AMINOGLYCOSIDE_TERMS)
                and (_age(ctx.patient, ctx.combined_text) is not None and _age(ctx.patient, ctx.combined_text) > 65)
            ),
            action="warn",
            message="Aminoglycosides in elderly: reduce dose (declining renal clearance)",
            paninian_concept="nitya antaranga apavada (age overrides bahiranga dosing)",
            is_exception=True,
            overrides=["RULE_D002"],
        ),
        UtsargaApavadaRule(
            rule_id="RULE_A004",
            description="Codeine contraindicated under age 12",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _mentions_drug(ctx.combined_text, ["codeine"])
                and (_age(ctx.patient, ctx.combined_text) is not None and _age(ctx.patient, ctx.combined_text) < 12)
            ),
            action="block",
            message="Codeine contraindicated under age 12 (respiratory depression risk)",
            paninian_concept="nitya antaranga apavada (pediatric opioid exception)",
        ),

        # --- Diagnostic rules (Anitya, Bahiranga) ---
        UtsargaApavadaRule(
            rule_id="RULE_DX001",
            description="Chest pain + diaphoresis → cardiac workup required",
            severity="anitya",
            scope="bahiranga",
            condition=lambda ctx: (
                _contains_any(ctx.combined_text, ["chest pain"])
                and _contains_any(ctx.combined_text, ["diaphoresis", "sweating", "clammy"])
            ),
            action="warn",
            message="Chest pain with diaphoresis: urgent cardiac workup required",
            paninian_concept="anitya bahiranga utsarga (diagnostic guideline)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_DX002",
            description="Fever + stiff neck + photophobia → meningitis protocol",
            severity="anitya",
            scope="bahiranga",
            condition=lambda ctx: (
                _contains_any(ctx.combined_text, ["fever"])
                and _contains_any(ctx.combined_text, ["stiff neck", "neck stiffness", "meningismus"])
                and _contains_any(ctx.combined_text, ["photophobia", "light sensitivity"])
            ),
            action="warn",
            message="Fever, stiff neck, photophobia: activate meningitis protocol",
            paninian_concept="anitya bahiranga utsarga (diagnostic triad guideline)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_DX003",
            description="Unilateral leg swelling + tachycardia → DVT/PE workup",
            severity="anitya",
            scope="bahiranga",
            condition=lambda ctx: (
                _contains_any(ctx.combined_text, ["leg swelling", "unilateral swelling", "calf swelling"])
                and _contains_any(ctx.combined_text, ["tachycardia", "heart rate", "hr 1"])
            ),
            action="warn",
            message="Unilateral leg swelling with tachycardia: evaluate for DVT/PE",
            paninian_concept="anitya bahiranga utsarga (VTE workup guideline)",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_DX004",
            description="Thunderclap headache → subarachnoid hemorrhage workup",
            severity="anitya",
            scope="bahiranga",
            condition=lambda ctx: _contains_any(
                ctx.combined_text,
                ["thunderclap headache", "worst headache", "sudden severe headache"],
            ),
            action="warn",
            message="Thunderclap headache: urgent subarachnoid hemorrhage workup required",
            paninian_concept="anitya bahiranga utsarga (neurologic red flag)",
        ),

        # --- Guideline conflict / meta (Paribhasha) ---
        UtsargaApavadaRule(
            rule_id="RULE_GC001",
            description="Drug indicated AND contraindicated → block + explain",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: False,  # evaluated post-hoc by engine
            action="block",
            message="Conflicting indication and contraindication detected for same therapy",
            principle="Paribhasha",
            paninian_concept="paribhasha: simultaneous indication + contraindication → block",
            requires_specialist=True,
        ),
        UtsargaApavadaRule(
            rule_id="RULE_GC002",
            description="Two guidelines conflict → return both with evidence level",
            severity="anitya",
            scope="bahiranga",
            condition=lambda ctx: False,  # evaluated post-hoc by engine
            action="warn",
            message="Conflicting guidelines detected — present both with evidence levels",
            principle="Paribhasha",
            paninian_concept="paribhasha: preserve both conflicting bahiranga guidelines",
        ),
        UtsargaApavadaRule(
            rule_id="RULE_GC003",
            description="Specialist override flag for high-risk decisions",
            severity="nitya",
            scope="antaranga",
            condition=lambda ctx: (
                _contains_any(ctx.combined_text, ["chemotherapy", "immunosuppressant", "anticoagulation"])
                and _has_condition(ctx.patient, ctx.combined_text, ["malignancy", "transplant", "pregnancy"])
            ),
            action="warn",
            message="High-risk therapy in complex patient — specialist review recommended",
            principle="Paribhasha",
            paninian_concept="paribhasha: specialist gate for antaranga high-risk cases",
            requires_specialist=True,
        ),
    ]

    return rules


MEDICAL_RULES: list[UtsargaApavadaRule] = _build_medical_rules()


# ---------------------------------------------------------------------------
# PaniniRuleEngine
# ---------------------------------------------------------------------------


class PaniniRuleEngine:
    """
    Orchestrates all five Paninian principles over a registry of medical rules.

    Every call to :meth:`evaluate` appends to the trace log — the primary
    research artifact showing which sutra-level principle fired and why.
    """

    def __init__(self) -> None:
        self._rules: list[UtsargaApavadaRule] = list(MEDICAL_RULES)
        self._anuvrtti = AnuvrttiBag()
        self._paribhasha = Paribhasha()
        self._nitya_anitya = NityaAnityaClassifier()
        self._scope_resolver = AntarangaBahirangaResolver()
        self._trace: list[dict[str, Any]] = []
        self._last_result: RuleResult | None = None

    def add_rule(self, rule: UtsargaApavadaRule) -> None:
        """Register an additional rule beyond the built-in MEDICAL_RULES set."""
        self._rules.append(rule)

    def evaluate(
        self,
        query: str,
        retrieved_context: str,
        patient_context: dict[str, Any] | None = None,
    ) -> RuleResult:
        """
        Evaluate all rules against query + retrieval + inherited patient context.

        Applies, in order:
        1. Anuvrtti context merge
        2. Per-rule firing
        3. Utsarga-apavada exception overrides
        4. Paribhasha conflict resolution (incl. GC001/GC002 post-hoc)
        5. Nitya-anitya suppression
        6. Antaranga-bahiranga scope resolution
        """
        patient = self._anuvrtti.merge(patient_context or {})
        if patient_context is not None and "active_pregnancy" in patient_context:
            patient["active_pregnancy"] = bool(patient_context["active_pregnancy"])
        else:
            patient["active_pregnancy"] = detect_active_pregnancy(query)
        patient["pregnant"] = patient["active_pregnancy"]
        patient["pregnancy"] = patient["active_pregnancy"]

        context = RuleContext(
            query=query,
            retrieved_context=retrieved_context,
            patient=patient,
        )

        fired: list[dict[str, Any]] = []
        for rule in self._rules:
            if rule.rule_id in ("RULE_GC001", "RULE_GC002"):
                continue
            hit = rule.evaluate(context)
            if hit:
                fired.append(hit)

        fired = self._apply_utsarga_apavada(fired)
        fired = self._apply_paribhasha_conflicts(fired)
        fired = self._evaluate_gc_rules(fired)
        fired = self._nitya_anitya.filter_suppressed(fired)
        fired = self._scope_resolver.resolve(fired)

        blocked = [r for r in fired if r["action"] == "block"]
        warnings = [r for r in fired if r["action"] == "warn"]
        allowed = len(blocked) == 0

        if blocked:
            final_action: Action = "block"
        elif warnings:
            final_action = "warn"
        else:
            final_action = "allow"

        messages = [r["message"] for r in fired]
        requires_specialist = any(r.get("requires_specialist") for r in fired)

        result = RuleResult(
            allowed=allowed,
            action=final_action,
            trace=fired,
            blocked=blocked,
            warnings=warnings,
            messages=messages,
            requires_specialist=requires_specialist,
        )

        self._trace.extend(fired)
        self._last_result = result
        return result

    def _apply_utsarga_apavada(
        self,
        fired: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Exception rules (apavada) suppress overridden general rules (utsarga)."""
        suppressed: set[str] = set()
        for rule in fired:
            if rule.get("is_exception"):
                for overridden_id in rule.get("overrides", []):
                    suppressed.add(overridden_id)
                    chain = list(rule.get("override_chain", []))
                    chain.append(overridden_id)
                    rule["override_chain"] = chain
                    rule["paninian_concept"] = (
                        f"apavada: {rule['rule_id']} overrides {overridden_id}"
                    )
        return [r for r in fired if r["rule_id"] not in suppressed]

    def _apply_paribhasha_conflicts(
        self,
        fired: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Pairwise paribhasha resolution among same-action conflicts."""
        if len(fired) < 2:
            return fired

        resolved = list(fired)
        dosage_rules = [r for r in resolved if r["rule_id"].startswith("RULE_D")]
        age_rules = [r for r in resolved if r["rule_id"].startswith("RULE_A")]

        for d_rule in dosage_rules:
            for a_rule in age_rules:
                if d_rule["action"] == a_rule["action"] == "warn":
                    winner = self._paribhasha.dosage_vs_age(d_rule, a_rule)
                    resolved = [r for r in resolved if r["rule_id"] != d_rule["rule_id"]]
                    resolved.append(winner)

        for i, rule_a in enumerate(resolved):
            for rule_b in resolved[i + 1:]:
                if (
                    rule_a["action"] != rule_b["action"]
                    and rule_a["rule_id"] != rule_b["rule_id"]
                ):
                    winner = self._paribhasha.resolve_conflict(rule_a, rule_b)
                    if NityaAnityaClassifier.is_nitya(rule_a) and NityaAnityaClassifier.is_anitya(rule_b):
                        winner = self._paribhasha.nitya_blocks_anitya_override(rule_a, rule_b)
                    resolved = [
                        r for r in resolved
                        if r["rule_id"] not in {rule_a["rule_id"], rule_b["rule_id"]}
                    ]
                    resolved.append(winner)

        return resolved

    def _evaluate_gc_rules(
        self,
        fired: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Post-hoc paribhasha meta-rules for guideline conflicts."""
        result = list(fired)
        blocks = [r for r in fired if r["action"] == "block"]
        warns = [r for r in fired if r["action"] == "warn"]

        # GC001: same drug space has both block and warn from different families
        block_drugs = {r["rule_id"] for r in blocks}
        if blocks and warns:
            for gc in self._rules:
                if gc.rule_id == "RULE_GC001":
                    hit = gc.evaluate(
                        RuleContext(query="", retrieved_context="", patient={})
                    )
                    if hit:
                        hit["fired"] = True
                        hit["message"] = (
                            f"Indication/contraindication conflict: "
                            f"blocks={list(block_drugs)}, warnings={[w['rule_id'] for w in warns]}"
                        )
                        result.append(hit)

        # GC002: multiple bahiranga warnings without blocks
        bahiranga_warns = [r for r in warns if r["scope"] == "bahiranga"]
        if len(bahiranga_warns) >= 2 and not blocks:
            for gc in self._rules:
                if gc.rule_id == "RULE_GC002":
                    hit = gc.evaluate(
                        RuleContext(query="", retrieved_context="", patient={})
                    )
                    if hit:
                        hit["fired"] = True
                        hit["message"] = (
                            "Conflicting bahiranga guidelines: "
                            + ", ".join(r["rule_id"] for r in bahiranga_warns)
                        )
                        result.append(hit)

        return result

    def set_patient_context(self, patient_dict: dict[str, Any]) -> None:
        """Set inherited anuvrtti patient context for all subsequent evaluations."""
        self._anuvrtti.set_context(patient_dict)

    def get_trace(self) -> list[dict[str, Any]]:
        """Full audit log of every rule that fired across all evaluations."""
        return list(self._trace)

    def get_blocked_rules(self) -> list[dict[str, Any]]:
        """Rules that blocked content in the most recent evaluation."""
        if self._last_result is None:
            return []
        return list(self._last_result.blocked)

    def get_warnings(self) -> list[dict[str, Any]]:
        """Rules that warned but allowed in the most recent evaluation."""
        if self._last_result is None:
            return []
        return list(self._last_result.warnings)

    def summary(self) -> dict[str, Any]:
        """
        Aggregate counts for paper metrics.

        Returns fired / blocked / warned counts and per-principle breakdown.
        """
        fired = self._trace
        by_principle: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        for entry in fired:
            by_principle[entry.get("principle", "UtsargaApavada")] = (
                by_principle.get(entry.get("principle", "UtsargaApavada"), 0) + 1
            )
            by_severity[entry.get("severity", "nitya")] = (
                by_severity.get(entry.get("severity", "nitya"), 0) + 1
            )
            by_scope[entry.get("scope", "bahiranga")] = (
                by_scope.get(entry.get("scope", "bahiranga"), 0) + 1
            )

        return {
            "total_fired": len(fired),
            "blocked": sum(1 for e in fired if e["action"] == "block"),
            "warned": sum(1 for e in fired if e["action"] == "warn"),
            "allowed": sum(1 for e in fired if e["action"] == "allow"),
            "by_principle": by_principle,
            "by_severity": by_severity,
            "by_scope": by_scope,
            "unique_rules_fired": len({e["rule_id"] for e in fired}),
        }


def test_active_pregnancy_detection() -> None:
    """
    Regression test for spurious pregnancy rule activation.

    dev_822 — historical pregnancy mention only (must NOT activate rules).
    dev_678 — active 30-week gestation eclampsia case (must activate rules).
    """
    cases = {
        "dev_822_87011fe2cb (false positive — historical pregnancy)": (
            "A 32-year-old GP2 presents to an outpatient clinic for a routine gynecologic "
            "examination. The patient appears well, although she mentions that during the "
            "past 6 months she has noticed small amounts of vaginal bleeding in the middle "
            "of her menstrual cycles. Her last pregnancy was 3 years ago. Her subsequent "
            "menstrual cycles have been regular, lasting about 2–3 days. She does not smoke, "
            "drinks alcohol occasionally, and has never used illicit drugs. An ultrasound "
            "reveals a fleshy mass with a pedunculated stalk deep in the cervical canal."
        ),
        "dev_678_4be418bdc9 (true positive — active eclampsia)": (
            "A 27-year-old woman is brought to the emergency department by her coworker after "
            "having a generalized seizure at work. Her coworker reports that she is at "
            "30 weeks' gestation and has mentioned headache and right upper quadrant pain "
            "earlier that day. Her temperature is 37°C (98.6°F), pulse is 91/min, and blood "
            "pressure is 170/102 mm Hg."
        ),
    }

    # Context that previously triggered RULE_P005 on dev_822 via combined_text.
    alcohol_context = (
        "Alcohol use during pregnancy causes fetal alcohol spectrum disorder. "
        "Patients who drink alcohol should be counseled to abstain."
    )

    print("\n" + "=" * 72)
    print(" Active pregnancy detection — regression test")
    print("=" * 72)

    for label, question in cases.items():
        active = detect_active_pregnancy(question)
        engine = PaniniRuleEngine()
        result = engine.evaluate(
            query=question,
            retrieved_context=alcohol_context,
        )
        pregnancy_rules = [
            entry["rule_id"]
            for entry in result.trace
            if entry["rule_id"].startswith("RULE_P00")
        ]

        print(f"\n{label}")
        print(f"  detect_active_pregnancy() : {active}")
        print(f"  Pregnancy rules fired     : {pregnancy_rules or '(none)'}")
        expected_active = "dev_678" in label
        status = "PASS" if active == expected_active else "FAIL"
        rules_ok = (len(pregnancy_rules) > 0) == expected_active
        rules_status = "PASS" if rules_ok else "FAIL"
        print(f"  Detector expected         : {expected_active} [{status}]")
        print(f"  Rules expected            : {'fire' if expected_active else 'silent'} [{rules_status}]")


def _main() -> None:
    """Demonstrate engine with a renal-failure + NSAID scenario."""
    import argparse

    parser = argparse.ArgumentParser(description="Paninian rule engine demo")
    parser.add_argument(
        "--test-pregnancy",
        action="store_true",
        help="Run active-pregnancy regression test (dev_822 vs dev_678)",
    )
    args = parser.parse_args()

    if args.test_pregnancy:
        test_active_pregnancy_detection()
        return

    engine = PaniniRuleEngine()
    engine.set_patient_context({
        "age": 58,
        "conditions": ["renal failure", "CKD stage 4"],
        "gfr": 22,
        "medications": ["lisinopril"],
        "allergies": [],
    })

    result = engine.evaluate(
        query="Can we prescribe ibuprofen 600mg TID for this patient's arthritis pain?",
        retrieved_context="NSAIDs are first-line for osteoarthritis pain management.",
        patient_context={"gfr": 22},
    )

    print(f"Allowed: {result.allowed}  Action: {result.action}")
    print(f"Rules fired: {len(result.trace)}")
    for entry in result.trace:
        print(f"  [{entry['rule_id']}] {entry['action']}: {entry['message']}")
    print("Summary:", engine.summary())


if __name__ == "__main__":
    _main()
