"""
core/gate.py — Neuro-symbolic validation gate G(x).

Three independent constraint components:
  c_schema   — rigid JSON field completeness
  c_ontology — at least one SNOMED CT term present
  c_logic    — AJCC 8th Edition clinical-logic consistency

G(x) = c_schema AND c_ontology AND c_logic
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GateResult:
    schema:     bool = False
    ontology:   bool = False
    logic:      bool = False
    violations: list = field(default_factory=list)

    @property
    def full(self) -> bool:
        return self.schema and self.ontology and self.logic

    @property
    def schema_only(self) -> bool:
        return self.schema

    @property
    def schema_onto(self) -> bool:
        return self.schema and self.ontology

    def label(self) -> str:
        if self.full:         return "C3_full"
        if self.schema_onto:  return "C2_schema_onto"
        if self.schema_only:  return "C1_schema"
        return "C0_none"


def check_schema(parsed_json) -> bool:
    if not isinstance(parsed_json, dict):
        return False
    notes = parsed_json.get("notes", [])
    if not notes or not isinstance(notes[0], dict):
        return False
    stg = notes[0].get("staging", {})
    required = ("T", "N", "M", "stage_group")
    return all(
        stg.get(k) not in (None, "", "Unknown", "unknown")
        for k in required
    )


def check_ontology(snomed_codes: list) -> bool:
    return isinstance(snomed_codes, list) and len(snomed_codes) > 0


def check_logic(parsed_json) -> tuple[bool, list]:
    """
    Returns (passes: bool, violations: list[str]).
    Checks AJCC 8th Edition clinical-logic constraints:
      - M1 staging must map to Stage IV
      - Stage group must be non-empty if T/N/M are present
    """
    if not parsed_json or "notes" not in parsed_json:
        return False, ["missing_notes_key"]
    violations = []
    for idx, note in enumerate(parsed_json.get("notes", [])):
        stg   = note.get("staging", {}) if isinstance(note, dict) else {}
        m     = str(stg.get("M", "")).upper()
        group = str(stg.get("stage_group", "")).upper()
        t     = str(stg.get("T", "")).upper()
        n     = str(stg.get("N", "")).upper()
        if "M1" in m and "IV" not in group:
            violations.append(f"visit_{idx}:M1_without_StageIV (group='{group}')")
        if t and n and m and not group:
            violations.append(f"visit_{idx}:missing_stage_group")
    return len(violations) == 0, violations


def evaluate(parsed_json, snomed_codes: list) -> GateResult:
    """Evaluates all three gate components and returns a GateResult."""
    c1          = check_schema(parsed_json)
    c2          = check_ontology(snomed_codes)
    passes, vio = check_logic(parsed_json)
    return GateResult(
        schema=c1, ontology=c2, logic=passes, violations=vio
    )


def gate_label_to_adapter(gate_label: str) -> str:
    """Maps gate checkpoint label to adapter name from config."""
    return {
        "C0_none":        "adapter_A_ungated",
        "C1_schema":      "adapter_B_schema",
        "C2_schema_onto": "adapter_C_schema_onto",
        "C3_full":        "adapter_D_full_norag",   # overridden to E for RAG condition
    }.get(gate_label, "adapter_A_ungated")
