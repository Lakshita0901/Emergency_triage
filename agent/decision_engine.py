"""Deterministic uncertainty calculation and routing decision engine for TriageFlow AI.

NO LLM calls. All routing is pure deterministic Python per spec.
"""
from typing import Any, TypedDict

CRITICAL_FIELDS: list[str] = [
    "oxygen_saturation",
    "heart_rate",
    "respiratory_rate",
    "systolic_bp",
    "chest_pain",
    "shortness_of_breath",
]

_TOTAL_CRITICAL = len(CRITICAL_FIELDS)  # 6


class UncertaintyResult(TypedDict):
    uncertainty: float
    missing_fields: list[str]


class RoutingResult(TypedDict):
    routing: str | None  # DISCHARGE | PRIMARY_CARE | EMERGENCY | ESCALATE_TO_HUMAN | None
    reason: str


def calculate_uncertainty(state: dict[str, Any]) -> UncertaintyResult:
    """Compute uncertainty score.

    Formula:
        uncertainty = (missing_critical / total_critical)
                      + min(0.15 * num_contradictions, 0.30)
        capped at 1.0

    Args:
        state: patient state dict; must contain optional keys for each critical field
               and optionally "contradictions" (list[str]).

    Returns:
        UncertaintyResult with 'uncertainty' (float) and 'missing_fields' (list[str]).
    """
    missing: list[str] = [
        f for f in CRITICAL_FIELDS if state.get(f) is None
    ]
    contradiction_list: list[str] = state.get("contradictions") or []
    num_contradictions = len(contradiction_list)

    base = len(missing) / _TOTAL_CRITICAL
    contradiction_penalty = min(0.15 * num_contradictions, 0.30)
    uncertainty = min(base + contradiction_penalty, 1.0)

    return UncertaintyResult(uncertainty=round(uncertainty, 4), missing_fields=missing)


def decide_routing(state: dict[str, Any]) -> RoutingResult:
    """Apply deterministic routing rules.

    Rules (evaluated in order):
        1. risk≥7 AND (uncertainty≥0.50 OR contradictions exist OR critical fields missing)
               → ESCALATE_TO_HUMAN
        2. risk≥7 otherwise → EMERGENCY
        3. 4≤risk≤6 → PRIMARY_CARE
        4. risk<4 AND uncertainty≤0.25 → DISCHARGE
        5. else → None (ask more questions)

    Args:
        state: must contain "risk_score" (int), "uncertainty" (float),
               "contradictions" (list[str]), "missing_critical_fields" (list[str]).

    Returns:
        RoutingResult with 'routing' (str|None) and 'reason' (str).
    """
    risk: int = state.get("risk_score", 0)
    uncertainty: float = state.get("uncertainty", 0.0)
    contradictions: list[str] = state.get("contradictions") or []
    missing: list[str] = state.get("missing_critical_fields") or []

    if risk >= 7:
        if uncertainty >= 0.50:
            return RoutingResult(
                routing="ESCALATE_TO_HUMAN",
                reason=f"High risk (score={risk}) with high uncertainty ({uncertainty:.2f}≥0.50). Human review required.",
            )
        if contradictions:
            return RoutingResult(
                routing="ESCALATE_TO_HUMAN",
                reason=f"High risk (score={risk}) with {len(contradictions)} contradiction(s): {'; '.join(contradictions)}.",
            )
        if missing:
            return RoutingResult(
                routing="ESCALATE_TO_HUMAN",
                reason=f"High risk (score={risk}) with missing critical field(s): {', '.join(missing)}.",
            )
        return RoutingResult(
            routing="EMERGENCY",
            reason=f"High risk score ({risk}≥7) with sufficient data and no contradictions.",
        )

    if 4 <= risk <= 6:
        return RoutingResult(
            routing="PRIMARY_CARE",
            reason=f"Moderate risk score ({risk}). Recommend primary care follow-up.",
        )

    if risk < 4 and uncertainty <= 0.25:
        return RoutingResult(
            routing="DISCHARGE",
            reason=f"Low risk score ({risk}<4) with low uncertainty ({uncertainty:.2f}≤0.25). Safe to discharge.",
        )

    return RoutingResult(
        routing=None,
        reason=f"Risk={risk}, uncertainty={uncertainty:.2f}. Insufficient data to route — collect more information.",
    )
