"""
AI-DRISHTI Physics Engine — SRP Speed Optimizer & Rod Floating Detector
--------------------------------------------------------------------------
Directly addresses Problem Statement requirements:
  - "Continuously optimize SRP operation by adjusting stroke speed and SPM"
  - "Detect rod floating and minimize impact loading"

THE KEY FINDING THIS MODULE IS BUILT ON (from real srp_data.pdf):
  The field's actual current practice INCREASES SPM as viscosity rises
  (3.0 -> 4.5 SPM as viscosity goes 4,200 -> 11,800 cP, fit below,
  R^2=0.86). This is backwards: pushing the pump faster into thicker,
  slower-to-flow oil reduces pump fillage further, which is exactly
  what drives wells into "Failure Risk" status in the source data.
  This matches the problem statement's description of SPM being
  "adjusted manually and reactively" rather than being set correctly
  ahead of time.

WHAT ROD FLOATING IS (plain terms):
  The pump rod moves down faster than the thick oil can flow into the
  barrel beneath it. The rod "floats" — outruns the fluid — then slams
  into it when the fluid finally catches up. That repeated impact is
  what damages rods and pump components. It happens when SPM is too
  high for the CURRENT oil viscosity, which is exactly the mismatch
  the field's own (backwards) SPM practice creates.

THE FIX THIS MODULE PROVIDES:
  Instead of reacting after cavitation/rod-floating already happened,
  continuously recommend the SAFE SPM for the CURRENT viscosity —
  proactively lower(!) SPM as oil thickens, opposite to field practice,
  keeping pump fillage in a healthy range and avoiding impact loading
  before it starts.
"""

import math
from dataclasses import dataclass

from .field_calibration import srp_fillage_pct, srp_full_state


# =====================================================================
# Field's ACTUAL current practice, fitted from real srp_data.pdf
# (spm vs viscosity, R^2=0.86) — used purely as a comparison baseline
# to show "what they do now" vs "what we recommend instead".
# =====================================================================
FIELD_PRACTICE_SPM_SLOPE = 0.000208
FIELD_PRACTICE_SPM_INTERCEPT = 2.3333


def field_practice_spm(viscosity_cp: float) -> float:
    """What the field's current (reactive, backwards) practice would run."""
    return round(FIELD_PRACTICE_SPM_SLOPE * viscosity_cp + FIELD_PRACTICE_SPM_INTERCEPT, 2)


# =====================================================================
# Fillage-SPM relationship (physically motivated)
# Standard SRP pump-fillage principle: for a roughly fixed reservoir
# inflow rate, fillage is inversely proportional to displacement rate,
# and displacement rate scales with SPM. So for a REFERENCE spm at
# which the field-calibrated fillage_pct(visc) was measured, fillage
# at a DIFFERENT spm scales as:
#     fillage(visc, spm) ~= fillage_calibrated(visc) * (reference_spm / spm)
# This lets us ask "what SPM would bring fillage up to a healthy
# target?" — which is exactly the recommendation the PS asks for.
# =====================================================================
TARGET_HEALTHY_FILLAGE_PCT = 88.0   # aim to stay solidly in "Normal" band
MIN_SAFE_SPM = 1.5
MAX_SAFE_SPM = 5.0


def recommended_safe_spm(viscosity_cp: float,
                          target_fillage_pct: float = TARGET_HEALTHY_FILLAGE_PCT) -> float:
    """
    Recommend the SPM that should keep pump fillage near the healthy
    target for the CURRENT viscosity — this is the actual "optimize
    SRP by adjusting stroke speed and SPM" deliverable the PS asks for.

    As viscosity rises, this recommends LOWER SPM (opposite of the
    field's current reactive practice) to protect fillage.
    """
    reference_spm = field_practice_spm(viscosity_cp)  # spm at which field_calibration's
                                                          # fillage fit was implicitly measured
    fillage_at_reference = srp_fillage_pct(viscosity_cp)

    if fillage_at_reference <= 0:
        return MIN_SAFE_SPM

    # Solve for spm such that fillage scales to target
    recommended = reference_spm * (fillage_at_reference / target_fillage_pct)
    return round(min(max(recommended, MIN_SAFE_SPM), MAX_SAFE_SPM), 2)


def predicted_fillage_at_spm(viscosity_cp: float, spm: float) -> float:
    """Estimate fillage % if running at a GIVEN spm (not the field-practice one)."""
    reference_spm = field_practice_spm(viscosity_cp)
    fillage_at_reference = srp_fillage_pct(viscosity_cp)
    if spm <= 0:
        spm = 0.1
    scaled = fillage_at_reference * (reference_spm / spm)
    return round(min(max(scaled, 0.0), 100.0), 1)


# =====================================================================
# Rod floating detection
# =====================================================================
ROD_FLOATING_FILLAGE_THRESHOLD_PCT = 65.0   # below this, treat as active floating risk
ROD_FLOATING_SPM_EXCESS_THRESHOLD = 0.4     # spm above recommended -> flag risk


@dataclass
class RodFloatingAssessment:
    is_floating_risk: bool
    predicted_fillage_pct: float
    current_spm: float
    recommended_spm: float
    spm_excess: float
    severity: str          # "none", "watch", "risk", "active_floating"
    explanation: str


def assess_rod_floating(viscosity_cp: float, current_spm: float) -> RodFloatingAssessment:
    """
    Given current viscosity and the SPM actually being run, determine
    whether the pump is at risk of (or actively experiencing) rod
    floating, and what SPM would fix it.

    This is the proactive detector the PS asks for: "detect rod
    floating and minimize impact loading" — evaluated continuously
    against live viscosity, not just after damage is already visible.
    """
    rec_spm = recommended_safe_spm(viscosity_cp)
    predicted_fillage = predicted_fillage_at_spm(viscosity_cp, current_spm)
    spm_excess = round(current_spm - rec_spm, 2)

    if predicted_fillage < ROD_FLOATING_FILLAGE_THRESHOLD_PCT and spm_excess > ROD_FLOATING_SPM_EXCESS_THRESHOLD:
        severity = "active_floating"
        explanation = (
            f"Pump fillage predicted at {predicted_fillage}% — well below the "
            f"{ROD_FLOATING_FILLAGE_THRESHOLD_PCT}% safety line. Current SPM "
            f"({current_spm}) is {spm_excess} above the recommended safe SPM "
            f"({rec_spm}) for this viscosity. Rod is likely outrunning fluid "
            f"fill on each stroke — active rod floating / impact loading risk."
        )
        is_risk = True
    elif spm_excess > ROD_FLOATING_SPM_EXCESS_THRESHOLD:
        severity = "risk"
        explanation = (
            f"Current SPM ({current_spm}) exceeds the recommended safe SPM "
            f"({rec_spm}) for viscosity {viscosity_cp:.0f} cP. Fillage is "
            f"trending down ({predicted_fillage}%) — reduce SPM proactively "
            f"before floating begins."
        )
        is_risk = True
    elif spm_excess > 0:
        severity = "watch"
        explanation = (
            f"SPM is slightly above the recommended level for current "
            f"viscosity. No immediate risk, but trending in the wrong "
            f"direction as the reservoir cools."
        )
        is_risk = False
    else:
        severity = "none"
        explanation = "SPM is at or below the recommended safe level for current viscosity."
        is_risk = False

    return RodFloatingAssessment(
        is_floating_risk=is_risk,
        predicted_fillage_pct=predicted_fillage,
        current_spm=current_spm,
        recommended_spm=rec_spm,
        spm_excess=spm_excess,
        severity=severity,
        explanation=explanation,
    )


if __name__ == "__main__":
    print("=== SPM Recommendation vs. Field's Actual (Backwards) Practice ===")
    print(f"{'visc':>7} {'field_practice_spm':>19} {'recommended_spm':>16} {'direction':>12}")
    for visc in [4200, 5700, 6400, 7200, 8100, 9200, 10500, 11800]:
        field_spm = field_practice_spm(visc)
        rec_spm = recommended_safe_spm(visc)
        direction = "SLOW DOWN" if rec_spm < field_spm else "speed up" if rec_spm > field_spm else "same"
        print(f"{visc:>7} {field_spm:>19} {rec_spm:>16} {direction:>12}")

    print("\n=== Rod Floating Assessment: what happens if field keeps its current practice ===")
    for visc in [6400, 8100, 10500, 11800]:
        field_spm = field_practice_spm(visc)
        assessment = assess_rod_floating(visc, field_spm)
        print(f"\nvisc={visc} cP, running field-practice SPM={field_spm}")
        print(f"  severity: {assessment.severity}")
        print(f"  {assessment.explanation}")

    print("\n=== Rod Floating Assessment: what happens if we follow OUR recommendation instead ===")
    for visc in [6400, 8100, 10500, 11800]:
        rec_spm = recommended_safe_spm(visc)
        assessment = assess_rod_floating(visc, rec_spm)
        print(f"\nvisc={visc} cP, running RECOMMENDED SPM={rec_spm}")
        print(f"  severity: {assessment.severity}  |  predicted fillage: {assessment.predicted_fillage_pct}%")
