"""
AI-DRISHTI Physics Engine — Steam-Oil Ratio (SOR) Tracking
------------------------------------------------------------
Directly addresses Problem Statement requirement:
  "Optimize steam and energy usage, reduce Steam-Oil Ratio (SOR)"

SOR is a standard, widely-used thermal-EOR efficiency metric:

    SOR = (steam injected, cold-water-equivalent barrels or tonnes)
          / (oil produced, barrels)

Lower SOR = more oil produced per unit of steam = more efficient,
cheaper operation (steam generation is the dominant operating cost
in CSS). This module computes SOR from the same field-calibrated
physics outputs already being generated, and estimates how much SOR
improves when the SRP optimizer's recommended SPM (see srp_optimizer.py)
is followed instead of the field's current reactive practice — this
ties CSS efficiency directly to SRP behavior, which is itself another
concrete instance of the "these two systems should be optimized
together, not separately" thesis running through the whole project.
"""

from dataclasses import dataclass
from typing import List

# Real data from Css_optimization_data.pdf: steam_volume (tonnes) per
# injection day, alongside oil_production_after (BPD). Used to derive
# a tonnes-to-barrels conversion baseline so SOR can be reported in
# the conventional bbl-steam / bbl-oil form.
STEAM_TONNES_TO_BBL = 6.29  # standard cold-water-equivalent conversion
                              # (1 tonne water ~= 6.29 bbl); steam quality
                              # correction is folded into this approximation
                              # for hackathon-level SOR reporting


@dataclass
class CycleSORResult:
    total_steam_bbl: float
    total_oil_bbl: float
    sor: float
    efficiency_note: str


def compute_sor(total_steam_tonnes: float, total_oil_bbl: float) -> CycleSORResult:
    """
    Compute Steam-Oil Ratio for a completed (or in-progress) cycle.
    Typical CSS operations target SOR in the 2-6 range depending on
    reservoir quality; higher SOR means more steam wasted per barrel
    recovered.
    """
    steam_bbl = total_steam_tonnes * STEAM_TONNES_TO_BBL
    if total_oil_bbl <= 0:
        return CycleSORResult(steam_bbl, total_oil_bbl, float("inf"),
                               "No oil produced yet — SOR undefined.")

    sor = round(steam_bbl / total_oil_bbl, 3)

    if sor < 3.0:
        note = "Excellent efficiency — well below typical CSS SOR range."
    elif sor < 5.0:
        note = "Good efficiency — within typical healthy CSS SOR range (2-6)."
    elif sor < 8.0:
        note = "Elevated SOR — steam usage is high relative to oil recovered."
    else:
        note = "Poor efficiency — investigate cycle parameters or well conditions."

    return CycleSORResult(round(steam_bbl, 1), round(total_oil_bbl, 1), sor, note)


def estimate_sor_improvement_from_spm_fix(baseline_sor: float,
                                            baseline_avg_fillage_pct: float,
                                            optimized_avg_fillage_pct: float) -> dict:
    """
    Estimate how SOR improves when the SRP optimizer's recommended SPM
    is followed instead of the field's current reactive practice.

    Rationale: oil recovered for the SAME steam input scales roughly
    with average pump fillage over the production phase (a poorly
    filled pump leaves recoverable heated oil in the wellbore instead
    of lifting it before the reservoir cools further) — so improving
    average fillage directly improves barrels-per-steam-unit, i.e.
    lowers SOR, for an unchanged steam program.
    """
    if baseline_avg_fillage_pct <= 0:
        return {"error": "invalid baseline fillage"}

    fillage_improvement_ratio = optimized_avg_fillage_pct / baseline_avg_fillage_pct
    improved_sor = round(baseline_sor / fillage_improvement_ratio, 3)
    pct_reduction = round((1 - improved_sor / baseline_sor) * 100, 1)

    return {
        "baseline_sor": baseline_sor,
        "estimated_optimized_sor": improved_sor,
        "estimated_sor_reduction_pct": pct_reduction,
        "basis": (f"Average fillage improves from {baseline_avg_fillage_pct}% to "
                  f"{optimized_avg_fillage_pct}% by following recommended SPM "
                  f"instead of field practice, proportionally improving barrels "
                  f"recovered per unit steam.")
    }


def sor_across_cycles(cycle_steam_tonnes: List[float], cycle_oil_bbl: List[float]) -> List[CycleSORResult]:
    """Compute SOR per cycle across a well's history — feeds a trend chart."""
    return [compute_sor(s, o) for s, o in zip(cycle_steam_tonnes, cycle_oil_bbl)]


if __name__ == "__main__":
    print("=== SOR from real CSS data pattern (BGW_01, Cycle 1) ===")
    # Using the real per-day steam volumes and final day's oil production
    # rate as a rough single-cycle proxy for hackathon demo purposes
    total_steam_tonnes = 72 + 73.2 + 74.4 + 75.6 + 76.8  # sum of 5 injection days
    # Approximate cumulative oil over a subsequent ~60-day production run
    # using the declining rate pattern our thermal model already produces
    approx_cycle_oil_bbl = 8500  # representative mid-range from our calibrated simulation

    result = compute_sor(total_steam_tonnes, approx_cycle_oil_bbl)
    print(f"Total steam: {total_steam_tonnes} tonnes ({result.total_steam_bbl} bbl equiv)")
    print(f"Total oil: {result.total_oil_bbl} bbl")
    print(f"SOR: {result.sor}  ->  {result.efficiency_note}")

    print("\n=== Estimated SOR improvement from following SPM recommendation ===")
    # From the srp_optimizer demo: field practice degrades to ~53-60% fillage
    # at high viscosity, our recommendation holds ~88% fillage throughout
    improvement = estimate_sor_improvement_from_spm_fix(
        baseline_sor=result.sor,
        baseline_avg_fillage_pct=68.0,   # rough average across field practice's declining trend
        optimized_avg_fillage_pct=88.0,  # our optimizer's held-steady level
    )
    for k, v in improvement.items():
        print(f"  {k}: {v}")
