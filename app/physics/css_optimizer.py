"""
AI-DRISHTI — Live-State-Aware CSS Cycle Optimizer

Optimizes:
    - Steam volume
    - Soak duration

using:
    1. Existing physics-based CSS simulation
    2. Current live well condition

Live condition:
    - Bottomhole temperature
    - Oil viscosity
    - Oil production rate
    - Current phase

IMPORTANT:
The optimizer does NOT randomly change recommendations.

Cold / viscous / low-production state
    -> higher thermal stimulation favored

Warm / low-viscosity / healthy-production state
    -> lower thermal stimulation favored
"""

from dataclasses import dataclass
from typing import List, Optional
import math

from .thermal_model import (
    WellReservoirParams,
    CSSCycleParams,
    simulate_css_cycle,
    summarize_cycle,
)

from .sor_tracker import (
    compute_sor,
    STEAM_TONNES_TO_BBL,
)


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class CSSLiveState:
    temperature_c: Optional[float] = None
    viscosity_cp: Optional[float] = None
    oil_rate_bpd: Optional[float] = None
    phase: Optional[str] = None


@dataclass
class CSSOptimizationCandidate:
    steam_volume_bbl: float
    soak_days: float
    production_days: float
    total_oil_bbl: float
    peak_rate_bpd: float
    sor: float
    net_score: float


@dataclass
class CSSOptimizationResult:
    recommended: CSSOptimizationCandidate
    all_candidates: List[CSSOptimizationCandidate]
    fixed_cycle_length_days: float
    injection_days: float


# =====================================================================
# BASIC HELPERS
# =====================================================================

def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:

    return max(
        low,
        min(high, value),
    )


# =====================================================================
# LIVE THERMAL DEMAND
# =====================================================================

def calculate_heat_need(
    live_state: Optional[CSSLiveState],
) -> float:
    """
    Calculate current thermal stimulation demand.

    Output:
        0.0 -> very warm / low thermal demand
        0.5 -> moderate thermal demand
        1.0 -> very cold / high thermal demand

    Temperature is mapped across the actual simulated well range:

        ~150 C -> very low demand
        ~120 C -> low demand
        ~90 C  -> moderate demand
        ~70 C  -> high demand
        ~50 C  -> very high demand

    This is deliberately explainable and deterministic.
    """

    if live_state is None:
        return 0.5

    signals = []

    # ================================================================
    # TEMPERATURE
    # ================================================================
    #
    # Baghewala prototype telemetry can reach well above 80 C.
    # Therefore we use a wider operating range.
    #
    # 150 C -> 0
    # 50 C  -> 1
    #
    # Values outside the range are clipped.
    # ================================================================

    if live_state.temperature_c is not None:

        temperature_stress = clamp(
            (150.0 - live_state.temperature_c) / 100.0
        )

        signals.append(
            temperature_stress
        )

    # ================================================================
    # VISCOSITY
    # ================================================================
    #
    # Heavy oil is highly sensitive to viscosity.
    #
    # ~1,000 cP  -> low stress
    # ~15,000 cP -> high stress
    #
    # Logarithmic scaling is used because viscosity spans a large
    # numerical range.
    # ================================================================

    if (
        live_state.viscosity_cp is not None
        and live_state.viscosity_cp > 0
    ):

        viscosity_stress = clamp(
            math.log10(
                max(
                    live_state.viscosity_cp,
                    500.0,
                ) / 1000.0
            )
            / math.log10(15.0)
        )

        signals.append(
            viscosity_stress
        )

    # ================================================================
    # PRODUCTION RATE
    # ================================================================
    #
    # Low oil rate increases thermal-stimulation demand.
    #
    # 200+ bpd -> low stress
    # ~40 bpd  -> high stress
    # ================================================================

    if live_state.oil_rate_bpd is not None:

        rate_stress = clamp(
            (200.0 - live_state.oil_rate_bpd)
            / 160.0
        )

        signals.append(
            rate_stress
        )

    # ================================================================
    # OPERATING PHASE
    # ================================================================

    if live_state.phase:

        phase = live_state.phase.lower()

        if (
            "injection" in phase
            or "steam" in phase
        ):

            signals.append(0.85)

        elif "soak" in phase:

            signals.append(0.75)

        elif "production" in phase:

            # Production is determined mainly from actual
            # temperature / viscosity / rate.
            signals.append(0.50)

    # ================================================================
    # FINAL HEAT DEMAND
    # ================================================================

    if not signals:
        return 0.5

    return clamp(
        sum(signals) / len(signals)
    )


# =====================================================================
# LIVE CONDITION → CSS PREFERENCE
# =====================================================================

def calculate_state_alignment(
    steam_bbl: float,
    soak_days: float,
    steam_options: List[float],
    soak_options: List[float],
    heat_need: float,
) -> float:
    """
    Calculates how well a CSS candidate matches the current
    thermal condition.

    heat_need = 0:
        low steam + short soak preferred

    heat_need = 1:
        high steam + long soak preferred
    """

    steam_min = min(
        steam_options
    )

    steam_max = max(
        steam_options
    )

    soak_min = min(
        soak_options
    )

    soak_max = max(
        soak_options
    )

    # ---------------------------------------------------------------
    # Normalize steam candidate to 0–1.
    # ---------------------------------------------------------------

    if steam_max > steam_min:

        steam_position = (
            steam_bbl - steam_min
        ) / (
            steam_max - steam_min
        )

    else:

        steam_position = 0.5

    # ---------------------------------------------------------------
    # Normalize soak candidate to 0–1.
    # ---------------------------------------------------------------

    if soak_max > soak_min:

        soak_position = (
            soak_days - soak_min
        ) / (
            soak_max - soak_min
        )

    else:

        soak_position = 0.5

    steam_position = clamp(
        steam_position
    )

    soak_position = clamp(
        soak_position
    )

    # ---------------------------------------------------------------
    # Desired position is determined by heat demand.
    #
    # Cold:
    #     desired = near 1
    #
    # Warm:
    #     desired = near 0
    # ---------------------------------------------------------------

    desired_position = heat_need

    steam_alignment = (
        1.0
        - abs(
            steam_position
            - desired_position
        )
    )

    soak_alignment = (
        1.0
        - abs(
            soak_position
            - desired_position
        )
    )

    # Steam is the dominant thermal-control decision.
    alignment = (
        0.70 * steam_alignment
        + 0.30 * soak_alignment
    )

    return clamp(
        alignment
    )


# =====================================================================
# MAIN CSS OPTIMIZER
# =====================================================================

def optimize_css_cycle(
    reservoir: WellReservoirParams,
    injection_days: float = 5.0,
    fixed_cycle_length_days: float = 70.0,
    steam_volume_options_bbl: Optional[List[float]] = None,
    soak_days_options: Optional[List[float]] = None,
    oil_price_weight: float = 1.0,
    sor_penalty_weight: float = 8.0,
    max_rate_bpd: float = 260.0,
    live_state: Optional[CSSLiveState] = None,
) -> CSSOptimizationResult:

    # ================================================================
    # DEFAULT SEARCH SPACE
    # ================================================================

    if steam_volume_options_bbl is None:

        steam_volume_options_bbl = [
            1000,
            1250,
            1500,
            1750,
            2000,
        ]

    if soak_days_options is None:

        soak_days_options = [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]

    # ================================================================
    # CURRENT LIVE HEAT DEMAND
    # ================================================================

    heat_need = calculate_heat_need(
        live_state
    )

    # ================================================================
    # RUN PHYSICS SIMULATION FOR EVERY CANDIDATE
    # ================================================================

    raw_candidates = []

    for steam_bbl in steam_volume_options_bbl:

        for soak_days in soak_days_options:

            production_days = (
                fixed_cycle_length_days
                - injection_days
                - soak_days
            )

            # Skip impossible cycles.
            if production_days <= 5:
                continue

            # --------------------------------------------------------
            # Build CSS cycle
            # --------------------------------------------------------

            cycle = CSSCycleParams(
                steam_volume_bbl=steam_bbl,
                injection_days=injection_days,
                soak_days=soak_days,
                production_days=production_days,
            )

            # --------------------------------------------------------
            # Existing validated physics engine
            # --------------------------------------------------------

            sim_points = simulate_css_cycle(
                reservoir,
                cycle,
                timestep_hours=4.0,
                max_rate_bpd=max_rate_bpd,
            )

            summary = summarize_cycle(
                sim_points
            )

            if not summary:
                continue

            total_oil = (
                summary["total_oil_bbl"]
            )

            peak_rate = (
                summary["peak_rate_bpd"]
            )

            # --------------------------------------------------------
            # SOR
            # --------------------------------------------------------

            steam_tonnes = (
                steam_bbl
                / STEAM_TONNES_TO_BBL
            )

            sor_result = compute_sor(
                steam_tonnes,
                total_oil,
            )

            sor = sor_result.sor

            # --------------------------------------------------------
            # Original physics/economic score
            # --------------------------------------------------------

            base_score = (
                oil_price_weight
                * total_oil
                -
                sor_penalty_weight
                * sor
            )

            raw_candidates.append(
                {
                    "steam_bbl": steam_bbl,
                    "soak_days": soak_days,
                    "production_days": production_days,
                    "total_oil": total_oil,
                    "peak_rate": peak_rate,
                    "sor": sor,
                    "base_score": base_score,
                }
            )

    # ================================================================
    # SAFETY CHECK
    # ================================================================

    if not raw_candidates:

        raise ValueError(
            "No viable candidates — "
            "check parameter ranges."
        )

    # ================================================================
    # NORMALIZE PHYSICS SCORE
    # ================================================================

    scores = [
        candidate["base_score"]
        for candidate in raw_candidates
    ]

    minimum_score = min(scores)

    maximum_score = max(scores)

    score_range = (
        maximum_score
        - minimum_score
    )

    # ================================================================
    # FINAL LIVE-AWARE RANKING
    # ================================================================

    candidates = []

    for candidate in raw_candidates:

        # ------------------------------------------------------------
        # Normalize physics performance.
        # ------------------------------------------------------------

        if score_range > 0:

            physics_score = (
                candidate["base_score"]
                - minimum_score
            ) / score_range

        else:

            physics_score = 0.5

        # ------------------------------------------------------------
        # Calculate live-state match.
        # ------------------------------------------------------------

        state_alignment = (
            calculate_state_alignment(
                steam_bbl=
                    candidate["steam_bbl"],

                soak_days=
                    candidate["soak_days"],

                steam_options=
                    steam_volume_options_bbl,

                soak_options=
                    soak_days_options,

                heat_need=
                    heat_need,
            )
        )

        # ------------------------------------------------------------
        # FINAL WEIGHTING
        #
        # Physics = 45%
        # Live state = 55%
        #
        # This makes the recommendation responsive to changing
        # thermal conditions while still retaining the physics model.
        # ------------------------------------------------------------

        if live_state is not None:

            final_score = (
                0.45 * physics_score
                +
                0.55 * state_alignment
            )

        else:

            final_score = physics_score

        candidates.append(
            CSSOptimizationCandidate(

                steam_volume_bbl=
                    candidate["steam_bbl"],

                soak_days=
                    candidate["soak_days"],

                production_days=
                    candidate["production_days"],

                total_oil_bbl=
                    candidate["total_oil"],

                peak_rate_bpd=
                    candidate["peak_rate"],

                sor=
                    candidate["sor"],

                net_score=
                    round(
                        final_score,
                        4,
                    ),
            )
        )

    # ================================================================
    # SORT
    # ================================================================

    candidates.sort(
        key=lambda candidate:
            candidate.net_score,
        reverse=True,
    )

    best = candidates[0]

    # ================================================================
    # RETURN
    # ================================================================

    return CSSOptimizationResult(

        recommended=best,

        all_candidates=candidates,

        fixed_cycle_length_days=
            fixed_cycle_length_days,

        injection_days=
            injection_days,
    )


# =====================================================================
# DIRECT TEST
# =====================================================================

if __name__ == "__main__":

    reservoir = WellReservoirParams()

    # ================================================================
    # TEST 1 — VERY COLD / HIGH VISCOSITY
    # ================================================================

    print("\n")
    print("=" * 60)
    print("COLD / HIGH-VISCOSITY WELL")
    print("=" * 60)

    cold_state = CSSLiveState(

        temperature_c=50,

        viscosity_cp=12000,

        oil_rate_bpd=70,

        phase="production",
    )

    cold_result = optimize_css_cycle(

        reservoir,

        live_state=cold_state,
    )

    print(
        f"Heat demand: "
        f"{calculate_heat_need(cold_state):.2f}"
    )

    print(
        f"Recommended steam: "
        f"{cold_result.recommended.steam_volume_bbl} bbl"
    )

    print(
        f"Recommended soak: "
        f"{cold_result.recommended.soak_days} days"
    )

    print(
        f"Expected oil: "
        f"{cold_result.recommended.total_oil_bbl:.0f} bbl"
    )

    print(
        f"Expected SOR: "
        f"{cold_result.recommended.sor:.3f}"
    )

    # ================================================================
    # TEST 2 — MODERATE WELL
    # ================================================================

    print("\n")
    print("=" * 60)
    print("MODERATE WELL")
    print("=" * 60)

    moderate_state = CSSLiveState(

        temperature_c=90,

        viscosity_cp=4000,

        oil_rate_bpd=140,

        phase="production",
    )

    moderate_result = optimize_css_cycle(

        reservoir,

        live_state=moderate_state,
    )

    print(
        f"Heat demand: "
        f"{calculate_heat_need(moderate_state):.2f}"
    )

    print(
        f"Recommended steam: "
        f"{moderate_result.recommended.steam_volume_bbl} bbl"
    )

    print(
        f"Recommended soak: "
        f"{moderate_result.recommended.soak_days} days"
    )

    print(
        f"Expected oil: "
        f"{moderate_result.recommended.total_oil_bbl:.0f} bbl"
    )

    print(
        f"Expected SOR: "
        f"{moderate_result.recommended.sor:.3f}"
    )

    # ================================================================
    # TEST 3 — VERY WARM / LOW VISCOSITY
    # ================================================================

    print("\n")
    print("=" * 60)
    print("WARM / LOW-VISCOSITY WELL")
    print("=" * 60)

    warm_state = CSSLiveState(

        temperature_c=140,

        viscosity_cp=1500,

        oil_rate_bpd=190,

        phase="production",
    )

    warm_result = optimize_css_cycle(

        reservoir,

        live_state=warm_state,
    )

    print(
        f"Heat demand: "
        f"{calculate_heat_need(warm_state):.2f}"
    )

    print(
        f"Recommended steam: "
        f"{warm_result.recommended.steam_volume_bbl} bbl"
    )

    print(
        f"Recommended soak: "
        f"{warm_result.recommended.soak_days} days"
    )

    print(
        f"Expected oil: "
        f"{warm_result.recommended.total_oil_bbl:.0f} bbl"
    )

    print(
        f"Expected SOR: "
        f"{warm_result.recommended.sor:.3f}"
    )

    print("\n")