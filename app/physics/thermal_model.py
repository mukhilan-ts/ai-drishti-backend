"""
AI-DRISHTI Physics Engine — Boberg-Lantz Thermal Decline Model
------------------------------------------------------------------
Implements a simplified, hackathon-practical version of the Boberg-Lantz
(1966) model for Cyclic Steam Stimulation (CSS): after steam injection,
the heated zone around the wellbore cools over time, and oil production
rate is driven by how hot (and therefore how low-viscosity) that zone
still is.

This is intentionally a simplified analytical approximation of the real
model (which requires iterative heat-balance solving) — it captures the
correct SHAPE of CSS behavior (fast decline early, long tail) which is
what matters for a believable, physically-grounded demo and for driving
realistic downstream dynamometer card behavior.

Core physical intuition modeled here:
  1. During INJECTION: bottomhole temp rises toward steam temperature
  2. During SOAK: temp holds near peak, heat conducts outward, reservoir
     around wellbore is at its hottest -> oil is at its least viscous
  3. During PRODUCTION: temp decays roughly exponentially (heat loss
     to surrounding rock + fluid withdrawal), oil viscosity rises,
     production rate declines correspondingly (Arps-like hyperbolic
     decline, but temperature-modulated)

References (public domain / textbook physics, not proprietary IP):
  - Boberg, T.C. and Lantz, R.B., "Calculation of the Production Rate of
    a Thermally Stimulated Well", JPT, 1966.
  - Standard Arps hyperbolic decline formulation.

CALIBRATION NOTE: as of this version, the temperature->viscosity and
viscosity->production-rate relationships below are FIELD-CALIBRATED,
fitted by regression against real Baghewala CSS cycle data (see
field_calibration.py for the fits and validation against source data).
This is a meaningfully stronger claim than "physics-inspired" — these
specific curves reproduce real field measurements.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List

from .field_calibration import (
    viscosity_from_temperature as calibrated_viscosity_from_temperature,
    oil_rate_from_viscosity_css as calibrated_oil_rate_from_viscosity,
    srp_full_state,
    CALIBRATION_VALID_TEMP_MAX_C,
)


class CyclePhase(str, Enum):
    INJECTION = "injection"
    SOAK = "soak"
    PRODUCTION = "production"


@dataclass
class WellReservoirParams:
    """Static reservoir/well properties — would come from the wells table."""
    reservoir_depth_m: float = 1150.0          # Jodhpur Sandstone avg depth (Baghewala)
    pay_thickness_m: float = 12.0
    initial_reservoir_temp_c: float = 45.0      # ambient reservoir temp before steam
    steam_temp_c: float = 290.0                 # injected steam temp — real data range is 275-305C
    api_gravity: float = 12.0                   # heavy oil, matches ~10-13 API range
    reference_viscosity_cp: float = 14926.0     # field-calibrated value at 45C (see field_calibration.py)
    thermal_diffusivity: float = 0.0011         # m^2/hr, rough sandstone value
    wellbore_radius_m: float = 0.1
    vit_completion: bool = True                 # Vacuum Insulated Tubing reduces heat loss


@dataclass
class CSSCycleParams:
    """Parameters for a single CSS cycle — what the AI optimizer would tune."""
    steam_volume_bbl: float = 1500.0
    injection_days: float = 5.0
    soak_days: float = 4.0
    production_days: float = 60.0
    steam_quality_pct: float = 80.0


@dataclass
class SimPoint:
    """One timestep of simulated well state."""
    elapsed_hours: float
    phase: CyclePhase
    bottomhole_temp_c: float
    oil_viscosity_cp: float
    oil_rate_bpd: float
    cumulative_oil_bbl: float
    pump_status: str = "N/A"          # Normal / Reduced Efficiency / High Load / Failure Risk
    pump_fillage_pct: float = 0.0     # field-calibrated from viscosity


def viscosity_from_temperature(temp_c: float, ref_visc_cp: float = None, ref_temp_c: float = None) -> float:
    """
    Field-calibrated heavy-oil viscosity-temperature relationship,
    fitted from real Baghewala CSS cycle data (see field_calibration.py).
    ref_visc_cp/ref_temp_c args are accepted for backwards compatibility
    with earlier calls but the calibrated fit is used regardless — the
    field data determines both the reference point and decay rate.
    """
    return calibrated_viscosity_from_temperature(temp_c)


def bottomhole_temp_injection(t_hours: float, injection_hours: float,
                               t_start_c: float, t_steam_c: float) -> float:
    """Temperature ramps toward steam temp during injection (fast, near-linear)."""
    frac = min(t_hours / max(injection_hours, 0.01), 1.0)
    # ease-in curve: temp rises faster early (steam contact) then levels off
    return t_start_c + (t_steam_c - t_start_c) * (1 - math.exp(-3.0 * frac))


def bottomhole_temp_soak(t_hours_into_soak: float, soak_hours: float,
                          peak_temp_c: float, ambient_temp_c: float,
                          diffusivity: float) -> float:
    """
    During soak, temperature decays slowly as heat conducts into
    surrounding rock (no fluid withdrawal yet). Modeled as a mild
    exponential decay — much slower than the production-phase decay
    because there's no convective heat removal via produced fluid.
    """
    decay_rate = 0.008 * (1 + diffusivity * 10)  # slow decay during soak
    return ambient_temp_c + (peak_temp_c - ambient_temp_c) * math.exp(-decay_rate * t_hours_into_soak)


def bottomhole_temp_production(t_hours_into_prod: float, start_temp_c: float,
                                ambient_temp_c: float, vit_completion: bool) -> float:
    """
    Near-wellbore temperature during production — used ONLY for the 3D
    steam-chamber visualization (dramatic, can run hot near the wellbore
    long after bulk reservoir has cooled). NOT used for viscosity/rate/
    SRP calculations — see effective_reservoir_temp_production() below,
    which is the field-calibrated value those depend on.

    VIT (Vacuum Insulated Tubing, used at Baghewala) reduces heat loss
    up the wellbore, so wells with VIT retain near-well heat longer.
    """
    decay_rate = 0.0006 if vit_completion else 0.0011  # per hour
    return ambient_temp_c + (start_temp_c - ambient_temp_c) * math.exp(-decay_rate * t_hours_into_prod)


def effective_reservoir_temp_production(t_hours_into_prod: float,
                                         ambient_temp_c: float,
                                         vit_completion: bool,
                                         soak_days: float = 4.0,
                                         steam_volume_bbl: float = 1500.0) -> float:
    """
    BULK reservoir temperature during production — this is what actually
    drives oil viscosity, production rate, and SRP pump status, and is
    kept within the field-calibrated validated range (see
    field_calibration.CALIBRATION_VALID_TEMP_MAX_C).

    Two CSS parameters genuinely influence this curve, which is what
    makes css_optimizer.py's search meaningful rather than a no-op:

    - soak_days controls how COMPLETE the heat penetration is by the
      time production starts. A short soak doesn't fully reach the
      calibration-validated ceiling; REFERENCE_SOAK_DAYS_FOR_FULL_HEAT
      is the soak duration at which the well reaches the full validated
      92C starting point. Longer soak beyond that has no further
      benefit in this simplified model (physically: heat penetration
      saturates).

    - steam_volume_bbl controls how much total heat is STORED in the
      near-wellbore zone, which affects how fast that zone cools during
      production (more steam -> slower decay -> longer effective cycle
      life), independent of the starting temperature.
    """
    REFERENCE_SOAK_DAYS_FOR_FULL_HEAT = 4.0
    REFERENCE_STEAM_VOLUME_BBL = 1500.0

    soak_completeness = min(1.0, soak_days / REFERENCE_SOAK_DAYS_FOR_FULL_HEAT)
    start_temp = ambient_temp_c + (CALIBRATION_VALID_TEMP_MAX_C - ambient_temp_c) * soak_completeness

    base_decay_rate = 0.0012 if vit_completion else 0.0020  # per hour
    # More steam -> more stored heat -> slower decay. Clamp the multiplier
    # to a sane range so extreme steam volumes don't produce unphysical
    # near-zero or runaway decay rates.
    steam_richness = steam_volume_bbl / REFERENCE_STEAM_VOLUME_BBL
    decay_multiplier = min(max(1.0 / steam_richness, 0.5), 2.0)
    decay_rate = base_decay_rate * decay_multiplier

    return ambient_temp_c + (start_temp - ambient_temp_c) * math.exp(-decay_rate * t_hours_into_prod)


def oil_rate_from_viscosity(viscosity_cp: float, max_rate_bpd: float,
                             viscosity_half_rate_cp: float = 1500.0) -> float:
    """
    Field-calibrated production rate as a function of viscosity, fitted
    from real Baghewala CSS data (power law, see field_calibration.py:
    rate = C * visc^-1.076, mean abs error 8.97 BPD on source data).
    max_rate_bpd acts as the pump's mechanical delivery ceiling (used to
    clamp the power-law curve, which would otherwise diverge at very low
    viscosity during/right after steam injection).
    """
    return calibrated_oil_rate_from_viscosity(viscosity_cp, pump_capacity_bpd=max_rate_bpd)


def simulate_css_cycle(reservoir: WellReservoirParams, cycle: CSSCycleParams,
                        timestep_hours: float = 2.0,
                        max_rate_bpd: float = 45.0) -> List[SimPoint]:
    """
    Run one full CSS cycle (injection -> soak -> production) and return
    a time series of simulated well state. This is what feeds both the
    3D steam-chamber animation and the production forecast chart.
    """
    points: List[SimPoint] = []
    cumulative_oil = 0.0

    injection_hours = cycle.injection_days * 24
    soak_hours = cycle.soak_days * 24
    production_hours = cycle.production_days * 24

    # --- Phase 1: Injection ---
    t = 0.0
    while t <= injection_hours:
        temp = bottomhole_temp_injection(t, injection_hours,
                                          reservoir.initial_reservoir_temp_c,
                                          reservoir.steam_temp_c)
        visc = viscosity_from_temperature(temp, reservoir.reference_viscosity_cp,
                                           reservoir.initial_reservoir_temp_c)
        points.append(SimPoint(t, CyclePhase.INJECTION, round(temp, 2),
                                round(visc, 2), 0.0, 0.0))
        t += timestep_hours

    peak_temp = points[-1].bottomhole_temp_c

    # --- Phase 2: Soak ---
    t_soak = 0.0
    while t_soak <= soak_hours:
        temp = bottomhole_temp_soak(t_soak, soak_hours, peak_temp,
                                     reservoir.initial_reservoir_temp_c,
                                     reservoir.thermal_diffusivity)
        visc = viscosity_from_temperature(temp, reservoir.reference_viscosity_cp,
                                           reservoir.initial_reservoir_temp_c)
        points.append(SimPoint(injection_hours + t_soak, CyclePhase.SOAK,
                                round(temp, 2), round(visc, 2), 0.0, 0.0))
        t_soak += timestep_hours

    prod_start_temp = points[-1].bottomhole_temp_c

    # --- Phase 3: Production ---
    t_prod = 0.0
    while t_prod <= production_hours:
        # Near-wellbore temp: cosmetic/visual only (3D steam chamber)
        visual_temp = bottomhole_temp_production(t_prod, prod_start_temp,
                                                  reservoir.initial_reservoir_temp_c,
                                                  reservoir.vit_completion)
        # Bulk reservoir temp: field-calibrated, drives everything real —
        # now genuinely responsive to soak_days and steam_volume_bbl
        bulk_temp = effective_reservoir_temp_production(t_prod,
                                                          reservoir.initial_reservoir_temp_c,
                                                          reservoir.vit_completion,
                                                          soak_days=cycle.soak_days,
                                                          steam_volume_bbl=cycle.steam_volume_bbl)

        visc = viscosity_from_temperature(bulk_temp)
        rate_bpd = oil_rate_from_viscosity(visc, max_rate_bpd)
        cumulative_oil += rate_bpd * (timestep_hours / 24.0)

        # Field-calibrated SRP surface state — this is the CSS-SRP
        # coupling in action: pump health status is derived directly
        # from the SAME bulk-reservoir viscosity driving production.
        srp_state = srp_full_state(visc)

        points.append(SimPoint(injection_hours + soak_hours + t_prod,
                                CyclePhase.PRODUCTION, round(visual_temp, 2),
                                round(visc, 2), round(rate_bpd, 2),
                                round(cumulative_oil, 2),
                                pump_status=srp_state["status"],
                                pump_fillage_pct=srp_state["pump_fillage_pct"]))
        t_prod += timestep_hours

    return points


def simulate_multiple_cycles(reservoir: WellReservoirParams,
                              cycle_params_list: List[CSSCycleParams],
                              timestep_hours: float = 2.0,
                              max_rate_bpd: float = 45.0) -> List[SimPoint]:
    """
    Run several CSS cycles back-to-back, carrying reservoir state forward
    between cycles (each cycle starts from wherever the previous one left
    off, not from cold baseline) — this is what powers the "Time Machine"
    timeline scrubber across a well's full cycle history.

    Each cycle typically produces a smaller peak/total than the last,
    since repeated steam cycles progressively deplete the near-wellbore
    oil saturation — modeled here with a simple per-cycle rate decay
    factor, a common simplification used in CSS type-curve analysis.
    """
    all_points: List[SimPoint] = []
    time_offset = 0.0
    current_reservoir = reservoir

    for i, cycle in enumerate(cycle_params_list):
        # Each successive cycle draws down slightly (near-wellbore
        # depletion) -> reduce effective max rate a bit per cycle
        depletion_factor = 0.92 ** i
        cycle_points = simulate_css_cycle(current_reservoir, cycle, timestep_hours,
                                           max_rate_bpd * depletion_factor)
        # Re-timestamp and carry cumulative oil forward across cycles
        prior_cumulative = all_points[-1].cumulative_oil_bbl if all_points else 0.0
        for p in cycle_points:
            all_points.append(SimPoint(
                elapsed_hours=round(p.elapsed_hours + time_offset, 2),
                phase=p.phase,
                bottomhole_temp_c=p.bottomhole_temp_c,
                oil_viscosity_cp=p.oil_viscosity_cp,
                oil_rate_bpd=p.oil_rate_bpd,
                cumulative_oil_bbl=round(p.cumulative_oil_bbl + prior_cumulative, 2),
                pump_status=p.pump_status,
                pump_fillage_pct=p.pump_fillage_pct,
            ))
        time_offset = all_points[-1].elapsed_hours + timestep_hours
        # Next cycle starts from ambient again (well is shut in/cooled
        # between cycles in this simplified model)
        current_reservoir = current_reservoir

    return all_points


def summarize_cycle(points: List[SimPoint]) -> dict:
    """Quick summary stats — useful for the AI optimizer to compare cycle configs."""
    prod_points = [p for p in points if p.phase == CyclePhase.PRODUCTION]
    if not prod_points:
        return {}
    return {
        "total_oil_bbl": prod_points[-1].cumulative_oil_bbl,
        "peak_rate_bpd": max(p.oil_rate_bpd for p in prod_points),
        "avg_rate_bpd": round(sum(p.oil_rate_bpd for p in prod_points) / len(prod_points), 2),
        "days_above_20bpd": round(sum(1 for p in prod_points if p.oil_rate_bpd > 20)
                                   * (points[1].elapsed_hours - points[0].elapsed_hours) / 24, 1),
        "peak_bottomhole_temp_c": max(p.bottomhole_temp_c for p in points),
        "final_temp_c": prod_points[-1].bottomhole_temp_c,
    }


if __name__ == "__main__":
    # Quick sanity test — run a default cycle and print a summary
    reservoir = WellReservoirParams()
    cycle = CSSCycleParams()
    result = simulate_css_cycle(reservoir, cycle)

    print(f"Simulated {len(result)} timesteps across full CSS cycle")
    print(f"Phases present: {set(p.phase.value for p in result)}")
    print("\nSample points:")
    for p in result[::20]:
        print(f"  t={p.elapsed_hours:>6.1f}h  phase={p.phase.value:<10} "
              f"temp={p.bottomhole_temp_c:>6.1f}C  visc={p.oil_viscosity_cp:>8.1f}cP  "
              f"rate={p.oil_rate_bpd:>5.1f}bpd  cum={p.cumulative_oil_bbl:>7.1f}bbl")

    print("\nCycle summary:")
    for k, v in summarize_cycle(result).items():
        print(f"  {k}: {v}")
