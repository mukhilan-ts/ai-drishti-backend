"""
AI-DRISHTI Physics Engine — Dynamometer Card Synthesizer
------------------------------------------------------------
Generates realistic Sucker Rod Pump (SRP) dynamometer cards: a plot of
polished rod LOAD (kg or lbs) vs. rod POSITION over one full up-down
stroke cycle. In a reciprocating pump, this closed loop's SHAPE is the
primary diagnostic signal field engineers use to identify pump problems
— normal operation, fluid pound, gas interference, rod parting, tubing
leak, etc.

THE KEY COUPLING (this is AI-DRISHTI's core differentiator):
  Card shape is generated as a function of fluid VISCOSITY, which comes
  directly from the thermal_model's CSS cycle simulation. Right after a
  steam cycle, oil is thin and the pump behaves close to "normal" even
  under conditions that would look like fluid pound on cold, viscous oil.
  As the reservoir cools (weeks into the production phase), the SAME
  pump running the SAME way starts to show a card shape that drifts
  toward fluid-pound-like behavior purely because the fluid is thicker
  and slower to fill the barrel on each upstroke — NOT because anything
  actually broke.

  This is exactly the physics that lets the anomaly detector distinguish
  "expected thermal-decline pump behavior" from "genuine mechanical
  fault" — the single most important intelligence layer in the whole
  system.

Card conditions modeled:
  - normal              : full, near-rectangular loop (ideal positive displacement)
  - fluid_pound         : load drops sharply on downstroke (barrel not full of liquid)
  - gas_interference     : rounded corners, reduced card area (gas compresses instead of lifting)
  - rod_parting          : load collapses to near-zero for a portion of stroke
  - tubing_leak          : reduced peak load and altered card area (fluid leaking back)

References: card-shape descriptions follow standard API RP 11L /
Gibbs-diagnostic conventions taught in petroleum production engineering
courses — general domain knowledge, not proprietary software output.
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class PumpCondition(str, Enum):
    NORMAL = "normal"
    FLUID_POUND = "fluid_pound"
    GAS_INTERFERENCE = "gas_interference"
    ROD_PARTING = "rod_parting"
    TUBING_LEAK = "tubing_leak"


@dataclass
class DynoCardParams:
    """Surface unit + rod string parameters that shape the ideal card."""
    polished_rod_max_load_kg: float = 1800.0   # peak load, calibrate to unit size
    polished_rod_min_load_kg: float = 600.0    # minimum load (traveling valve closed, fluid weight off)
    stroke_length_m: float = 2.0         # matches real srp_data.pdf (consistently 2m across all rows)
    points_per_stroke: int = 100                # resolution of the generated card


def _base_normal_card(params: DynoCardParams, viscosity_cp: float,
                       ref_viscosity_cp: float = 1500.0) -> List[Tuple[float, float]]:
    """
    Ideal/normal card: a somewhat rounded rectangle. Real SRP cards are
    never perfectly rectangular (rod stretch, fluid inertia, friction),
    so we build the loop from two smooth curves (up-stroke, down-stroke)
    that both start/end at the same corners.

    viscosity_cp modulates how "sharp" the card corners are: thinner oil
    (low viscosity, e.g. right after steam) -> crisper, more rectangular
    card. Thicker oil (high viscosity, cooled reservoir) -> corners round
    off and peak load is reached more gradually -> this IS the physical
    mechanism behind the CSS-phase coupling described above.
    """
    n = params.points_per_stroke
    max_load = params.polished_rod_max_load_kg
    min_load = params.polished_rod_min_load_kg

    # Viscosity -> corner sharpness. Clamp to a sane exponent range.
    visc_ratio = min(max(viscosity_cp / ref_viscosity_cp, 0.2), 6.0)
    sharpness = max(1.2, 5.0 / visc_ratio)  # higher = sharper corners (thin oil)

    points = []
    # Upstroke: position 0 -> 1, load rises from min to max
    for i in range(n // 2):
        pos = i / (n // 2 - 1)
        # smoothstep-like curve, exponent controls corner sharpness
        load_frac = pos ** (1.0 / sharpness)
        load = min_load + (max_load - min_load) * load_frac
        points.append((pos, load))

    # Downstroke: position 1 -> 0, load falls from max back to min
    for i in range(n // 2):
        pos = 1.0 - i / (n // 2 - 1)
        load_frac = pos ** (1.0 / sharpness)
        # downstroke load is offset lower than upstroke at same position
        # (this offset is what gives the card its characteristic loop
        # "width" / enclosed area, representing useful work done)
        load = min_load + (max_load - min_load) * load_frac * 0.72
        points.append((pos, load))

    return points


def _apply_fluid_pound(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Fluid pound: the barrel isn't fully liquid-filled at the start of
    the downstroke (incomplete pump fillage), so load drops sharply
    ("pounds off") partway through the downstroke instead of decreasing
    smoothly — classic sharp vertical drop on the card's right side.
    """
    n = len(points)
    half = n // 2
    modified = list(points)
    pound_start_idx = half + int(half * 0.15)  # pound occurs early in downstroke
    pound_load_drop = 0.55  # fraction of load lost instantly
    for i in range(pound_start_idx, min(pound_start_idx + int(half * 0.25), n)):
        pos, load = modified[i]
        modified[i] = (pos, load * pound_load_drop)
    return modified


def _apply_gas_interference(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Gas interference: trapped gas compresses instead of the valve
    opening cleanly -> corners round off heavily and the overall card
    area (enclosed loop) shrinks, since less net lifting work is done
    per stroke.
    """
    modified = []
    for pos, load in points:
        min_l = min(l for _, l in points)
        max_l = max(l for _, l in points)
        # compress the load range toward the middle -> rounded, smaller card
        center = (min_l + max_l) / 2
        compressed = center + (load - center) * 0.6
        modified.append((pos, compressed))
    return modified


def _apply_rod_parting(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Rod parting (broken rod string): load collapses to near-zero for a
    contiguous portion of the stroke (the rod above the break just falls,
    no load is transmitted below the break point).
    """
    n = len(points)
    modified = list(points)
    collapse_start = int(n * 0.35)
    collapse_end = int(n * 0.75)
    min_load = min(l for _, l in points)
    for i in range(collapse_start, collapse_end):
        pos, _ = modified[i]
        modified[i] = (pos, min_load * random.uniform(0.05, 0.2))
    return modified


def _apply_tubing_leak(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Tubing leak: fluid leaks back through a hole in the tubing string
    above the pump, so peak load is reduced overall (less net fluid
    weight lifted) and the card area shrinks, but WITHOUT the sharp
    pound-off or full collapse seen in the other fault types -- a more
    subtle, uniformly "deflated" card. This is often the hardest fault
    for junior engineers to distinguish from normal thermal decline,
    which is exactly why the CSS-phase-aware model matters.
    """
    modified = []
    for pos, load in points:
        min_l = min(l for _, l in points)
        deflated = min_l + (load - min_l) * 0.68
        modified.append((pos, deflated))
    return modified


_FAULT_APPLICATORS = {
    PumpCondition.FLUID_POUND: _apply_fluid_pound,
    PumpCondition.GAS_INTERFERENCE: _apply_gas_interference,
    PumpCondition.ROD_PARTING: _apply_rod_parting,
    PumpCondition.TUBING_LEAK: _apply_tubing_leak,
}


def generate_dyno_card(params: DynoCardParams, viscosity_cp: float,
                        condition: PumpCondition = PumpCondition.NORMAL,
                        noise_level: float = 0.015) -> List[Tuple[float, float]]:
    """
    Generate one full dynamometer card (list of (position, load) points)
    for a given fluid viscosity and pump condition, with light sensor
    noise added for realism.
    """
    base = _base_normal_card(params, viscosity_cp)

    if condition != PumpCondition.NORMAL:
        base = _FAULT_APPLICATORS[condition](base)

    # Add light gaussian-ish noise to avoid an unrealistically perfect curve
    noisy = []
    load_range = params.polished_rod_max_load_kg - params.polished_rod_min_load_kg
    for pos, load in base:
        noise = random.gauss(0, noise_level * load_range)
        noisy.append((round(pos, 4), round(max(0.0, load + noise), 2)))

    return noisy


def card_shape_features(card: List[Tuple[float, float]]) -> dict:
    """
    Extract simple summary features from a card — these are exactly the
    kind of inputs a classical ML classifier (XGBoost, etc.) would use
    alongside CSS-phase context. Kept intentionally simple/interpretable
    for hackathon judge Q&A ("why did the model flag this?").
    """
    loads = [l for _, l in card]
    peak = max(loads)
    trough = min(loads)
    card_area = _shoelace_area(card)  # enclosed loop area = net work per stroke
    return {
        "peak_load_kg": round(peak, 2),
        "min_load_kg": round(trough, 2),
        "load_range_kg": round(peak - trough, 2),
        "card_area": round(card_area, 2),
    }


def _shoelace_area(points: List[Tuple[float, float]]) -> float:
    """Shoelace formula for polygon area — used as a proxy for 'card area',
    a standard real-world dynamometer diagnostic (larger area = more net
    useful work per stroke; sudden area drop = a developing fault)."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


if __name__ == "__main__":
    # Sanity test: generate a normal card at low viscosity (post-steam)
    # and at high viscosity (cooled), plus each fault type, and print
    # summary features to confirm they're distinguishable.
    params = DynoCardParams()

    print("=== Viscosity coupling demo (THE key differentiator) ===")
    for visc, label in [(150, "just after steam (thin oil)"),
                         (5000, "weeks into production (cooling)"),
                         (11000, "fully cooled (cold baseline)")]:
        card = generate_dyno_card(params, visc, PumpCondition.NORMAL)
        feats = card_shape_features(card)
        print(f"  visc={visc:>6}cP ({label:<32}) -> {feats}")

    print("\n=== Fault type comparison (all at fixed mid viscosity=1500cP) ===")
    for condition in PumpCondition:
        card = generate_dyno_card(params, 1500, condition)
        feats = card_shape_features(card)
        print(f"  {condition.value:<20} -> {feats}")
