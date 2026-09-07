"""
AI-DRISHTI — Well Registry
----------------------------
Defines the set of demo wells (named after real Baghewala well IDs from
the provided field data: BGW_01, BGW_02, BGW_03) and precomputes each
well's full multi-cycle physics simulation ONCE at startup, caching the
result in memory. The replay service then streams from this cache
rather than recomputing physics on every websocket tick.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List

from app.physics.thermal_model import (
    WellReservoirParams,
    CSSCycleParams,
    SimPoint,
    simulate_multiple_cycles,
    summarize_cycle,
)


@dataclass
class WellMeta:
    well_id: str
    well_name: str
    field_name: str = "Baghewala"
    reservoir_depth_m: float = 1150.0
    completion_type: str = "CSS+SRP"
    vit_completion: bool = True


# Demo well definitions — IDs match the real field data provided
# (Css_optimization_data.pdf / srp_data.pdf reference BGW_01, BGW_02).
# BGW_03 is added as a third demo well with slightly different
# reservoir parameters to show field-level variation.
WELL_DEFINITIONS: Dict[str, WellMeta] = {
    "BGW_01": WellMeta(well_id="BGW_01", well_name="BGW-01", vit_completion=True),
    "BGW_02": WellMeta(well_id="BGW_02", well_name="BGW-02", vit_completion=True),
    "BGW_03": WellMeta(well_id="BGW_03", well_name="BGW-03", vit_completion=False),
}

# Number of CSS cycles to precompute per well (feeds the Time Machine
# timeline scrubber, which needs multi-cycle history to be interesting)
CYCLES_PER_WELL = 3

# In-memory cache: well_id -> list of SimPoint across all cycles
_well_cache: Dict[str, List[SimPoint]] = {}
_well_summaries: Dict[str, dict] = {}


def _build_well_simulation(meta: WellMeta) -> List[SimPoint]:
    reservoir = WellReservoirParams(
        reservoir_depth_m=meta.reservoir_depth_m,
        vit_completion=meta.vit_completion,
    )
    cycles = [CSSCycleParams() for _ in range(CYCLES_PER_WELL)]
    return simulate_multiple_cycles(reservoir, cycles, timestep_hours=2.0, max_rate_bpd=260.0)


def initialize_all_wells() -> None:
    """Call once at app startup to precompute every demo well's simulation."""
    for well_id, meta in WELL_DEFINITIONS.items():
        points = _build_well_simulation(meta)
        _well_cache[well_id] = points
        _well_summaries[well_id] = summarize_cycle(points)
    print(f"[well_registry] Initialized {len(_well_cache)} wells, "
          f"{sum(len(p) for p in _well_cache.values())} total simulated points")


def get_well_ids() -> List[str]:
    return list(WELL_DEFINITIONS.keys())


def get_well_meta(well_id: str) -> WellMeta | None:
    return WELL_DEFINITIONS.get(well_id)


def get_well_points(well_id: str) -> List[SimPoint]:
    if well_id not in _well_cache:
        raise KeyError(f"Unknown well_id: {well_id}")
    return _well_cache[well_id]


def get_well_summary(well_id: str) -> dict:
    return _well_summaries.get(well_id, {})


def get_all_wells_overview() -> List[dict]:
    """Used by the Field Overview page — one row per well with latest status."""
    overview = []
    for well_id, meta in WELL_DEFINITIONS.items():
        points = _well_cache.get(well_id, [])
        prod_points = [p for p in points if p.phase.value == "production"]
        latest = prod_points[-1] if prod_points else None
        overview.append({
            "well_id": well_id,
            "well_name": meta.well_name,
            "field_name": meta.field_name,
            "vit_completion": meta.vit_completion,
            "latest_status": latest.pump_status if latest else "N/A",
            "latest_oil_rate_bpd": latest.oil_rate_bpd if latest else 0.0,
            "cumulative_oil_bbl": latest.cumulative_oil_bbl if latest else 0.0,
            "summary": _well_summaries.get(well_id, {}),
        })
    return overview
