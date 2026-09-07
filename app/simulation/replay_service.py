"""
AI-DRISHTI — Live Replay Service
------------------------------------
Streams a well's precomputed physics simulation (from well_registry) as
if it were live SCADA telemetry, at a configurable playback speed (e.g.
compress 200+ days of simulated history into a few minutes of demo
time). This is the software equivalent of what the hardware rig would
have streamed — same downstream schema, same WebSocket contract.

During production-phase ticks, also generates a matching dynamometer
card point using the CURRENT tick's viscosity — this is where the
CSS-SRP coupling becomes visible in the live demo: the card shape
visibly changes as viscosity rises through a cycle's production phase.
"""

import asyncio
import random
import time
from typing import AsyncGenerator, Optional

from app.physics.dynocard_synthesizer import (
    DynoCardParams,
    PumpCondition,
    generate_dyno_card,
)
from app.simulation.well_registry import get_well_points

_dyno_params = DynoCardParams()

# How often (in simulated points) to inject a demo "fault" into the
# dynamometer card stream, purely for live-demo drama — a real
# deployment would derive condition from actual sensor anomalies, not
# a timer. Set to None to disable.
DEMO_FAULT_EVERY_N_TICKS = 45
DEMO_FAULT_DURATION_TICKS = 8


def _pick_demo_condition(tick_index: int) -> PumpCondition:
    """
    Periodically injects a fault-like card shape into the stream so a
    live demo has something dramatic to point at, independent of
    natural CSS-phase drift. Cycles through fault types for variety.
    """
    if DEMO_FAULT_EVERY_N_TICKS is None:
        return PumpCondition.NORMAL
    cycle_pos = tick_index % DEMO_FAULT_EVERY_N_TICKS
    if cycle_pos < DEMO_FAULT_DURATION_TICKS:
        fault_types = [PumpCondition.FLUID_POUND, PumpCondition.GAS_INTERFERENCE,
                       PumpCondition.ROD_PARTING, PumpCondition.TUBING_LEAK]
        fault_index = (tick_index // DEMO_FAULT_EVERY_N_TICKS) % len(fault_types)
        return fault_types[fault_index]
    return PumpCondition.NORMAL


async def stream_well_telemetry(
    well_id: str,
    speed_multiplier: float = 500.0,
    tick_seconds: float = 0.5,
    loop: bool = True,
) -> AsyncGenerator[dict, None]:
    """
    Async generator yielding one telemetry message per tick.

    speed_multiplier: how many simulated hours pass per real-time tick.
        e.g. tick_seconds=0.5 and speed_multiplier=500 means every 0.5
        real seconds advances the simulation by ~1 sim-hour worth of
        points at the underlying 2-hour timestep resolution — tune
        these two numbers together to control overall demo pacing.
    tick_seconds: real-world delay between yielded messages.
    loop: if True, restarts from the beginning after reaching the end
        (good for an unattended booth demo); if False, stops.
    """
    points = get_well_points(well_id)
    if not points:
        raise ValueError(f"No simulation data for well_id={well_id}")

    tick_index = 0
    while True:
        for point in points:
            message = {
                "type": "telemetry_update",
                "well_id": well_id,
                "source": "physics_simulator",
                "elapsed_hours": point.elapsed_hours,
                "elapsed_days": round(point.elapsed_hours / 24.0, 2),
                "phase": point.phase.value,
                "bottomhole_temp_c": point.bottomhole_temp_c,
                "oil_viscosity_cp": point.oil_viscosity_cp,
                "oil_rate_bpd": point.oil_rate_bpd,
                "cumulative_oil_bbl": point.cumulative_oil_bbl,
                "pump_status": point.pump_status,
                "pump_fillage_pct": point.pump_fillage_pct,
                "server_time": time.time(),
            }

            # Only production-phase ticks have a meaningful dynamometer
            # card (pump isn't stroking during injection/soak)
            if point.phase.value == "production":
                condition = _pick_demo_condition(tick_index)
                card = generate_dyno_card(_dyno_params, point.oil_viscosity_cp, condition)
                message["dyno_card"] = {
                    "condition": condition.value,
                    "points": [{"position": p, "load": l} for p, l in card],
                    "is_demo_injected_fault": condition != PumpCondition.NORMAL,
                }
                tick_index += 1

            yield message
            await asyncio.sleep(tick_seconds)

        if not loop:
            break


async def get_single_snapshot(well_id: str, at_hours: Optional[float] = None) -> dict:
    """
    Non-streaming helper: get one telemetry snapshot at a specific
    elapsed-hours point (or the last point if not specified). Used by
    the Time Machine timeline scrubber's REST fallback / initial load.
    """
    points = get_well_points(well_id)
    if not points:
        raise ValueError(f"No simulation data for well_id={well_id}")

    if at_hours is None:
        target = points[-1]
    else:
        target = min(points, key=lambda p: abs(p.elapsed_hours - at_hours))

    result = {
        "well_id": well_id,
        "elapsed_hours": target.elapsed_hours,
        "elapsed_days": round(target.elapsed_hours / 24.0, 2),
        "phase": target.phase.value,
        "bottomhole_temp_c": target.bottomhole_temp_c,
        "oil_viscosity_cp": target.oil_viscosity_cp,
        "oil_rate_bpd": target.oil_rate_bpd,
        "cumulative_oil_bbl": target.cumulative_oil_bbl,
        "pump_status": target.pump_status,
        "pump_fillage_pct": target.pump_fillage_pct,
    }

    if target.phase.value == "production":
        card = generate_dyno_card(_dyno_params, target.oil_viscosity_cp, PumpCondition.NORMAL)
        result["dyno_card"] = {"points": [{"position": p, "load": l} for p, l in card]}

    return result
