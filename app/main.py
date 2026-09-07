"""
AI-DRISHTI — Main FastAPI Application
----------------------------------------

Connects:
    - Well registry / physics simulation
    - Live WebSocket telemetry
    - SRP optimizer
    - Rod floating detection
    - SOR tracker
    - Live CSS cycle optimizer
    - Well Copilot

Run from the backend/ directory:

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Swagger:
    http://127.0.0.1:8000/docs
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =====================================================================
# Simulation / Well Registry
# =====================================================================

from app.simulation.well_registry import (
    initialize_all_wells,
    get_well_ids,
    get_well_meta,
    get_well_points,
    get_well_summary,
    get_all_wells_overview,
)

from app.simulation.replay_service import (
    stream_well_telemetry,
    get_single_snapshot,
)


# =====================================================================
# Physics / Optimization
# =====================================================================

from app.physics.srp_optimizer import (
    recommended_safe_spm,
    field_practice_spm,
    assess_rod_floating,
)

from app.physics.sor_tracker import (
    compute_sor,
    estimate_sor_improvement_from_spm_fix,
)

from app.physics.css_optimizer import (
    optimize_css_cycle,
    CSSLiveState,
)

from app.physics.thermal_model import (
    WellReservoirParams,
)


# =====================================================================
# Application startup
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Precompute the representative well simulations once.
    initialize_all_wells()

    yield


app = FastAPI(
    title="AI-DRISHTI API",
    version="0.1.0",
    lifespan=lifespan,
)


# =====================================================================
# CORS
# =====================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# WebSocket subscriber management
# =====================================================================

_subscribers: Dict[str, List[WebSocket]] = {}

_stream_tasks: Dict[str, asyncio.Task] = {}


# =====================================================================
# Health
# =====================================================================

@app.get("/health")
async def health_check():

    return {
        "status": "ok",
        "service": "AI-DRISHTI API",
        "version": "0.1.0",
    }


# =====================================================================
# Wells
# =====================================================================

@app.get("/api/wells")
async def list_wells():

    return get_all_wells_overview()


@app.get("/api/wells/{well_id}")
async def well_details(well_id: str):

    meta = get_well_meta(well_id)

    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown well_id: {well_id}",
        )

    points = get_well_points(well_id)
    summary = get_well_summary(well_id)

    return {
        "well": meta,
        "summary": summary,
        "points": points,
    }


# =====================================================================
# Well timeline
# =====================================================================

@app.get("/api/wells/{well_id}/timeline")
async def well_timeline(well_id: str):

    if get_well_meta(well_id) is None:

        raise HTTPException(
            status_code=404,
            detail=f"Unknown well_id: {well_id}",
        )

    return get_well_points(well_id)


# =====================================================================
# Well snapshot
# =====================================================================

@app.get("/api/wells/{well_id}/snapshot")
async def well_snapshot(
    well_id: str,
    at_hours: float | None = None,
):

    try:

        return await get_single_snapshot(
            well_id,
            at_hours,
        )

    except ValueError:

        raise HTTPException(
            status_code=404,
            detail=f"Unknown well_id: {well_id}",
        )


# =====================================================================
# SRP OPTIMIZER
# =====================================================================

class SPMRequest(BaseModel):

    viscosity_cp: float

    current_spm: float | None = None


@app.post("/api/optimizer/spm-recommendation")
async def spm_recommendation(req: SPMRequest):

    """
    SRP optimization.

    Given current oil viscosity:

        viscosity → safe SPM recommendation

    If current SPM is supplied:

        viscosity + current SPM
        → rod floating risk assessment
    """

    rec_spm = recommended_safe_spm(
        req.viscosity_cp
    )

    field_spm = field_practice_spm(
        req.viscosity_cp
    )

    result = {

        "viscosity_cp":
            req.viscosity_cp,

        "recommended_spm":
            rec_spm,

        "field_practice_spm_for_comparison":
            field_spm,
    }

    # ---------------------------------------------------------------
    # Rod floating assessment
    # ---------------------------------------------------------------

    if req.current_spm is not None:

        assessment = assess_rod_floating(
            req.viscosity_cp,
            req.current_spm,
        )

        result["rod_floating_assessment"] = {

            "is_floating_risk":
                assessment.is_floating_risk,

            "severity":
                assessment.severity,

            "predicted_fillage_pct":
                assessment.predicted_fillage_pct,

            "explanation":
                assessment.explanation,
        }

    return result


# =====================================================================
# SOR TRACKER
# =====================================================================

class SORRequest(BaseModel):

    steam_tonnes: float

    oil_bbl: float

    baseline_avg_fillage_pct: float | None = None

    optimized_avg_fillage_pct: float | None = None


@app.post("/api/optimizer/sor")
async def sor_calculation(req: SORRequest):

    """
    Steam-Oil Ratio calculation.
    """

    result = compute_sor(
        req.steam_tonnes,
        req.oil_bbl,
    )

    response = {

        "total_steam_bbl":
            result.total_steam_bbl,

        "total_oil_bbl":
            result.total_oil_bbl,

        "sor":
            result.sor,

        "efficiency_note":
            result.efficiency_note,
    }

    # ---------------------------------------------------------------
    # Optional SOR improvement estimate
    # ---------------------------------------------------------------

    if (
        req.baseline_avg_fillage_pct is not None
        and
        req.optimized_avg_fillage_pct is not None
    ):

        response["improvement_estimate"] = (
            estimate_sor_improvement_from_spm_fix(
                result.sor,
                req.baseline_avg_fillage_pct,
                req.optimized_avg_fillage_pct,
            )
        )

    return response


# =====================================================================
# LIVE CSS OPTIMIZER
# =====================================================================

class CSSOptimizeRequest(BaseModel):

    # Normal CSS optimization parameters
    injection_days: float = 5.0

    fixed_cycle_length_days: float = 70.0

    steam_volume_options_bbl: List[float] | None = None

    soak_days_options: List[float] | None = None

    # ---------------------------------------------------------------
    # LIVE WELL CONDITIONS
    # ---------------------------------------------------------------

    current_temperature_c: float | None = None

    current_viscosity_cp: float | None = None

    current_oil_rate_bpd: float | None = None

    current_phase: str | None = None


@app.post("/api/optimizer/css-cycle")
async def css_cycle_optimization(
    req: CSSOptimizeRequest,
):

    """
    LIVE CSS CYCLE OPTIMIZER

    The optimizer considers:

        Current temperature
                ↓
        Current viscosity
                ↓
        Current oil production rate
                ↓
        Current well phase
                ↓
        Physics-based CSS simulation
                ↓
        Candidate steam + soak combinations
                ↓
        Best CSS recommendation

    This allows the recommendation to change
    as the well condition changes.
    """

    # ===============================================================
    # Reservoir model
    # ===============================================================

    reservoir = WellReservoirParams()


    # ===============================================================
    # Create LIVE state
    # ===============================================================

    live_state = CSSLiveState(

        temperature_c=
            req.current_temperature_c,

        viscosity_cp=
            req.current_viscosity_cp,

        oil_rate_bpd=
            req.current_oil_rate_bpd,

        phase=
            req.current_phase,
    )


    # ===============================================================
    # Run LIVE optimizer
    # ===============================================================

    result = optimize_css_cycle(

        reservoir,

        injection_days=
            req.injection_days,

        fixed_cycle_length_days=
            req.fixed_cycle_length_days,

        steam_volume_options_bbl=
            req.steam_volume_options_bbl,

        soak_days_options=
            req.soak_days_options,

        live_state=
            live_state,
    )


    # ===============================================================
    # Determine explanation
    # ===============================================================

    temperature = req.current_temperature_c

    viscosity = req.current_viscosity_cp

    oil_rate = req.current_oil_rate_bpd


    # ---------------------------------------------------------------
    # Cold / high-viscosity condition
    # ---------------------------------------------------------------

    if (
        temperature is not None
        and viscosity is not None
        and (
            temperature < 65
            or viscosity > 8000
        )
    ):

        recommendation_reason = (
            "Cooling / high-viscosity state → "
            "higher thermal stimulation is favored."
        )


    # ---------------------------------------------------------------
    # Warm / low-viscosity condition
    # ---------------------------------------------------------------

    elif (
        temperature is not None
        and viscosity is not None
        and temperature > 115
        and viscosity < 3000
    ):

        recommendation_reason = (
            "Warm / lower-viscosity state → "
            "excessive steam input is less favored."
        )


    # ---------------------------------------------------------------
    # Low production condition
    # ---------------------------------------------------------------

    elif (
        oil_rate is not None
        and oil_rate < 100
    ):

        recommendation_reason = (
            "Low production state → "
            "additional thermal stimulation is favored."
        )


    # ---------------------------------------------------------------
    # Moderate condition
    # ---------------------------------------------------------------

    else:

        recommendation_reason = (
            "Moderate well condition → "
            "balanced steam and soak strategy is favored."
        )


    # ===============================================================
    # Return response
    # ===============================================================

    return {

        # -----------------------------------------------------------
        # FINAL RECOMMENDATION
        # -----------------------------------------------------------

        "recommended": {

            "steam_volume_bbl":
                result.recommended.steam_volume_bbl,

            "soak_days":
                result.recommended.soak_days,

            "production_days":
                result.recommended.production_days,

            "expected_total_oil_bbl":
                result.recommended.total_oil_bbl,

            "expected_peak_rate_bpd":
                result.recommended.peak_rate_bpd,

            "expected_sor":
                result.recommended.sor,
        },


        # -----------------------------------------------------------
        # LIVE STATE USED BY AI
        # -----------------------------------------------------------

        "state_used": {

            "temperature_c":
                req.current_temperature_c,

            "viscosity_cp":
                req.current_viscosity_cp,

            "oil_rate_bpd":
                req.current_oil_rate_bpd,

            "phase":
                req.current_phase,
        },


        # -----------------------------------------------------------
        # EXPLAINABLE AI DECISION
        # -----------------------------------------------------------

        "recommendation_reason":
            recommendation_reason,


        # -----------------------------------------------------------
        # TOP CSS CANDIDATES
        # -----------------------------------------------------------

        "top_candidates": [

            {

                "steam_volume_bbl":
                    c.steam_volume_bbl,

                "soak_days":
                    c.soak_days,

                "production_days":
                    c.production_days,

                "total_oil_bbl":
                    c.total_oil_bbl,

                "sor":
                    c.sor,

                "net_score":
                    c.net_score,
            }

            for c in result.all_candidates[:10]
        ],
    }


# =====================================================================
# COPILOT
# =====================================================================

class CopilotRequest(BaseModel):

    well_id: str

    question: str


@app.post("/api/copilot/ask")
async def copilot_ask(req: CopilotRequest):

    """
    Well Copilot endpoint.

    The actual Copilot implementation can be
    connected here.
    """

    try:

        from app.copilot import answer_question, build_well_context

    except ImportError:

        raise HTTPException(
            status_code=500,
            detail="Copilot module is not available.",
        )


    ctx = build_well_context(
        req.well_id
    )

    if ctx is None:

        raise HTTPException(
            status_code=404,
            detail=f"Unknown well_id: {req.well_id}",
        )


    result = answer_question(
        req.question,
        ctx,
    )


    return {

        "well_id":
            req.well_id,

        "question":
            req.question,

        "answer":
            result["answer"],

        "source":
            result["source"],
    }


# =====================================================================
# LIVE TELEMETRY WEBSOCKET
# =====================================================================

async def _run_replay_and_broadcast(
    well_id: str,
):

    """
    Background task.

    Reads telemetry from the replay service and
    broadcasts it to all connected clients watching
    the same well.
    """

    try:

        async for message in stream_well_telemetry(
            well_id,
            speed_multiplier=500.0,
            tick_seconds=0.4,
        ):

            dead = []


            # -------------------------------------------------------
            # Broadcast telemetry
            # -------------------------------------------------------

            for ws in _subscribers.get(
                well_id,
                [],
            ):

                try:

                    await ws.send_json(
                        message
                    )

                except Exception:

                    dead.append(ws)


            # -------------------------------------------------------
            # Remove dead connections
            # -------------------------------------------------------

            for ws in dead:

                if ws in _subscribers.get(
                    well_id,
                    [],
                ):

                    _subscribers[
                        well_id
                    ].remove(ws)


            # -------------------------------------------------------
            # Stop if nobody is listening
            # -------------------------------------------------------

            if not _subscribers.get(
                well_id
            ):

                break


    except asyncio.CancelledError:

        pass


    finally:

        _stream_tasks.pop(
            well_id,
            None,
        )


# =====================================================================
# WebSocket endpoint
# =====================================================================

@app.websocket("/ws/live/{well_id}")
async def live_well_feed(
    websocket: WebSocket,
    well_id: str,
):

    # ---------------------------------------------------------------
    # Validate well
    # ---------------------------------------------------------------

    if get_well_meta(well_id) is None:

        await websocket.close(
            code=4004,
            reason=f"Unknown well_id: {well_id}",
        )

        return


    # ---------------------------------------------------------------
    # Accept connection
    # ---------------------------------------------------------------

    await websocket.accept()


    _subscribers.setdefault(
        well_id,
        [],
    ).append(websocket)


    print(
        f"[ws] Client subscribed to "
        f"{well_id} "
        f"({len(_subscribers[well_id])} total)"
    )


    # ---------------------------------------------------------------
    # Start shared replay task
    # ---------------------------------------------------------------

    if well_id not in _stream_tasks:

        _stream_tasks[
            well_id
        ] = asyncio.create_task(
            _run_replay_and_broadcast(
                well_id
            )
        )


    # ---------------------------------------------------------------
    # Keep connection alive
    # ---------------------------------------------------------------

    try:

        while True:

            await websocket.receive_text()


    except WebSocketDisconnect:

        pass


    except Exception:

        pass


    finally:

        if websocket in _subscribers.get(
            well_id,
            [],
        ):

            _subscribers[
                well_id
            ].remove(websocket)


        print(
            f"[ws] Client disconnected from "
            f"{well_id}"
        )


# =====================================================================
# Run directly
# =====================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )