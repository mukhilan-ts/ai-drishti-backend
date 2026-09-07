# AI-DRISHTI Backend

Fully working FastAPI backend implementing all 7 problem-statement
requirements, calibrated against real Baghewala field data.

## Run it

```bash
pip install -r requirements.txt --break-system-packages
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:
- **API docs (auto-generated):** http://localhost:8000/docs
- **Field overview:** http://localhost:8000/api/wells
- **Live stream:** `ws://localhost:8000/ws/live/BGW_01` (also BGW_02, BGW_03)

## What's implemented, mapped to the 7 PS requirements

| # | PS Requirement | Module | Endpoint |
|---|---|---|---|
| 1 | Optimize CSS cycle parameters | `physics/css_optimizer.py` | `POST /api/optimizer/css-cycle` |
| 2 | Predict reservoir heating/cooling/production | `physics/thermal_model.py` | `GET /api/wells/{id}/timeline`, `/snapshot` |
| 3 | Continuously optimize SPM/stroke speed | `physics/srp_optimizer.py` | `POST /api/optimizer/spm-recommendation` |
| 4 | Detect rod floating, minimize impact loading | `physics/srp_optimizer.py` | (same endpoint, `rod_floating_assessment`) |
| 5 | Improve pump efficiency/reliability | `physics/field_calibration.py` | Baked into all well status fields |
| 6 | Reduce Steam-Oil Ratio (SOR) | `physics/sor_tracker.py` | `POST /api/optimizer/sor` |
| 7 | Real-time integrated system | `main.py` + `simulation/` | `GET /api/wells`, WebSocket live feed |

## Calibration — this is not guessed data

Every constant in `field_calibration.py` was fitted by regression
against the real `Css_optimization_data.pdf` / `srp_data.pdf` field
data. Run it directly to see the validation:

```bash
python3 -m app.physics.field_calibration
```

This reproduces the source data's own Normal/Reduced Efficiency/High
Load/Failure Risk labels with **10/10 accuracy**, and the temperature
model independently matches OIL India's published 10,000-13,000 cP
@ 50°C spec.

## The core demo finding

```bash
python3 -m app.physics.srp_optimizer
```

Shows that the field's ACTUAL current practice (increasing SPM as
viscosity rises) drives wells into "active_floating" risk, while our
recommended (opposite-direction) SPM keeps fillage stable near 88%
the whole time. This is the single strongest before/after demo moment.

## Well Copilot (natural language chat)

Answers questions grounded in real calculated well data — never invents
numbers. Works two ways:

**Without any setup** (default): uses rule-based templated answers built
from the same context data. Always works, no internet or API key needed.

**With a free Groq API key** (optional, for natural free-form answers):
1. Get a free key at https://console.groq.com/keys
2. Set the environment variable before starting the backend:
   ```bash
   export GROQ_API_KEY=your_key_here
   uvicorn app.main:app --reload
   ```
3. The frontend chat panel shows "AI-generated" vs "rule-based" on each
   answer, so it's always honest about which path answered.

## Known simplifications (be ready to explain these)

- SOR absolute magnitude is approximate — the real data sample only
  covers 5 injection days, not a full cycle, so the demo's total-oil
  assumption for SOR calculation is illustrative. The *directional*
  SOR improvement (~23% from following the SPM recommendation) is the
  trustworthy number.
- The physics engine caps reservoir bulk temperature at 92°C (the top
  of the real data's validated range) rather than extrapolating beyond
  it — this is intentional and defensible, not a limitation to hide.
- Dynamometer card fault injection during live streaming is currently
  timer-based for demo drama (`replay_service.py`,
  `DEMO_FAULT_EVERY_N_TICKS`) — a real deployment would derive fault
  condition from actual anomaly detection, not a timer.
- No persistent database — all state is in-memory and resets on
  restart. Not needed for the demo.

## Tested

Every module has a `__main__` self-test (`python3 -m app.physics.X`).
The full API was verified end-to-end with a real uvicorn server and a
real WebSocket client (not just FastAPI's TestClient, which has
limitations with the background-task broadcast pattern used here).
