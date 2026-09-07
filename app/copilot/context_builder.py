"""
AI-DRISHTI Copilot — Context Builder
----------------------------------------
Gathers real, calculated facts about a well into a structured summary
that the LLM (or the rule-based fallback) answers FROM — never asked
to invent numbers itself. This is the grounding step that keeps the
copilot honest: every number it can mention already came from the
same calibrated physics engine powering the rest of the app.
"""

from dataclasses import dataclass
from typing import Optional

from app.simulation.well_registry import get_well_meta, get_well_points, get_well_summary
from app.physics.srp_optimizer import recommended_safe_spm, field_practice_spm, assess_rod_floating
from app.physics.css_optimizer import optimize_css_cycle
from app.physics.thermal_model import WellReservoirParams


@dataclass
class WellContext:
    well_id: str
    well_name: str
    reservoir_depth_m: float
    vit_completion: bool
    current_phase: str
    current_temp_c: float
    current_viscosity_cp: float
    current_rate_bpd: float
    cumulative_oil_bbl: float
    pump_status: str
    pump_fillage_pct: float
    field_practice_spm: float
    recommended_spm: float
    rod_floating_severity: str
    rod_floating_explanation: str
    css_recommended_steam_bbl: float
    css_recommended_soak_days: float
    css_expected_sor: float

    def as_prompt_text(self) -> str:
        """Render as a compact, LLM-friendly fact sheet."""
        return f"""Well: {self.well_name} (Baghewala field)
Reservoir depth: {self.reservoir_depth_m:.0f} m
Completion: {'VIT (Vacuum Insulated Tubing)' if self.vit_completion else 'Standard'}

CURRENT STATE:
- Cycle phase: {self.current_phase}
- Bottomhole temperature: {self.current_temp_c:.1f} C
- Oil viscosity: {self.current_viscosity_cp:.0f} cP
- Oil rate: {self.current_rate_bpd:.1f} bpd
- Cumulative production this cycle: {self.cumulative_oil_bbl:.0f} bbl
- Pump status: {self.pump_status} ({self.pump_fillage_pct:.1f}% fillage)

SPM / ROD FLOATING ANALYSIS (for current viscosity):
- Field's current practice would run: {self.field_practice_spm} SPM
- AI-DRISHTI recommends: {self.recommended_spm} SPM
- Rod floating risk severity: {self.rod_floating_severity}
- Explanation: {self.rod_floating_explanation}

CSS CYCLE OPTIMIZER RECOMMENDATION:
- Recommended steam volume: {self.css_recommended_steam_bbl:.0f} bbl
- Recommended soak time: {self.css_recommended_soak_days:.0f} days
- Expected Steam-Oil Ratio (SOR): {self.css_expected_sor:.3f}
"""


def build_well_context(well_id: str) -> Optional[WellContext]:
    meta = get_well_meta(well_id)
    if meta is None:
        return None

    points = get_well_points(well_id)
    latest = points[-1] if points else None
    if latest is None:
        return None

    viscosity = latest.oil_viscosity_cp
    field_spm = field_practice_spm(viscosity)
    rec_spm = recommended_safe_spm(viscosity)
    assessment = assess_rod_floating(viscosity, field_spm)

    # Run the CSS optimizer fresh (cheap enough to compute on demand)
    reservoir = WellReservoirParams(
        reservoir_depth_m=meta.reservoir_depth_m,
        vit_completion=meta.vit_completion,
    )
    css_result = optimize_css_cycle(reservoir)

    return WellContext(
        well_id=well_id,
        well_name=meta.well_name,
        reservoir_depth_m=meta.reservoir_depth_m,
        vit_completion=meta.vit_completion,
        current_phase=latest.phase.value,
        current_temp_c=latest.bottomhole_temp_c,
        current_viscosity_cp=viscosity,
        current_rate_bpd=latest.oil_rate_bpd,
        cumulative_oil_bbl=latest.cumulative_oil_bbl,
        pump_status=latest.pump_status,
        pump_fillage_pct=latest.pump_fillage_pct,
        field_practice_spm=field_spm,
        recommended_spm=rec_spm,
        rod_floating_severity=assessment.severity,
        rod_floating_explanation=assessment.explanation,
        css_recommended_steam_bbl=css_result.recommended.steam_volume_bbl,
        css_recommended_soak_days=css_result.recommended.soak_days,
        css_expected_sor=css_result.recommended.sor,
    )
