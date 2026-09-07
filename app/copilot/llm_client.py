"""
AI-DRISHTI Copilot — LLM Client with Guaranteed Fallback
--------------------------------------------------------------
Calls Groq's free, fast API (OpenAI-compatible) to answer questions
grounded in the WellContext fact sheet. If no API key is configured,
the request fails, or there's no internet access, this automatically
falls back to a rule-based templated answer built from the SAME
context data — so the copilot demo NEVER breaks, with or without a
network connection.

To enable the real LLM:
    1. Get a free key at https://console.groq.com/keys
    2. Set environment variable: GROQ_API_KEY=your_key_here
    3. Restart the backend

Without a key set, every question still gets answered — just from the
simpler rule-based path instead of natural free-form language.
"""

import os
import re
from typing import Optional

import requests

from app.copilot.context_builder import WellContext

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are the AI-DRISHTI Well Copilot, embedded in a digital twin \
dashboard for Baghewala heavy-oil wells (CSS + Sucker Rod Pump operations).

You will be given a FACT SHEET of real, already-calculated numbers for one well. \
Answer the user's question using ONLY those facts \u2014 do not invent numbers, \
well names, or data that isn't in the fact sheet. If the fact sheet doesn't \
contain what's needed to answer, say so plainly rather than guessing.

Keep answers concise (2-5 sentences unless the question needs a list), in plain \
engineering English a field operator would understand \u2014 avoid unnecessary \
jargon, and when you do use a technical term (SPM, fillage, SOR), briefly say \
what it means in-line.
"""


def _call_groq(question: str, context_text: str) -> Optional[str]:
    """Returns the LLM's answer, or None if the call fails for any reason."""
    if not GROQ_API_KEY:
        return None

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"FACT SHEET:\n{context_text}\n\nQUESTION: {question}"},
                ],
                "temperature": 0.3,
                "max_tokens": 400,
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[copilot] Groq call failed, falling back to rule-based answer: {e}")
        return None


def _rule_based_fallback(question: str, ctx: WellContext) -> str:
    """
    Simple keyword-routed templated answers, built from the exact same
    context data the LLM would have used. Guarantees the copilot always
    answers something grounded, even fully offline.
    """
    q = question.lower()

    if any(k in q for k in ["spm", "speed", "stroke", "pump", "rod float", "floating",
                             "slow down", "speed up", "faster", "slower"]):
        return (
            f"At {ctx.well_name}'s current viscosity ({ctx.current_viscosity_cp:.0f} cP), "
            f"field practice would run {ctx.field_practice_spm} strokes per minute (SPM), "
            f"but that carries a '{ctx.rod_floating_severity}' rod floating risk. "
            f"AI-DRISHTI recommends {ctx.recommended_spm} SPM instead \u2014 slower, "
            f"which keeps the pump filling properly instead of outrunning the thickening oil. "
            f"{ctx.rod_floating_explanation}"
        )

    if any(k in q for k in ["steam", "soak", "css", "cycle", "inject"]):
        return (
            f"For {ctx.well_name}'s next CSS cycle, the optimizer recommends "
            f"{ctx.css_recommended_steam_bbl:.0f} bbl of steam with a "
            f"{ctx.css_recommended_soak_days:.0f}-day soak, projecting a Steam-Oil "
            f"Ratio (SOR, steam used per barrel of oil recovered) of "
            f"{ctx.css_expected_sor:.3f}. This comes from simulating multiple "
            f"steam/soak combinations rather than relying on fixed field habits."
        )

    if any(k in q for k in ["status", "health", "how is", "doing", "condition"]):
        return (
            f"{ctx.well_name} is currently in the {ctx.current_phase} phase, running at "
            f"{ctx.current_temp_c:.0f}\u00b0C bottomhole temperature with oil viscosity around "
            f"{ctx.current_viscosity_cp:.0f} cP. Pump status is '{ctx.pump_status}' at "
            f"{ctx.pump_fillage_pct:.1f}% fillage, producing {ctx.current_rate_bpd:.1f} bpd "
            f"({ctx.cumulative_oil_bbl:.0f} bbl cumulative this cycle)."
        )

    if any(k in q for k in ["sor", "steam-oil", "efficiency", "waste"]):
        return (
            f"The CSS optimizer projects a Steam-Oil Ratio of {ctx.css_expected_sor:.3f} "
            f"for {ctx.well_name}'s recommended next cycle "
            f"({ctx.css_recommended_steam_bbl:.0f} bbl steam, "
            f"{ctx.css_recommended_soak_days:.0f}-day soak). Lower SOR means more oil "
            f"recovered per unit of steam used \u2014 following the recommended SPM instead "
            f"of field practice also improves this indirectly, by keeping the pump filling "
            f"properly instead of leaving heated oil unrecovered downhole."
        )

    # Generic fallback — always answers with something grounded
    return (
        f"Here's what I know about {ctx.well_name} right now: {ctx.current_phase} phase, "
        f"{ctx.current_temp_c:.0f}\u00b0C, {ctx.current_viscosity_cp:.0f} cP viscosity, "
        f"pump status '{ctx.pump_status}'. Ask me about pump speed (SPM), steam cycle "
        f"planning, or Steam-Oil Ratio for a more specific answer."
    )


def answer_question(question: str, ctx: WellContext) -> dict:
    """
    Main entry point. Tries the real LLM first, falls back automatically.
    Returns a dict so the API can tell the frontend which path answered
    (useful for an honest "answered via AI" vs "answered via rules" badge).
    """
    context_text = ctx.as_prompt_text()

    llm_answer = _call_groq(question, context_text)
    if llm_answer:
        return {"answer": llm_answer, "source": "llm", "model": GROQ_MODEL}

    fallback_answer = _rule_based_fallback(question, ctx)
    return {"answer": fallback_answer, "source": "rule_based", "model": None}
