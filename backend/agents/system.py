"""
SmartVolt Agno multi-agent system.

Design: the NUMBERS are computed by deterministic Python tools (N-BEATSx forecast
+ the recommendation engine) so they are always correct. The Agno LLM agents add
the reasoning layer: a plain-language explanation of the plan and a chat assistant.

If no LLM API key is configured, the system still works in "direct" mode using a
built-in template explanation, so the app never breaks.

Environment:
  SMARTVOLT_LLM_PROVIDER = groq | google | openai   (default groq)
  GROQ_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY
"""
import os
import warnings
warnings.filterwarnings("ignore")

from .recommender import recommend

PROVIDER = os.environ.get("SMARTVOLT_LLM_PROVIDER", "groq").lower()

# Which chat model to use on each provider. These can be overridden with an
# environment variable, so if a provider retires a model you can switch without
# touching the code. Groq retired "llama-3.3-70b-versatile" (deprecated 16 Aug
# 2026), which is why the old hard-coded value stopped working; the default
# below is a current Groq production model.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _has_key():
    return bool(os.environ.get("GROQ_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("OPENAI_API_KEY"))


def _make_model():
    """Create an Agno model for the configured free provider."""
    if os.environ.get("GROQ_API_KEY"):
        from agno.models.groq import Groq
        return Groq(id=GROQ_MODEL)
    if os.environ.get("GOOGLE_API_KEY"):
        from agno.models.google import Gemini
        return Gemini(id=GOOGLE_MODEL)
    if os.environ.get("OPENAI_API_KEY"):
        from agno.models.openai import OpenAIChat
        return OpenAIChat(id=OPENAI_MODEL)
    return None


def _plan_text(plan, forecast):
    """Compact text summary of the numeric plan, fed to the LLM or used directly."""
    lines = [f"Day: {plan['day']}",
             f"Average price: {plan['average_price']} EUR/MWh",
             f"Cheapest hours: {', '.join(plan['cheapest_hours'])}",
             f"Most expensive hours: {', '.join(plan['most_expensive_hours'])}",
             "Appliance advice:"]
    for r in plan["appliance_recommendations"]:
        lines.append(f"  - {r['appliance']}: best at {r['best_start']}, "
                     f"saves EUR {r['saving_eur']}")
    b = plan["battery_advice"]
    lines.append(f"Battery: charge {', '.join(b['charge_hours'])}, "
                 f"discharge {', '.join(b['discharge_hours'])} "
                 f"({b['saving_eur_per_mwh']} EUR/MWh spread)")
    return "\n".join(lines)


def _fmt(hhmm):
    """24h 'HH:MM' -> friendly '1 PM' / '12 AM'."""
    try:
        h, m = [int(x) for x in hhmm.split(":")]
    except Exception:
        return hhmm
    ap = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ap}" if m else f"{h12} {ap}"


def _template_summary(plan):
    """Plain-language summary WITHOUT an LLM (fallback mode)."""
    ch = _fmt(plan["cheapest_hours"][0])
    ex = _fmt(plan["most_expensive_hours"][0])
    top = max(plan["appliance_recommendations"], key=lambda r: r["saving_eur"], default=None)
    parts = [f"On {plan['day']} electricity is cheapest around {ch} and most expensive "
             f"around {ex} (average {plan['average_price']} EUR/MWh)."]
    if top:
        parts.append(f"The biggest saving is to run your {top['appliance'].lower()} "
                     f"at {_fmt(top['best_start'])}.")
    parts.append("Charge batteries or the EV during the cheap night hours and avoid "
                 "heavy use during the expensive block.")
    return " ".join(parts)


# ---- Agno agents (built only when an LLM key exists) ----
def build_supervisor(model):
    """A supervisor Team coordinating two specialist agents (Agno 2.x API)."""
    from agno.agent import Agent
    from agno.team import Team
    forecast_expert = Agent(
        name="Forecast Analyst",
        model=model,
        instructions=["You explain electricity price forecasts and their risk in simple terms."],
    )
    strategy_expert = Agent(
        name="Energy Strategy Advisor",
        model=model,
        instructions=["You turn a price plan into friendly, practical money-saving advice for a household."],
    )
    supervisor = Team(
        name="SmartVolt Supervisor",
        model=model,
        members=[forecast_expert, strategy_expert],
        instructions=[
            "You coordinate an electricity money-saving assistant for Spain.",
            "Use ONLY the numbers you are given; never invent prices or hours.",
            "Give a short, clear, friendly explanation a normal person understands.",
        ],
    )
    return supervisor


def _agent(model, name, instructions):
    """A single plain Agno agent (no tools/no delegation -> reliable with Groq)."""
    from agno.agent import Agent
    return Agent(name=name, model=model, instructions=instructions)


def _run(agent, prompt):
    resp = agent.run(prompt)
    text = (getattr(resp, "content", None) or str(resp)).strip()
    # Some providers return an API error as the message content instead of
    # raising (e.g. a decommissioned model). Detect that and raise so the
    # caller falls back to the built-in answer rather than showing raw JSON.
    low = text.lower()
    if text.startswith("{") and '"error"' in low and (
            "model_not_found" in low or "does not exist" in low
            or "invalid_request_error" in low or "decommission" in low):
        raise RuntimeError("LLM returned an error response: " + text[:200])
    return text


def explain(plan, forecast):
    """Two agents collaborate: a Forecast Analyst reads the risk, then an Energy
    Strategy Advisor turns it into friendly household advice. Orchestrated in
    Python so it works reliably on the free Groq model."""
    if not _has_key():
        return {"mode": "direct", "summary": _template_summary(plan)}
    try:
        model = _make_model()
        numbers = _plan_text(plan, forecast)
        analyst = _agent(model, "Forecast Analyst",
                         ["You explain electricity price forecasts for Spain in simple terms.",
                          "Use ONLY the numbers given; never invent prices or hours.",
                          "Reply in 2-3 short sentences: the price shape of the day and any risky hours."])
        analysis = _run(analyst, "Summarise today's price picture from these numbers:\n\n" + numbers)

        advisor = _agent(model, "Energy Strategy Advisor",
                         ["You give friendly, practical money-saving advice to a Spanish household.",
                          "Use ONLY the numbers given; never invent prices or hours.",
                          "Write 4-5 warm, clear sentences on the best times to use electricity and why."])
        advice = _run(advisor, "A forecast analyst noted:\n" + analysis +
                      "\n\nNow write the household advice using these numbers:\n\n" + numbers)
        return {"mode": "agent", "summary": advice}
    except Exception as e:
        return {"mode": "direct", "summary": _template_summary(plan),
                "note": f"LLM unavailable ({e}); used built-in explanation."}


def chat(question, plan, forecast):
    """AI assistant: answer a user's question about the plan (single reliable agent)."""
    if not _has_key():
        return {"mode": "direct",
                "answer": "The AI chat needs a free LLM key (Groq or Gemini) to be set. "
                          "Meanwhile, here is the plan:\n" + _template_summary(plan)}
    try:
        model = _make_model()
        assistant = _agent(model, "SmartVolt Assistant",
                           ["You are a helpful electricity money-saving assistant for Spain.",
                            "Answer using ONLY the plan numbers provided; never invent prices or hours.",
                            "If the question is unrelated to the energy plan, gently say you can only help "
                            "with the electricity plan.",
                            "Be short, clear and friendly."])
        answer = _run(assistant, f"User question: {question}\n\n"
                      f"Use only this electricity plan for Spain:\n{_plan_text(plan, forecast)}")
        return {"mode": "agent", "answer": answer}
    except Exception as e:
        return {"mode": "direct", "answer": f"LLM unavailable ({e}). Plan:\n{_template_summary(plan)}"}
