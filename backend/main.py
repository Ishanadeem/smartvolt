"""
FastAPI backend for the Electricity Price Forecasting web app.
Serves the frontend and a small JSON API backed by the trained N-BEATSx model.
"""
import os
import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .model_service import ForecastService
from .decision_service import decide
from .database import Database
from . import data_pipeline
from .agents.recommender import recommend
from .agents.system import explain as agent_explain, chat as agent_chat
from .green_service import compute_green
from pydantic import BaseModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(HERE, "frontend")

# If the user pasted a free Groq key into api_key.txt, load it so the AI agents turn on.
_keyfile = os.path.join(HERE, "api_key.txt")
if os.path.exists(_keyfile) and not os.environ.get("GROQ_API_KEY"):
    # Scan for the line that holds the key, so the instruction lines in the
    # file don't stop it loading (the old code required the WHOLE file to
    # start with gsk_, so a pasted key left below the instructions was ignored).
    for _line in open(_keyfile, encoding="utf-8").read().splitlines():
        _line = _line.strip()
        if _line.startswith("gsk_") and "XXXX" not in _line:
            os.environ["GROQ_API_KEY"] = _line
            break

app = FastAPI(title="Electricity Price Forecasting")

# services are loaded once at startup
forecaster = ForecastService()
db = Database()
db.populate_history(forecaster.frame)   # load cleaned prices into price_history


@app.get("/api/data/status")
def data_status():
    return forecaster.data_status()


@app.post("/api/upload")
async def upload(energy: UploadFile = File(...), weather: UploadFile = File(...)):
    tmp = tempfile.mkdtemp()
    try:
        ep = os.path.join(tmp, "energy.csv")
        wp = os.path.join(tmp, "weather.csv")
        with open(ep, "wb") as f:
            shutil.copyfileobj(energy.file, f)
        with open(wp, "wb") as f:
            shutil.copyfileobj(weather.file, f)
        # clean + merge -> data/user_merged.csv (built-in demo file is left intact)
        from .model_service import USER_FILE
        data_pipeline.build_merged(ep, wp, out=USER_FILE)
        forecaster.use_user()                   # switch active data to the upload
        db.populate_history(forecaster.frame)   # refresh price_history
        return {"ok": True, **forecaster.data_status()}
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Could not process the files: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/api/use_builtin")
def use_builtin():
    """Switch the active dataset back to the built-in Spanish demo data."""
    forecaster.use_builtin()
    db.populate_history(forecaster.frame)
    return {"ok": True, **forecaster.data_status()}


@app.get("/api/metrics")
def metrics():
    return {"metrics": db.get_metrics()}


@app.get("/api/forecast")
def forecast(day: str):
    try:
        fc = forecaster.forecast(day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # persist to history so the History page shows every forecast you run
    try:
        # Store the SAME household euro saving the Savings page shows for "today",
        # so the running total adds up in the same unit (previously this stored a
        # cheapest-vs-dearest price spread in EUR/MWh, which mixed units).
        default_apps = ["washing_machine", "dishwasher", "ev", "water_heater"]
        saving = round(recommend(fc, default_apps)["total_appliance_saving_eur"], 2)
        db.save_forecast(day, fc["rows"], saving)
    except Exception:
        pass
    return fc


@app.get("/api/decision")
def decision(day: str):
    try:
        fc = forecaster.forecast(day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    dec = decide(fc)
    # persist the forecast + decisions (matches the ERD: FORECAST + DECISION)
    db.save_forecast(day, dec["rows"], dec["estimated_saving"])
    return dec


@app.get("/api/history")
def history():
    return {"history": db.get_history()}


@app.get("/api/green")
def green(day: str):
    """Greenest-hours view: real carbon intensity + energy mix from the dataset."""
    try:
        return compute_green(forecaster.frame, day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _parse_custom(custom: str):
    """custom string 'Pool pump~1.2~3;Heater~2~2' -> list of dicts."""
    out = []
    for part in (custom or "").split(";"):
        part = part.strip()
        if not part:
            continue
        bits = part.split("~")
        if len(bits) >= 3:
            out.append({"label": bits[0], "kw": bits[1], "hours": bits[2]})
    return out


@app.get("/api/plan")
def plan(day: str, appliances: str = "washing_machine,dishwasher,ev,water_heater",
         custom: str = ""):
    """SmartVolt: forecast -> money-saving recommendations -> plain-language explanation."""
    try:
        fc = forecaster.forecast(day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    apps = [a.strip() for a in appliances.split(",") if a.strip()]
    p = recommend(fc, apps, _parse_custom(custom))
    ex = agent_explain(p, fc)
    return {"plan": p, "explanation": ex}


class ChatIn(BaseModel):
    question: str
    day: str
    appliances: list[str] = ["washing_machine", "ev"]


@app.post("/api/chat")
def chat(payload: ChatIn):
    """AI assistant: answer the user's question about their energy plan."""
    try:
        fc = forecaster.forecast(payload.day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    p = recommend(fc, payload.appliances)
    return agent_chat(payload.question, p, fc)


# ---- login / accounts ----
import secrets as _secrets
TOKENS = {}   # token -> username (in-memory session)


class AuthIn(BaseModel):
    username: str
    password: str


class ProfileIn(BaseModel):
    token: str
    appliances: list[str]


@app.post("/api/signup")
def signup(p: AuthIn):
    if len(p.username.strip()) < 3 or len(p.password) < 4:
        raise HTTPException(status_code=400, detail="Username needs 3+ letters and password 4+ characters.")
    if not db.create_user(p.username, p.password):
        raise HTTPException(status_code=400, detail="That username is already taken.")
    tok = _secrets.token_hex(16); TOKENS[tok] = p.username.strip().lower()
    return {"token": tok, "username": p.username.strip().lower(),
            "appliances": db.get_appliances(p.username), "prefs": db.get_prefs(p.username)}


@app.post("/api/login")
def login(p: AuthIn):
    if not db.check_user(p.username, p.password):
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    tok = _secrets.token_hex(16); TOKENS[tok] = p.username.strip().lower()
    return {"token": tok, "username": p.username.strip().lower(),
            "appliances": db.get_appliances(p.username), "prefs": db.get_prefs(p.username)}


class SocialIn(BaseModel):
    provider: str
    email: str = ""


@app.post("/api/social")
def social_login(p: SocialIn):
    """Simplified social sign-in for the demo. Real OAuth (Google/Apple) would
    replace this once the app is deployed to an https domain with provider creds.
    Creates or reuses a local account keyed by the email/provider."""
    prov = (p.provider or "Google").strip()
    email = (p.email or f"{prov.lower()}.user@example.com").strip().lower()
    uname = email.split("@")[0] + "." + prov.lower()
    if not db.check_user(uname, "social-" + email):
        db.create_user(uname, "social-" + email)
    tok = _secrets.token_hex(16); TOKENS[tok] = uname.strip().lower()
    return {"token": tok, "username": f"{prov} user",
            "appliances": db.get_appliances(uname), "prefs": db.get_prefs(uname)}


@app.get("/api/ai_status")
def ai_status():
    """Report whether the AI agents are live, and the reason if not."""
    from .agents import system as sysmod
    if not sysmod._has_key():
        return {"ai": False, "reason": "No LLM key found in api_key.txt."}
    try:
        import groq  # noqa
    except Exception:
        return {"ai": False, "reason": "The 'groq' package is not installed (pip install groq)."}
    try:
        sysmod.build_supervisor(sysmod._make_model())
        return {"ai": True, "reason": "AI agents are ready."}
    except Exception as e:
        return {"ai": False, "reason": f"Agent build failed: {e}"}


@app.post("/api/profile")
def save_profile(p: ProfileIn):
    user = TOKENS.get(p.token)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in again.")
    db.set_appliances(user, p.appliances)
    return {"ok": True}


class PrefsIn(BaseModel):
    token: str
    prefs: dict


@app.post("/api/prefs")
def save_prefs(p: PrefsIn):
    user = TOKENS.get(p.token)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in again.")
    db.set_prefs(user, p.prefs)
    return {"ok": True}


class PwChange(BaseModel):
    token: str
    old_password: str
    new_password: str


@app.post("/api/change_password")
def change_password(p: PwChange):
    user = TOKENS.get(p.token)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in again.")
    if len(p.new_password) < 4:
        raise HTTPException(status_code=400, detail="New password must be at least 4 characters.")
    if not db.change_password(user, p.old_password, p.new_password):
        raise HTTPException(status_code=400, detail="Your current password is wrong.")
    return {"ok": True}


class TokenIn(BaseModel):
    token: str


@app.post("/api/delete_account")
def delete_account(p: TokenIn):
    user = TOKENS.get(p.token)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in again.")
    db.delete_user(user)
    TOKENS.pop(p.token, None)
    return {"ok": True}


# ---- serve the frontend ----
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


app.mount("/", StaticFiles(directory=FRONTEND), name="static")
