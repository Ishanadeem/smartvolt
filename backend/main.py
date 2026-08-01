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

# Full model needs PyTorch; the cloud deployment ships without it and serves
# pre-computed forecasts instead. Pick whichever is available.
if os.environ.get("SMARTVOLT_LITE") == "1":
    from .model_service_lite import ForecastService
else:
    try:
        from .model_service import ForecastService
    except Exception:
        from .model_service_lite import ForecastService
from .decision_service import decide
from .database import Database
from . import data_pipeline
from .agents.recommender import recommend
from .agents.system import explain as agent_explain, chat as agent_chat
from pydantic import BaseModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(HERE, "frontend")

# If the user pasted a free Groq key into api_key.txt, load it so the AI agents turn on.
_keyfile = os.path.join(HERE, "api_key.txt")
if os.path.exists(_keyfile) and not os.environ.get("GROQ_API_KEY"):
    _k = open(_keyfile, encoding="utf-8").read().strip()
    if _k and _k.startswith("gsk_"):
        os.environ["GROQ_API_KEY"] = _k

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
    if forecaster.data_status().get("mode") == "cloud":
        raise HTTPException(status_code=400, detail="Uploading a new dataset needs the live "
                            "model, which runs in the desktop version. The cloud demo uses the "
                            "built-in Spanish dataset.")
    tmp = tempfile.mkdtemp()
    try:
        ep = os.path.join(tmp, "energy.csv")
        wp = os.path.join(tmp, "weather.csv")
        with open(ep, "wb") as f:
            shutil.copyfileobj(energy.file, f)
        with open(wp, "wb") as f:
            shutil.copyfileobj(weather.file, f)
        data_pipeline.build_merged(ep, wp)      # clean + merge -> data/clean_merged.csv
        forecaster.reload_data()                # reload the new data
        db.populate_history(forecaster.frame)   # refresh price_history
        return {"ok": True, **forecaster.data_status()}
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Could not process the files: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/api/metrics")
def metrics():
    return {"metrics": db.get_metrics()}


def _log_history(day, fc):
    """Save a forecast to the History log (one entry per day; re-runs update it)."""
    try:
        prices = [r["predicted"] for r in fc["rows"]]
        saving = round(max(prices) - min(prices), 1)   # cheapest-vs-dearest spread
        db.save_forecast(day, fc["rows"], saving)
    except Exception:
        pass


@app.get("/api/forecast")
def forecast(day: str):
    try:
        fc = forecaster.forecast(day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _log_history(day, fc)   # show every forecast you run in History
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
    _log_history(day, fc)   # a plan is a forecast too -> log it in History
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
TOKENS = {}   # in-memory cache: token -> username (backed by the DB session table)


def _user_of(token):
    """Resolve a token to a username, from the in-memory cache or the DB
    (so sessions survive a server restart)."""
    if not token:
        return None
    u = TOKENS.get(token)
    if u:
        return u
    u = db.get_session(token)
    if u:
        TOKENS[token] = u
    return u


def _new_session(username):
    tok = _secrets.token_hex(16)
    uname = username.strip().lower()
    TOKENS[tok] = uname
    db.add_session(tok, uname)
    return tok


class AuthIn(BaseModel):
    username: str
    password: str


class TokenIn(BaseModel):
    token: str


class ProfileIn(BaseModel):
    token: str
    appliances: list[str]


@app.post("/api/signup")
def signup(p: AuthIn):
    if len(p.username.strip()) < 3 or len(p.password) < 4:
        raise HTTPException(status_code=400, detail="Username needs 3+ letters and password 4+ characters.")
    if not db.create_user(p.username, p.password):
        raise HTTPException(status_code=400, detail="That username is already taken.")
    tok = _new_session(p.username)
    return {"token": tok, "username": p.username.strip().lower(),
            "appliances": db.get_appliances(p.username), "prefs": db.get_prefs(p.username)}


@app.post("/api/login")
def login(p: AuthIn):
    if not db.check_user(p.username, p.password):
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    tok = _new_session(p.username)
    return {"token": tok, "username": p.username.strip().lower(),
            "appliances": db.get_appliances(p.username), "prefs": db.get_prefs(p.username)}


@app.get("/api/me")
def me(token: str):
    """Validate a saved token. Returns the account if valid, else 401 so the
    frontend can clear a stale/ghost session and show the login screen."""
    user = _user_of(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    return {"username": user, "appliances": db.get_appliances(user),
            "prefs": db.get_prefs(user)}


@app.post("/api/logout")
def logout(p: TokenIn):
    TOKENS.pop(p.token, None)
    db.del_session(p.token)
    return {"ok": True}


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
    tok = _new_session(uname)
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
    user = _user_of(p.token)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in again.")
    db.set_appliances(user, p.appliances)
    return {"ok": True}


class PrefsIn(BaseModel):
    token: str
    prefs: dict


@app.post("/api/prefs")
def save_prefs(p: PrefsIn):
    user = _user_of(p.token)
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
    user = _user_of(p.token)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in again.")
    if len(p.new_password) < 4:
        raise HTTPException(status_code=400, detail="New password must be at least 4 characters.")
    if not db.change_password(user, p.old_password, p.new_password):
        raise HTTPException(status_code=400, detail="Your current password is wrong.")
    return {"ok": True}


@app.post("/api/delete_account")
def delete_account(p: TokenIn):
    user = _user_of(p.token)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in again.")
    db.delete_user(user)
    db.del_user_sessions(user)
    TOKENS.pop(p.token, None)
    return {"ok": True}


# ---- serve the frontend ----
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


app.mount("/", StaticFiles(directory=FRONTEND), name="static")
