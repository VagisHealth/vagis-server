"""
Vagis backend — agent relay + two-system data pipeline (research + clinical).

Two parallel systems, one server, told apart by code prefix:

                        RESEARCH                    CLINICAL (physician)
  Provider code         RES001                      PHY001
  Person code           SE0010001K3P (subject)      PT0010001K3P (patient)
  Data retention        persistent (study data)     ephemeral (auto-purged 48h)
  Stored in             research_uploads            clinical_holds
  Governed by           study protocol + consent    individual review, no keep

The person-code prefix routes the data: SE -> persistent research store;
PT -> ephemeral clinical hold that self-deletes 48h after upload. The two live
in separate tables so clinical data physically cannot land in the persistent
store.

Endpoints (foundation):
  GET  /health                 -- status
  POST /chat                   -- agent relay (unchanged)
  POST /ingest                 -- app uploads a CSV; routed by code prefix
  POST /portal/validate        -- app checks an SE or PT code is real
  GET  /admin                  -- create providers (research or clinical)
  GET  /portal                 -- provider login (RES -> research, PHY -> clinical)
"""

from __future__ import annotations

import asyncio
import csv
import io
import os
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import anthropic
import psycopg2
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VAGIS_APP_TOKEN = os.environ.get("VAGIS_APP_TOKEN", "")
VAGIS_ADMIN_TOKEN = os.environ.get("VAGIS_ADMIN_TOKEN", "")
MODEL = os.environ.get("VAGIS_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.environ.get("VAGIS_MAX_TOKENS", "1024"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Per-phone question caps for the in-app agent. Each phone sends an anonymous
# ID (X-Vagis-Device); questions are counted per ID per UTC day. Set a cap to
# 0 to switch it off. Exempt IDs (comma-separated) are never capped.
AGENT_DAILY_CAP = int(os.environ.get("VAGIS_AGENT_DAILY_CAP", "20"))
AGENT_MONTHLY_CAP = int(os.environ.get("VAGIS_AGENT_MONTHLY_CAP", "200"))
AGENT_EXEMPT_DEVICES = {
    d.strip() for d in os.environ.get("VAGIS_AGENT_EXEMPT_DEVICES", "").split(",") if d.strip()
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Clinical (PT) uploads self-delete this many hours after they arrive.
CLINICAL_HOLD_HOURS = int(os.environ.get("VAGIS_CLINICAL_HOLD_HOURS", "48"))

# Unambiguous alphabet for the random tail: no O, 0, I, 1, L.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

# Prefixes that define the two systems.
PROVIDER_PREFIX = {"research": "RES", "clinical": "PHY"}   # 3 chars each
PERSON_PREFIX   = {"research": "SE",  "clinical": "PT"}    # 2 chars each
KIND_BY_PROVIDER_PREFIX = {v: k for k, v in PROVIDER_PREFIX.items()}
KIND_BY_PERSON_PREFIX   = {v: k for k, v in PERSON_PREFIX.items()}

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
# ---- Log noise ------------------------------------------------------------
# Render health-checks /health every few seconds. Those access-log lines bury
# everything worth reading, so they are filtered out by default. Set
# VAGIS_LOG_HEALTH=1 to see them again when debugging the health check itself.
import logging as _logging


class _QuietPolling(_logging.Filter):
    NOISY = ("/health", "/favicon.ico")

    def filter(self, record: _logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(p in msg for p in self.NOISY)


if os.environ.get("VAGIS_LOG_HEALTH", "").strip() not in ("1", "true", "True"):
    _logging.getLogger("uvicorn.access").addFilter(_QuietPolling())

app = FastAPI(title="Vagis Server")


# --------------------------------------------------------------------------
# Enrollment code scheme  (pure functions -- unit tested)
# --------------------------------------------------------------------------
# Provider: <PREFIX 3> + <seq 3>              e.g. RES001 / PHY001
# Person  : <PREFIX 2> + <provider 3> + <person 4> + <tail 3> = 12
#           e.g. SE0010001K3P (research) / PT0010001K3P (clinical)
def make_provider_code(kind: str, seq: int) -> str:
    if kind not in PROVIDER_PREFIX:
        raise ValueError(f"unknown kind {kind!r}")
    if not (1 <= seq <= 999):
        raise ValueError("provider sequence out of range (1-999)")
    return f"{PROVIDER_PREFIX[kind]}{seq:03d}"


def make_person_code(kind: str, provider_seq: int, person_seq: int) -> str:
    if kind not in PERSON_PREFIX:
        raise ValueError(f"unknown kind {kind!r}")
    if not (1 <= provider_seq <= 999):
        raise ValueError("provider sequence out of range (1-999)")
    if not (1 <= person_seq <= 9999):
        raise ValueError("person sequence out of range (1-9999)")
    tail = "".join(secrets.choice(CODE_ALPHABET) for _ in range(3))
    return f"{PERSON_PREFIX[kind]}{provider_seq:03d}{person_seq:04d}{tail}"


def parse_provider_code(code: str) -> Optional[dict[str, Any]]:
    code = (code or "").strip().upper()
    if len(code) != 6:
        return None
    prefix, digits = code[:3], code[3:6]
    if prefix not in KIND_BY_PROVIDER_PREFIX or not digits.isdigit():
        return None
    return {"provider_code": code, "kind": KIND_BY_PROVIDER_PREFIX[prefix], "seq": int(digits)}


def parse_person_code(code: str) -> Optional[dict[str, Any]]:
    code = (code or "").strip().upper()
    if len(code) != 12:
        return None
    prefix = code[:2]
    if prefix not in KIND_BY_PERSON_PREFIX:
        return None
    prov_digits, person_digits, tail = code[2:5], code[5:9], code[9:12]
    if not prov_digits.isdigit() or not person_digits.isdigit():
        return None
    if any(c not in CODE_ALPHABET for c in tail):
        return None
    kind = KIND_BY_PERSON_PREFIX[prefix]
    return {
        "person_code": code,
        "kind": kind,
        "provider_code": f"{PROVIDER_PREFIX[kind]}{prov_digits}",
        "provider_seq": int(prov_digits),
        "person_seq": int(person_digits),
        "tail": tail,
    }


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
# providers: both research (RES) and clinical (PHY), told apart by `kind`.
CREATE_PROVIDERS_SQL = """
CREATE TABLE IF NOT EXISTS providers (
    provider_code TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    name          TEXT,
    email         TEXT,
    secret        TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kind, seq)
);
"""

# persons: subjects (SE, research) and patients (PT, clinical).
CREATE_PERSONS_SQL = """
CREATE TABLE IF NOT EXISTS persons (
    person_code   TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    provider_code TEXT NOT NULL REFERENCES providers(provider_code),
    person_seq    INTEGER NOT NULL,
    label         TEXT,
    email         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (provider_code, person_seq)
);
"""

# research_uploads: PERSISTENT. One row per (subject, mode); re-upload replaces.
CREATE_RESEARCH_UPLOADS_SQL = """
CREATE TABLE IF NOT EXISTS research_uploads (
    id           SERIAL PRIMARY KEY,
    person_code  TEXT NOT NULL,
    mode         TEXT NOT NULL,
    filename     TEXT,
    csv_text     TEXT NOT NULL,
    row_count    INTEGER,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (person_code, mode)
);
"""

# clinical_holds: EPHEMERAL. Same shape plus expires_at; purged after it passes.
CREATE_CLINICAL_HOLDS_SQL = """
CREATE TABLE IF NOT EXISTS clinical_holds (
    id           SERIAL PRIMARY KEY,
    person_code  TEXT NOT NULL,
    mode         TEXT NOT NULL,
    filename     TEXT,
    csv_text     TEXT NOT NULL,
    row_count    INTEGER,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ NOT NULL,
    UNIQUE (person_code, mode)
);
"""

UPSERT_RESEARCH_SQL = """
INSERT INTO research_uploads (person_code, mode, filename, csv_text, row_count, uploaded_at)
VALUES (%s, %s, %s, %s, %s, now())
ON CONFLICT (person_code, mode)
DO UPDATE SET filename=EXCLUDED.filename, csv_text=EXCLUDED.csv_text,
              row_count=EXCLUDED.row_count, uploaded_at=now()
RETURNING uploaded_at;
"""

UPSERT_CLINICAL_SQL = """
INSERT INTO clinical_holds (person_code, mode, filename, csv_text, row_count, uploaded_at, expires_at)
VALUES (%s, %s, %s, %s, %s, now(), %s)
ON CONFLICT (person_code, mode)
DO UPDATE SET filename=EXCLUDED.filename, csv_text=EXCLUDED.csv_text,
              row_count=EXCLUDED.row_count, uploaded_at=now(), expires_at=EXCLUDED.expires_at
RETURNING uploaded_at, expires_at;
"""


# agent_usage: questions asked per phone per UTC day, for the per-phone caps.
CREATE_AGENT_USAGE_SQL = """
CREATE TABLE IF NOT EXISTS agent_usage (
    device_id  TEXT NOT NULL,
    day        DATE NOT NULL,
    questions  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (device_id, day)
);
"""


def db_connect():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="Database not configured.")
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Database connection failed: {type(e).__name__}")


def ensure_tables(cur) -> None:
    cur.execute(CREATE_PROVIDERS_SQL)   # referenced by persons, first
    cur.execute(CREATE_PERSONS_SQL)
    cur.execute(CREATE_RESEARCH_UPLOADS_SQL)
    cur.execute(CREATE_CLINICAL_HOLDS_SQL)
    cur.execute(CREATE_AGENT_USAGE_SQL)
    # Cache of the Anthropic Files API id for each stored CSV, so the analysis
    # agent reads data off disk in its sandbox instead of having it inlined in
    # the prompt. Re-uploaded only when the underlying CSV is newer.
    cur.execute("ALTER TABLE research_uploads "
                "ADD COLUMN IF NOT EXISTS anthropic_file_id TEXT;")
    cur.execute("ALTER TABLE research_uploads "
                "ADD COLUMN IF NOT EXISTS file_uploaded_at TIMESTAMPTZ;")


def purge_expired(cur) -> int:
    """Delete clinical holds whose window has passed. Returns rows removed."""
    cur.execute("DELETE FROM clinical_holds WHERE expires_at < now();")
    return cur.rowcount


@app.on_event("startup")
def init_db() -> None:
    if not DATABASE_URL:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            purge_expired(cur)
        conn.close()
    except Exception as e:
        print(f"[startup] db init failed: {type(e).__name__}: {e}")


# Background sweeper: purge expired clinical holds even with no traffic.
def _purge_loop() -> None:
    while True:
        time.sleep(900)  # every 15 minutes
        if not DATABASE_URL:
            continue
        try:
            conn = psycopg2.connect(DATABASE_URL)
            with conn, conn.cursor() as cur:
                n = purge_expired(cur)
            conn.close()
            if n:
                print(f"[purge] removed {n} expired clinical hold(s)")
        except Exception as e:
            print(f"[purge] sweep failed: {type(e).__name__}: {e}")


@app.on_event("startup")
def start_purge_thread() -> None:
    if DATABASE_URL:
        threading.Thread(target=_purge_loop, daemon=True).start()


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def check_app_auth(authorization: str | None) -> None:
    if not VAGIS_APP_TOKEN:
        raise HTTPException(status_code=500, detail="Server token not configured.")
    if authorization != f"Bearer {VAGIS_APP_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized.")


def check_admin_auth(authorization: str | None) -> None:
    if not VAGIS_ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="Admin token not configured.")
    if authorization != f"Bearer {VAGIS_ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Admin unauthorized.")


def authenticate_provider(cur, provider_code: str, secret: str) -> Optional[dict[str, Any]]:
    """Return provider dict if code+secret match, else None."""
    parsed = parse_provider_code(provider_code)
    if not parsed:
        return None
    cur.execute("SELECT provider_code, kind, seq, name, secret FROM providers WHERE provider_code = %s;",
                (parsed["provider_code"],))
    row = cur.fetchone()
    if not row or not secret or not secrets.compare_digest(row[4], secret):
        return None
    return {"provider_code": row[0], "kind": row[1], "seq": row[2], "name": row[3]}


# --------------------------------------------------------------------------
# Chat models + system prompt  (unchanged from prior version)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Request / response shapes
# --------------------------------------------------------------------------
class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    mode: str = ""
    date: str = ""
    metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    history_summary: str = ""
    conversation: list[Turn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str


class IssueResearcherRequest(BaseModel):
    name: str = ""
    email: str = ""


class IssueSubjectRequest(BaseModel):
    rp_code: str
    secret: str
    label: str = ""          # optional private note, e.g. "pilot subject 3"


class ValidateRequest(BaseModel):
    se_code: str


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------
def render_metrics(metrics: dict[str, dict[str, Any]]) -> str:
    if not metrics:
        return "No structured metrics were provided for this session."
    lines: list[str] = []
    for section, values in metrics.items():
        lines.append(f"## {section}")
        if isinstance(values, dict):
            for label, value in values.items():
                lines.append(f"- {label}: {value}")
        else:
            lines.append(f"- {values}")
        lines.append("")
    return "\n".join(lines).strip()


# Plain-language description of each app mode's metrics, so the agent can
# explain a mode even before the user has recorded in it. Only what the app
# actually shows -- no formulas or cut-offs. Modes not listed here rely on the
# session data alone, as before.
MODE_GUIDES: dict[str, str] = {
    "Load": """Load is a daytime monitor built from two things the ring measures: heart \
rate and hand movement (accelerometer). Movement is the foundation -- it decides \
which moments were still and which were active, and heart rate is read against it. \
Load does not use pulse amplitude.
Metrics the app shows:
- Timeline: heart rate and movement across the whole recording; lowest, average and \
maximum heart rate.
- Motion: bars show how the recording divided between four movement bands -- Still, \
Light, Moderate and High. Tiles: longest still stretch (longest unbroken rest), \
moving bouts per hour (stretches of moderate or high movement lasting a minute or \
more), longest moving bout, and average motion (overall movement intensity, in g).
- Response: bars show the typical (median) heart rate in each movement band. Heart \
rate follows movement with a short delay, so each stretch is assigned to a band by \
the movement just before it. Tiles: Still to High difference (how much heart rate \
rises from still to high movement), heart rate per motion (how much heart rate rises \
for a given amount of movement -- the heart-rate cost of moving), response delay \
(how many seconds heart rate takes to follow movement), and still heart rate range \
(how settled heart rate is while still; a wide range means it is unsettled at rest).
- Heart rate ceiling: a level set from the user's age (only if they entered one), \
and the minutes spent above it.
Load is a monitor, not a test: it does not score a recording or say what is good or \
bad. Changes over days and weeks are for the user to interpret, and comparing a \
recording with their own recent ones is more meaningful than any single value. \
It is aimed at people tracking day-to-day energy limits (e.g. ME/CFS, long COVID).""",

    "Sleep": """Sleep records a whole night from the ring: heart beats, pulse wave and \
movement. Tabs: Stages, Heart Rate, Breath, Cycling, Deep, Export.
- Stages: the night divided into Awake, REM, Light and Deep sleep, estimated from \
heart-rhythm patterns and movement. Ring-based staging is an estimate, not a sleep-lab \
study.
- Heart Rate: heart rate across the night.
- ADI (Autonomic Disturbance Index): how much slow, disturbance-type fluctuation there \
is in the heart rhythm across the night -- a general marker of how settled the \
nervous system was.
- Breath: breathing-related measures from the ring signals.
- Cycling: slow rises and falls in the pulse wave (the finger's blood-flow signal). \
Three kinds are shown: Vasocycling (the normal resting rhythm of the blood vessels, \
no bigger than the user's own Deep-sleep cycling), PWA Cycling (bigger cycling while \
the breathing movement in the pulse is reduced), and PWA Cycling + HR (the same with \
a heart-rate surge well beyond the user's own Deep-sleep swing). These are measured \
patterns, not a diagnosis; oxygen dips are not required for them to appear.
- Deep: measures taken only from Deep sleep, the calmest and most restorative stage, \
used as the user's own calm reference.
Sleep is a monitor, not a diagnostic test. Nightly values vary; trends across nights \
are more meaningful than a single night.""",

    "Stand": """Stand is an active stand test: the user lies or sits still, then \
stands, while the ring records. It shows how the heart and blood vessels respond to \
standing up.
- Heart Rate: resting heart rate, peak standing heart rate, the rise from rest to \
standing, how fast it rose and when it peaked, the rises at 1 and 2 minutes, and any \
recovery dip after the peak. All heart-rate values come from the ring's own heart-rate \
reading.
- Pulse Strength: how the size of the pulse at the finger changes on standing.
- Rise Timing: how quickly each pulse rises, relative to the heartbeat, before and \
after standing.
- Blood Pooling: signs of blood shifting into the lower body on standing.
- The app notes whether the heart-rate rise meets a widely used orthostatic \
criterion. That is a measured result, not a diagnosis of POTS or any condition -- only \
a clinician can diagnose.
Repeating the test under similar conditions and comparing results over time is more \
meaningful than one test.""",

    "Exertion": """Exertion tracks heart rate through daily activity. The ring can \
store heart rate while the phone is away and hand it over when it reconnects, so a \
recording can cover a whole day. Using the age and weight the user entered, it \
estimates how hard the body was working.
Metrics: resting heart rate, typical (median), high (95th percentile) and peak heart \
rate, heart rate as a share of estimated maximum, estimated METs (effort relative to \
rest), and energy rate (kcal per hour).
METs, maximum heart rate and energy values are population-based estimates, not \
measurements. Exertion is a monitor, not a medical test: it does not detect or predict \
post-exertional malaise. It is aimed at people pacing their energy (e.g. ME/CFS, long \
COVID, hEDS); comparing days with the user's own history is the useful part.""",

    "Breathwork": """Breathwork guides the user through slow, paced breathing (or free \
breathing) while the ring records, and shows how strongly the heart and blood vessels \
follow the breath.
Metrics: breathing pace (breaths per minute), RSA amplitude (how much the heart rate \
rises and falls with each breath -- larger usually means stronger breathing-linked \
heart-rhythm activity), coherence (how closely the heart rhythm and the pulse wave \
move together with the breath), ln LF (slow heart-rhythm power, which paced breathing \
around six breaths per minute tends to boost), and pulse-wave variability.
Breathwork is a practice and a monitor, not a treatment or test. Values change with \
pace, posture and practice; comparing sessions at the same pace is most meaningful.""",
}


def build_system_prompt(req: ChatRequest) -> str:
    metrics_block = render_metrics(req.metrics)
    guide = MODE_GUIDES.get(req.mode.strip(), "")
    guide_block = (
        f"""
--- ABOUT THIS MODE ---
{guide}
If no session data is shown below, the user has not recorded in this mode yet: \
explain what the mode and its metrics are, using only the description above, and \
do not invent any values.
"""
        if guide else ""
    )
    history_block = (
        req.history_summary.strip()
        if req.history_summary.strip()
        else "No session history yet."
    )
    return f"""You are the personal health assistant inside the Vagis app. You help \
the user understand their own autonomic nervous system data, recorded from a smart \
ring, in plain and accessible language.

What you are:
- An educational tool that explains what the user's own numbers mean and what \
generally influences them.
- Grounded in the user's actual session data, shown below. Refer to their specific \
values when you answer -- be concrete, not generic.

What you are not:
- You are not a diagnostic or medical device. You do not diagnose conditions, \
interpret data as evidence of any specific disease, or recommend treatments, \
medication changes, or procedures.
- If the user describes symptoms, asks whether something is wrong with them, or asks \
a clinical question, explain the relevant physiology in general terms and suggest \
they discuss it with their physician. Do not speculate about diagnoses.

GROUND EVERYTHING IN THE DATA SHOWN BELOW -- THIS IS THE MOST IMPORTANT RULE:
- The data below -- the latest recording under THIS SESSION and the SESSION HISTORY \
tables -- is the only data the app produces. Discuss ONLY these metrics and the \
general physiology behind them.
- If the user asks about a metric, score, or feature that is NOT in the data below, \
do not invent one. Say plainly that it isn't part of what the app shows for this \
session, and offer to discuss the metrics that ARE present instead. Never make up a \
metric name, a number, a formula, a threshold, or a normal range that is not given \
to you here.
- Never state a specific value for any metric unless that exact value appears in the \
data below. If you don't have a number, say you don't have it rather than estimating.
- It is always better to say "I don't have that" than to guess. Confident-sounding \
invention is the worst outcome and must be avoided.

EXPLAINING METRICS:
- You CAN and SHOULD explain, in plain language, what each metric shown below \
measures and what generally influences it -- this is one of your main jobs.
- Explain the concept and what it reflects about the body. Do NOT reveal or speculate \
about the internal calculation, formula, frequency bands, thresholds, or algorithm \
behind a Vagis metric. If asked how a metric is computed, describe what it represents \
and why it matters, not the math.
- When you explain a metric, connect it to the user's actual value for it where one \
is shown.

How to respond:
- Keep answers concise -- a few sentences unless the user asks for more detail.
- Use plain language. Define a term the first time you use it.
- Be warm and direct. The user is the expert on their own body and how they feel.
- Only discuss the data and physiology. If asked something unrelated, gently steer \
back to their health data.

{guide_block}
--- THIS SESSION ---
Mode: {req.mode or "(not specified)"}
Date: {req.date or "(not specified)"}

{metrics_block}

--- SESSION HISTORY ---
These tables are the user's own past recordings in each mode (this mode first, \
newest recording first; "—" means that metric was not available for that \
recording). Use them to compare the latest recording with earlier ones, to describe \
trends, and to connect modes when it helps. The user is in {req.mode or "this"} \
mode, so focus there unless they ask about another mode or a link between modes.

{history_block}
"""




# --------------------------------------------------------------------------
# Health + chat
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL,
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "app_token_set": bool(VAGIS_APP_TOKEN),
        "admin_token_set": bool(VAGIS_ADMIN_TOKEN),
        "database_url_set": bool(DATABASE_URL),
        "clinical_hold_hours": CLINICAL_HOLD_HOURS,
        "agent_daily_cap": AGENT_DAILY_CAP,
        "agent_monthly_cap": AGENT_MONTHLY_CAP,
        "agent_exempt_devices": len(AGENT_EXEMPT_DEVICES),
    }


# --------------------------------------------------------------------------
# Per-phone question caps
# --------------------------------------------------------------------------
# FAIL OPEN: if the database cannot be reached, the question is answered and
# simply not counted. A database hiccup must never take the agent down.
def _agent_usage(device_id: str) -> Optional[tuple[int, int]]:
    """(questions today, questions this month) for one phone, UTC."""
    if not DATABASE_URL:
        return None
    today = datetime.now(timezone.utc).date()
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn, conn.cursor() as cur:
            cur.execute(CREATE_AGENT_USAGE_SQL)
            cur.execute(
                "SELECT COALESCE(SUM(questions) FILTER (WHERE day = %s), 0), "
                "COALESCE(SUM(questions), 0) FROM agent_usage "
                "WHERE device_id = %s AND day >= %s;",
                (today, device_id, today.replace(day=1)),
            )
            row = cur.fetchone()
        conn.close()
        return int(row[0]), int(row[1])
    except Exception as e:
        print(f"[agent-cap] usage read failed: {type(e).__name__}: {e}")
        return None


def _agent_record(device_id: str) -> None:
    if not DATABASE_URL:
        return
    today = datetime.now(timezone.utc).date()
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_usage (device_id, day, questions) VALUES (%s, %s, 1) "
                "ON CONFLICT (device_id, day) DO UPDATE "
                "SET questions = agent_usage.questions + 1;",
                (device_id, today),
            )
        conn.close()
    except Exception as e:
        print(f"[agent-cap] usage write failed: {type(e).__name__}: {e}")


def _check_agent_cap(device_id: str) -> None:
    """Raise 429 with a message the app shows as-is when a cap is reached."""
    counts = _agent_usage(device_id)
    if counts is None:
        return
    today_n, month_n = counts
    if AGENT_DAILY_CAP > 0 and today_n >= AGENT_DAILY_CAP:
        now = datetime.now(timezone.utc)
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        hours = max(1, round((midnight - now).total_seconds() / 3600))
        raise HTTPException(
            status_code=429,
            detail=(f"You've reached today's limit of {AGENT_DAILY_CAP} questions. "
                    f"It resets in about {hours} hour{'s' if hours != 1 else ''}."),
        )
    if AGENT_MONTHLY_CAP > 0 and month_n >= AGENT_MONTHLY_CAP:
        raise HTTPException(
            status_code=429,
            detail=(f"You've reached this month's limit of {AGENT_MONTHLY_CAP} questions. "
                    "It resets on the 1st of next month."),
        )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, authorization: str | None = Header(default=None),
         x_vagis_device: str | None = Header(default=None)) -> ChatResponse:
    check_app_auth(authorization)
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic key not configured.")
    if not req.conversation:
        raise HTTPException(status_code=400, detail="No conversation provided.")

    # Older app builds send no device ID; they are not capped (they will be
    # replaced by the next TestFlight build).
    device_id = (x_vagis_device or "").strip()[:64]
    capped = bool(device_id) and device_id not in AGENT_EXEMPT_DEVICES
    if capped:
        _check_agent_cap(device_id)

    system_prompt = build_system_prompt(req)
    messages = [{"role": t.role, "content": t.content} for t in req.conversation]

    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages,
        )
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic error: {e.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {type(e).__name__}")

    reply = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()
    if not reply:
        raise HTTPException(status_code=502, detail="Empty reply from model.")

    # Counted only once answered, so a failed call never uses up a question.
    if device_id:
        _agent_record(device_id)
    return ChatResponse(reply=reply)




# --------------------------------------------------------------------------
# Ingestion  (routed by person-code prefix: SE -> persistent, PT -> ephemeral)
# --------------------------------------------------------------------------
def count_csv_rows(text: str) -> int:
    reader = csv.reader(io.StringIO(text))
    n = sum(1 for _ in reader)
    return max(0, n - 1)


def person_exists(cur, person_code: str, kind: str) -> bool:
    cur.execute("SELECT 1 FROM persons WHERE person_code = %s AND kind = %s;", (person_code, kind))
    return cur.fetchone() is not None


@app.post("/ingest")
async def ingest(
    enrollment_code: str = Form(...),
    mode: str = Form(...),
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Store one cumulative CSV. The code prefix routes it:
    SE -> research_uploads (persistent). PT -> clinical_holds (ephemeral, 48h)."""
    check_app_auth(authorization)

    parsed = parse_person_code(enrollment_code)
    if not parsed:
        raise HTTPException(status_code=400,
            detail="enrollment_code must be a valid SE or PT code.")
    code = parsed["person_code"]
    kind = parsed["kind"]

    mode_clean = (mode or "").strip().lower()
    if not mode_clean:
        raise HTTPException(status_code=400, detail="mode is required.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
    try:
        csv_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text CSV.")

    row_count = count_csv_rows(csv_text)

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            purge_expired(cur)
            if not person_exists(cur, code, kind):
                raise HTTPException(status_code=404,
                    detail="Unknown enrollment code. It must be issued before uploading.")
            if kind == "research":
                cur.execute(UPSERT_RESEARCH_SQL,
                            (code, mode_clean, file.filename, csv_text, row_count))
                uploaded_at = cur.fetchone()[0]
                return {
                    "status": "ok", "system": "research", "enrollment_code": code,
                    "mode": mode_clean, "row_count": row_count,
                    "uploaded_at": uploaded_at.isoformat(), "retention": "persistent",
                }
            else:  # clinical -> ephemeral hold
                expires = datetime.now(timezone.utc) + timedelta(hours=CLINICAL_HOLD_HOURS)
                cur.execute(UPSERT_CLINICAL_SQL,
                            (code, mode_clean, file.filename, csv_text, row_count, expires))
                uploaded_at, expires_at = cur.fetchone()
                return {
                    "status": "ok", "system": "clinical", "enrollment_code": code,
                    "mode": mode_clean, "row_count": row_count,
                    "uploaded_at": uploaded_at.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "retention": f"ephemeral ({CLINICAL_HOLD_HOURS}h)",
                }
    finally:
        conn.close()


class ValidateRequest(BaseModel):
    enrollment_code: str


@app.post("/portal/validate")
def validate_person(req: ValidateRequest,
                    authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """App checks an SE or PT code is well-formed AND issued. Returns its provider."""
    check_app_auth(authorization)
    parsed = parse_person_code(req.enrollment_code)
    if not parsed:
        return {"valid": False, "reason": "malformed"}

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            cur.execute("SELECT provider_code, kind FROM persons WHERE person_code = %s;",
                        (parsed["person_code"],))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return {"valid": False, "reason": "not_issued"}
    return {"valid": True, "enrollment_code": parsed["person_code"],
            "provider_code": row[0], "system": row[1]}


# --------------------------------------------------------------------------
# Admin JSON endpoints (create/list providers of either kind)
# --------------------------------------------------------------------------
class IssueProviderRequest(BaseModel):
    kind: str            # "research" or "clinical"
    name: str = ""
    email: str = ""


def _issue_provider(cur, kind: str, name: str, email: str) -> dict[str, Any]:
    if kind not in PROVIDER_PREFIX:
        raise HTTPException(status_code=400, detail="kind must be 'research' or 'clinical'.")
    secret = secrets.token_urlsafe(24)
    cur.execute("SELECT COALESCE(MAX(seq), 0) FROM providers WHERE kind = %s;", (kind,))
    next_seq = cur.fetchone()[0] + 1
    if next_seq > 999:
        raise HTTPException(status_code=409, detail="Provider capacity reached (999).")
    code = make_provider_code(kind, next_seq)
    cur.execute(
        "INSERT INTO providers (provider_code, kind, seq, name, email, secret) "
        "VALUES (%s,%s,%s,%s,%s,%s);",
        (code, kind, next_seq, name or None, email or None, secret))
    return {"provider_code": code, "kind": kind, "secret": secret}


@app.post("/admin/providers")
def issue_provider(req: IssueProviderRequest,
                   authorization: str | None = Header(default=None)) -> dict[str, Any]:
    check_admin_auth(authorization)
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            out = _issue_provider(cur, req.kind, req.name, req.email)
    finally:
        conn.close()
    return {"status": "ok", **out}


def _issue_person(cur, provider: dict[str, Any], label: str, email: str) -> dict[str, Any]:
    kind = provider["kind"]
    cur.execute("SELECT COALESCE(MAX(person_seq), 0) FROM persons WHERE provider_code = %s;",
                (provider["provider_code"],))
    next_person = cur.fetchone()[0] + 1
    if next_person > 9999:
        raise HTTPException(status_code=409, detail="Person capacity reached (9999).")
    code = make_person_code(kind, provider["seq"], next_person)
    cur.execute("INSERT INTO persons (person_code, kind, provider_code, person_seq, label, email) "
                "VALUES (%s,%s,%s,%s,%s,%s);",
                (code, kind, provider["provider_code"], next_person, label or None, email or None))
    return {"person_code": code, "person_seq": next_person}

# --------------------------------------------------------------------------
# Web pages  (NO JavaScript -- plain HTML forms)
# --------------------------------------------------------------------------
from html import escape as _esc
from urllib.parse import quote as _q


def _style() -> str:
    return """
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 980px; margin: 0 auto; padding: 24px; color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  h2 { font-size: 16px; font-weight: 600; margin: 0 0 14px; }
  .sub { color: #666; font-size: 14px; margin: 0 0 22px; }
  .card { background: #fff; border: 1px solid #e4e4e4; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  label { display: block; font-size: 13px; color: #444; margin: 12px 0 4px; }
  input, select { width: 100%; padding: 10px 12px; font-size: 15px; border: 1px solid #d0d0d0; border-radius: 8px; background: #fff; }
  button { margin-top: 10px; padding: 9px 16px; font-size: 14px; font-weight: 500; color: #fff;
           background: #0f6e56; border: none; border-radius: 8px; cursor: pointer; }
  button.secondary { background: #444; }
  button.small { padding: 6px 12px; font-size: 13px; margin: 0; }
  form.inline { display: inline; margin: 0; }
  .result { margin: 0 0 20px; padding: 16px; border-radius: 8px; background: #e1f5ee; border: 1px solid #9fe1cb; }
  .result .row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 15px; }
  .result .k { color: #085041; font-weight: 500; }
  .result .v { font-family: ui-monospace, Menlo, monospace; font-size: 16px; }
  .warn { color: #854f0b; font-size: 13px; margin-top: 8px; }
  .err { margin: 0 0 20px; padding: 14px 16px; border-radius: 8px; background: #fcebeb; border: 1px solid #f7c1c1; color: #a32d2d; }
  table { width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 14px; }
  th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }
  th { color: #666; font-weight: 600; font-size: 12px; text-transform: uppercase; background: #f4f4f4; position: sticky; top: 0; }
  td.mono, .mono { font-family: ui-monospace, Menlo, monospace; }
  .muted { color: #999; font-size: 13px; }
  .tablewrap { overflow-x: auto; border: 1px solid #eee; border-radius: 8px; }
  .backbtn { display: inline-block; margin-top: 8px; color: #0f6e56; font-size: 14px; background: none; padding: 0; border: none; cursor: pointer; }
  .pill { display: inline-block; font-size: 11px; color: #0f6e56; background: #e1f5ee; border-radius: 5px; padding: 2px 8px; margin-left: 6px; }
  .flag { display: inline-block; font-size: 11px; color: #7a3b00; background: #ffe6c7; border-radius: 5px; padding: 2px 8px; margin-left: 6px; font-weight: 600; }
  .subrow { display: flex; align-items: center; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #f0f0f0; }
  .subrow:last-child { border-bottom: none; }
  .badge { display:inline-block; font-size:11px; font-weight:600; padding:2px 9px; border-radius:6px; }
  .badge.res { background:#e1f5ee; color:#085041; }
  .badge.phy { background:#e6eefc; color:#1c458f; }
</style>
"""


def _hidden(provider_code: str, key: str) -> str:
    return (f'<input type="hidden" name="provider_code" value="{_esc(provider_code)}">'
            f'<input type="hidden" name="key" value="{_esc(key)}">')


def _mailto(email: str, subject: str, body: str, text: str) -> str:
    if not email:
        return ""
    href = f"mailto:{_q(email)}?subject={_q(subject)}&body={_q(body)}"
    return (f'<a href="{href}" style="display:inline-block;margin-top:10px;padding:9px 16px;'
            f'background:#0f6e56;color:#fff;border-radius:8px;font-size:14px;font-weight:500;'
            f'text-decoration:none;">{_esc(text)}</a>')


def _mode_label(m: str) -> str:
    return {"sleep": "Sleep", "rest": "Rest", "stand": "Stand", "breathwork": "Breathwork",
            "circadian": "Circadian", "circadian_rhythm": "Circadian Rhythm",
            "circadian_episodes": "Circadian Episodes",
            "sleep_pwr": "Pulse Wave Rhythms",
            "sleep_pwr_episodes": "Pulse Wave Rhythms Episodes",
            "exertion": "Exertion", "load": "Load",
            "quick_check": "Quick Check"}.get(m, m.capitalize())


# ---- Admin page ----------------------------------------------------------
def _admin_page(token: str = "", banner: str = "") -> str:
    tok = _esc(token)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vagis Admin</title>{_style()}</head><body>
  <h1>Vagis Admin</h1>
  <p class="sub">Create provider accounts for the research and clinical systems.</p>
  {banner}
  <div class="card">
    <h2>Create a provider</h2>
    <form method="post" action="/admin/ui/create">
      <label>Admin token</label>
      <input name="token" type="password" placeholder="Your VAGIS_ADMIN_TOKEN" value="{tok}" autocomplete="off">
      <label>System</label>
      <select name="kind">
        <option value="research">Research  (RES &mdash; persistent study data)</option>
        <option value="clinical">Clinical  (PHY &mdash; ephemeral, 48h)</option>
      </select>
      <label>Name (optional)</label>
      <input name="name" type="text" placeholder="Dr. Jane Smith">
      <label>Provider email (required)</label>
      <input name="email" type="text" placeholder="jane@example.com">
      <button type="submit">Create provider</button>
    </form>
  </div>
  <div class="card">
    <h2>AI Connect key</h2>
    <p class="sub">Make a private connector address so a person's Claude or ChatGPT can read their Session History.</p>
    <form method="post" action="/admin/ui/aiconnect">
      <label>Admin token</label>
      <input name="token" type="password" placeholder="Your VAGIS_ADMIN_TOKEN" value="{tok}" autocomplete="off">
      <label>Person code (SE)</label>
      <input name="person_code" type="text" placeholder="SE0010001K3P">
      <button type="submit">Make connector address</button>
    </form>
  </div>
  <div class="card">
    <h2>Providers</h2>
    <form method="post" action="/admin/ui/list">
      <label>Admin token</label>
      <input name="token" type="password" placeholder="Your VAGIS_ADMIN_TOKEN" value="{tok}" autocomplete="off">
      <button type="submit" class="secondary">Show list</button>
    </form>
  </div>
</body></html>"""


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> HTMLResponse:
    return HTMLResponse(_admin_page())


@app.post("/admin/ui/create", response_class=HTMLResponse)
def admin_ui_create(token: str = Form(""), kind: str = Form("research"),
                    name: str = Form(""), email: str = Form("")) -> HTMLResponse:
    if token.strip() != (VAGIS_ADMIN_TOKEN or "").strip() or not VAGIS_ADMIN_TOKEN:
        return HTMLResponse(_admin_page(token, '<div class="err">Admin token did not match.</div>'))
    if kind not in PROVIDER_PREFIX:
        return HTMLResponse(_admin_page(token, '<div class="err">Pick a valid system.</div>'))
    if not email.strip():
        return HTMLResponse(_admin_page(token, '<div class="err">A provider email is required.</div>'))

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            out = _issue_provider(cur, kind, name, email.strip())
    finally:
        conn.close()

    sys_label = "research" if kind == "research" else "clinical"
    portal_word = "subjects" if kind == "research" else "patients"
    mail_body = (
        f"Hello,\n\nYou have been set up as a {sys_label} provider on Vagis.\n\n"
        f"Provider ID: {out['provider_code']}\nKey: {out['secret']}\n\n"
        f"Sign in to the portal with these to manage your {portal_word} and view shared data. "
        f"Keep the Key private.\n\nThanks."
    )
    badge = "res" if kind == "research" else "phy"
    banner = (
        '<div class="result">'
        f'<div class="row"><span class="k">Provider ID <span class="badge {badge}">{sys_label}</span></span>'
        f'<span class="v">{out["provider_code"]}</span></div>'
        f'<div class="row"><span class="k">Key</span><span class="v">{out["secret"]}</span></div>'
        f'<div class="row"><span class="k">Email</span><span class="v">{_esc(email.strip())}</span></div>'
        '<div class="warn">The key is never shown again after you leave this page.</div>'
        + _mailto(email.strip(), "Your Vagis provider access", mail_body, "Email this provider")
        + '</div>'
    )
    return HTMLResponse(_admin_page(token, banner))


@app.post("/admin/ui/list", response_class=HTMLResponse)
def admin_ui_list(token: str = Form("")) -> HTMLResponse:
    if token.strip() != (VAGIS_ADMIN_TOKEN or "").strip() or not VAGIS_ADMIN_TOKEN:
        return HTMLResponse(_admin_page(token, '<div class="err">Admin token did not match.</div>'))
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            cur.execute(
                "SELECT p.provider_code, p.kind, p.name, p.email, p.created_at, "
                "(SELECT COUNT(*) FROM persons x WHERE x.provider_code = p.provider_code) "
                "FROM providers p ORDER BY p.kind, p.seq;")
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        table = '<p class="muted">No providers yet.</p>'
    else:
        body = ""
        for code, kind, name, email, created, n in rows:
            badge = "res" if kind == "research" else "phy"
            body += (f'<tr><td class="mono">{_esc(code)}</td>'
                     f'<td><span class="badge {badge}">{_esc(kind)}</span></td>'
                     f'<td>{_esc(name or "")}</td><td>{_esc(email or "")}</td>'
                     f'<td>{n}</td><td>{created.isoformat()[:10] if created else ""}</td></tr>')
        table = ('<div class="tablewrap"><table><thead><tr><th>Provider ID</th><th>System</th>'
                 '<th>Name</th><th>Email</th><th>People</th><th>Created</th></tr></thead>'
                 f'<tbody>{body}</tbody></table></div>')
    banner = f'<div class="card"><h2>Providers ({len(rows)})</h2>{table}<a class="backbtn" href="/admin">&larr; Back</a></div>'
    return HTMLResponse(_admin_page(token, banner))


# ---- Portal --------------------------------------------------------------
def _portal_login(banner: str = "") -> str:
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vagis Provider Portal</title>{_style()}</head><body>
  <h1>Vagis Provider Portal</h1>
  <p class="sub">Sign in to manage your people and view shared data.</p>
  {banner}
  <div class="card">
    <h2>Sign in</h2>
    <form method="post" action="/portal/ui/dashboard">
      <label>Provider ID</label>
      <input name="provider_code" type="text" placeholder="e.g. RES001 or PHY001" autocomplete="off">
      <label>Key</label>
      <input name="key" type="password" placeholder="Your key" autocomplete="off">
      <button type="submit">Sign in</button>
    </form>
  </div>
</body></html>"""


@app.get("/portal", response_class=HTMLResponse)
def portal_login_page() -> HTMLResponse:
    return HTMLResponse(_portal_login())



def _research_style() -> str:
    return """
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; background: #f4f6f8; color: #1a2b34; }
  .wrap { max-width: 1500px; margin: 0 auto; padding: 16px; height: 100vh; display:flex; flex-direction:column; }
  .topbar { display:flex; align-items:center; justify-content:space-between;
            background:#fff; border:0.5px solid #e2e6ea; border-radius:10px; padding:11px 16px; margin-bottom:12px; }
  .brand { display:flex; align-items:center; gap:9px; font-size:16px; font-weight:600; }
  .brand .logo { width:26px;height:26px;border-radius:6px;background:#1d6fa5;color:#fff;
                 display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700; }
  .tag { font-size:11px;color:#0c447c;background:#e6f1fb;border-radius:5px;padding:2px 8px;font-weight:500; }
  .who { font-size:12px;color:#5f6b72; }
  .banner { background:#e1f5ee;border:1px solid #9fe1cb;border-radius:9px;padding:12px 14px;margin-bottom:12px;
            font-size:14px;color:#085041; }
  .banner .v { font-family:ui-monospace,Menlo,monospace;font-weight:600; }
  .grid { display:grid; grid-template-columns:170px 170px 1fr; gap:10px; align-items:stretch;
          flex:1; min-height:0; }
  .card { background:#fff; border:0.5px solid #e2e6ea; border-radius:10px; padding:11px; }
  .lbl { font-size:10px;font-weight:600;color:#5f6b72;text-transform:uppercase;letter-spacing:.3px;margin-bottom:8px; }
  .leftcol { display:flex; flex-direction:column; gap:10px; min-height:0; height:100%; }
  .leftcol .subjcard { flex:1; display:flex; flex-direction:column; min-height:0; }
  .subjlist { display:flex; flex-direction:column; gap:2px; flex:1; min-height:60px; overflow-y:auto; }
  .subj { font-family:ui-monospace,Menlo,monospace; font-size:11px; color:#3a4750;
          padding:4px 6px; border-radius:4px; cursor:pointer; user-select:none; }
  .subj:hover { background:#f0f4f7; }
  .subj.sel { background:#eef6fc; color:#12456e; font-weight:600; }
  .midcol { display:flex; flex-direction:column; min-height:0; height:100%; }
  .box { margin-bottom:9px; }
  .box.grow { flex:1; display:flex; flex-direction:column; margin-bottom:9px; min-height:0; }
  .box.grow:last-child { margin-bottom:0; }
  .box .hd { display:flex;align-items:center;justify-content:space-between;margin-bottom:6px; }
  .box .hd .name { font-size:10px; font-weight:600; color:#5f6b72;
                   text-transform:uppercase; letter-spacing:.3px; }
  .box .hd .btns { display:flex;gap:4px; }
  .minibtn { font-size:10px; border:0.5px solid #ccd4da; background:#fff; border-radius:5px;
             padding:2px 6px; cursor:pointer; color:#3a4750; }
  .minibtn:hover { background:#f0f4f7; }
  .g1 { border:0.5px solid #cfe0ee; }
  .g2 { border:0.5px solid #d8e6d4; }
  .chip { font-family:ui-monospace,Menlo,monospace; font-size:10.5px; border-radius:4px;
          padding:3px 6px; margin-bottom:3px; display:flex; align-items:center; justify-content:space-between; }
  .g1 .chip { background:#f4f9fd; color:#12456e; }
  .g2 .chip { background:#f4faef; color:#2f5410; }
  .ind .chip { background:#f2f4f6; color:#2c3940; }
  .chip .rm { cursor:pointer; color:#c0392b; font-weight:700; margin-left:6px; }
  .groupbody { flex:1; min-height:90px; overflow-y:auto; border:1px dashed #dfe4e9;
               border-radius:6px; padding:6px; outline:none; }
  .groupbody:focus { border-color:#9fbdd8; }
  .groupbody:empty::before { content:"paste or add subjects"; font-size:10px; color:#b7c0c7; }
  .drop-sm { min-height:24px; border:1px dashed #dfe4e9; border-radius:6px; padding:5px; outline:none; }
  .drop-sm:focus { border-color:#9fbdd8; }
  .rosterbtn { width:100%; background:#fff; border:0.5px solid #cfe0ee; color:#1d6fa5;
               border-radius:6px; padding:7px; font-size:12px; cursor:pointer; margin-bottom:8px; }
  .rosterbtn:hover { background:#f4f9fd; }
  .agent { display:flex; flex-direction:column; min-height:0; height:100%; }
  .msgs { flex:1; overflow-y:auto; display:flex; flex-direction:column; gap:9px; padding:2px; min-height:0; }
  .msg { border-radius:9px; padding:10px 13px; font-size:13.5px; line-height:1.5; max-width:82%; white-space:pre-wrap; }
  .msg.user { background:#eef6fc; color:#12456e; align-self:flex-end; }
  .msg.bot { background:#f6f7f8; color:#2c3940; align-self:flex-start; }
  .msg.think { color:#93a0a8; font-style:italic; }
  .figwrap { padding:6px !important; background:#fff !important; border:0.5px solid #e2e6ea; max-width:92% !important; }
  .figimg { max-width:100%; border-radius:6px; display:block; }
  .dlrow { background:none !important; padding:2px !important; }
  .dlbtn { background:#1d6fa5; color:#fff; border:none; border-radius:8px; padding:9px 14px;
           font-size:12.5px; font-weight:500; cursor:pointer; }
  .dlbtn:hover { background:#185f90; }
  .dlbtn:disabled { opacity:.6; cursor:default; }
  .composer { display:flex; gap:8px; margin-top:11px; }
  .composer textarea { flex:1; border:0.5px solid #e2e6ea; border-radius:9px; padding:10px 12px;
          font-size:13.5px; font-family:inherit; resize:none; height:42px; }
  .send { width:42px; height:42px; background:#1d6fa5; border:none; border-radius:9px; color:#fff;
          font-size:18px; cursor:pointer; }
  .send:disabled { opacity:.5; cursor:default; }
  .issue { display:flex; flex-direction:column; gap:6px; }
  .issue input { border:0.5px solid #e2e6ea; border-radius:6px; padding:6px 8px; font-size:12px; }
  .issue button { background:#1d6fa5;color:#fff;border:none;border-radius:6px;padding:7px 11px;font-size:12px;cursor:pointer; }
  .hint { font-size:10px;color:#93a0a8;margin-top:6px;text-align:center; }
  /* ---- analysis preset bar ---- */
  .presets { border-bottom:0.5px solid #e2e6ea; padding-bottom:9px; margin-bottom:9px; }
  .presetrow { display:flex; align-items:flex-end; gap:8px; }
  .presetrow .pf { flex:1 1 0; }
  .presetrow .pacts { display:flex; gap:6px; flex:0 0 auto; }
  .pf { display:flex; flex-direction:column; gap:4px; min-width:0; }
  .pf label { font-size:10px; font-weight:600; color:#5f6b72;
              text-transform:uppercase; letter-spacing:.3px; }
  .pf select, .pf input[type=date] {
    width:100%; border:0.5px solid #e2e6ea; border-radius:6px; padding:7px 8px;
    font-size:12px; font-family:inherit; background:#fff; color:#2c3940; }
  .pf select:disabled, .pf input[type=date]:disabled { background:#f6f7f8; color:#b7c0c7; }
  .alltog { flex:0 0 auto; }
  .alltog button { border:0.5px solid #ccd4da; background:#fff; color:#3a4750;
                   border-radius:6px; padding:7px 9px; font-size:12px; font-family:inherit;
                   cursor:pointer; height:31px; white-space:nowrap; }
  .alltog button.on { background:#eef6fc; border-color:#9fbdd8; color:#12456e; font-weight:600; }
  .pbtn { background:#1d6fa5; color:#fff; border:none; border-radius:6px;
          padding:7px 11px; font-size:12px; cursor:pointer; height:31px; }
  .pbtn.ghost { background:#fff; border:0.5px solid #ccd4da; color:#3a4750; }
  .periodrow { display:none; align-items:flex-end; gap:8px; margin-top:8px; }
  .periodrow.on { display:flex; }
  .periodrow .ptag { flex:0 0 auto; font-size:10px; font-weight:600; color:#5f6b72;
                     text-transform:uppercase; letter-spacing:.3px; padding-bottom:8px; }
  .notice { position:fixed; top:70px; left:50%; transform:translateX(-50%); z-index:60;
            background:#fff; border:1px solid #9fe1cb; border-left:4px solid #1d9e73;
            border-radius:10px; padding:16px 40px 16px 18px; box-shadow:0 8px 26px rgba(0,0,0,.16);
            max-width:520px; width:90%; }
  .notice .noticex { position:absolute; top:8px; right:10px; background:none; border:none;
                     font-size:20px; color:#93a0a8; cursor:pointer; }
  .notice .result .row { display:flex; gap:10px; margin-bottom:6px; font-size:13px; }
  .notice .result .k { color:#5f6b72; min-width:110px; }
  .notice .result .v { font-family:ui-monospace,Menlo,monospace; font-weight:600; color:#12456e; }
  .notice .warn { font-size:12px; color:#5f6b72; margin:8px 0; }
  .notice a { display:inline-block; background:#1d6fa5; color:#fff; text-decoration:none;
              border-radius:7px; padding:8px 14px; font-size:13px; margin-top:4px; }
  .overlay { position:fixed; inset:0; background:rgba(20,30,40,.28); display:none;
             align-items:flex-start; justify-content:center; padding-top:40px; z-index:50; }
  .overlay.show { display:flex; }
  .rosterpanel { background:#fff; border:0.5px solid #d8dee3; border-radius:12px; width:86%;
                 max-width:640px; max-height:82vh; overflow:hidden; display:flex; flex-direction:column;
                 box-shadow:0 8px 26px rgba(0,0,0,.16); }
  .rosterhd { display:flex; align-items:center; justify-content:space-between; padding:13px 18px; border-bottom:0.5px solid #eceef1; }
  .rosterhd .title { font-size:15px; font-weight:600; color:#1a2b34; }
  .rosterhd .acts { display:flex; gap:8px; align-items:center; }
  .rosterhd .prbtn { font-size:12px; color:#1d6fa5; border:0.5px solid #cfe0ee; background:#fff;
                     border-radius:6px; padding:5px 11px; cursor:pointer; }
  .rosterhd .clbtn { font-size:18px; color:#93a0a8; cursor:pointer; background:none; border:none; }
  .rostermeta { font-size:11px; color:#5f6b72; padding:8px 18px 0; }
  .rosterbody { overflow-y:auto; padding:6px 18px 16px; }
  .rostertbl { width:100%; border-collapse:collapse; font-size:12.5px; }
  .rostertbl th { text-align:left; font-size:10px; color:#5f6b72; text-transform:uppercase; letter-spacing:.3px;
                  padding:9px 8px; border-bottom:1.5px solid #d5dbe0; position:sticky; top:0; background:#fff; }
  .rostertbl td { padding:8px; border-top:0.5px solid #eceef1; color:#2c3940; }
  .rostertbl td.mono { font-family:ui-monospace,Menlo,monospace; }
  @media print {
    body > .wrap { display:none !important; }
    .overlay { position:static; background:none; display:block; padding:0; }
    .rosterpanel { box-shadow:none; border:none; width:100%; max-width:100%; max-height:none; }
    .rosterhd .acts { display:none; }
  }
</style>
"""

_RESEARCH_BODY = r"""
<div class="wrap">
  <div class="topbar">
    <div class="brand"><span class="logo">V</span> Vagis Research Portal <span class="tag">research</span></div>
    <div class="who" id="who"></div>
  </div>
  <div id="banner"></div>
  <div class="grid">

    <div class="leftcol">
      <div class="card subjcard">
        <div class="lbl">Subjects</div>
        <div class="subjlist" id="subjlist"></div>
      </div>
      <div class="card addbox">
        <div class="lbl">Add subjects</div>
        <button type="button" class="rosterbtn" onclick="openRoster()">View subject roster</button>
        <form method="post" action="/portal/ui/issue" class="issue" id="issueForm">
          <input type="hidden" name="provider_code" id="pcField">
          <input type="hidden" name="key" id="keyField">
          <input type="text" name="email" placeholder="new subject email" style="width:100%">
          <input type="text" name="label" placeholder="label (optional)" style="width:100%">
          <button type="submit">Generate &amp; email code</button>
        </form>
      </div>
    </div>

    <div class="midcol">
      <div class="card box ind">
        <div class="hd"><span class="name">Individual</span>
          <span class="btns"><button class="minibtn" onclick="addSel('individual')">&rarr;</button>
          <button class="minibtn" onclick="clearBox('individual')">clear</button></span></div>
        <div id="individual"></div>
        <div class="drop drop-sm" data-box="individual" tabindex="0"></div>
      </div>
      <div class="card box g1 grow">
        <div class="hd"><span class="name">Group 1</span>
          <span class="btns"><button class="minibtn" onclick="addSel('group1')">&rarr;</button>
          <button class="minibtn" onclick="clearBox('group1')">clear</button></span></div>
        <div class="groupbody" id="group1" data-box="group1" tabindex="0"></div>
      </div>
      <div class="card box g2 grow">
        <div class="hd"><span class="name">Group 2</span>
          <span class="btns"><button class="minibtn" onclick="addSel('group2')">&rarr;</button>
          <button class="minibtn" onclick="clearBox('group2')">clear</button></span></div>
        <div class="groupbody" id="group2" data-box="group2" tabindex="0"></div>
      </div>
    </div>

    <div class="card agent">

      <!-- ===== analysis preset bar ===== -->
      <div class="presets">
        <div class="presetrow">

          <div class="pf">
            <label for="pScope">Scope</label>
            <select id="pScope">
              <option value="">&mdash;</option>
              <option value="individual">Individual</option>
              <option value="indcmp">Individual &mdash; compare periods</option>
              <option value="group1">Group 1</option>
              <option value="g1cmp">Group 1 &mdash; compare periods</option>
              <option value="group2">Group 2</option>
              <option value="groupcmp">Group 1 vs Group 2</option>
            </select>
          </div>

          <div class="pf">
            <label for="pStart">Date start</label>
            <input type="date" id="pStart" title="Date the recording was made">
          </div>

          <div class="pf">
            <label for="pEnd">Date end</label>
            <input type="date" id="pEnd" title="Date the recording was made">
          </div>

          <div class="pf alltog">
            <label for="pAll">&nbsp;</label>
            <button type="button" id="pAll" onclick="toggleAllDates()">All dates</button>
          </div>

          <div class="pf">
            <label for="pMode">Mode</label>
            <select id="pMode">
              <option value="">&mdash;</option>
              <optgroup label="Sleep">
                <option value="sleep">Sleep metrics</option>
                <option value="sleep_pwr">Pulse Wave Rhythms metrics</option>
                <option value="sleep_pwr_episodes">Pulse Wave Rhythms episodes</option>
                <option value="sleep_pwr_strips">Pulse Wave Rhythms strip</option>
              </optgroup>
              <optgroup label="Rest">
                <option value="rest">Rest metrics</option>
              </optgroup>
              <optgroup label="Stand">
                <option value="stand">Stand metrics</option>
              </optgroup>
              <optgroup label="Breathwork">
                <option value="breathwork">Breathwork metrics</option>
              </optgroup>
              <optgroup label="Circadian">
                <option value="circadian">Circadian metrics</option>
                <option value="circadian_rhythm">Irregular rhythm metrics</option>
                <option value="circadian_episodes">Episodes</option>
                <option value="circadian_strips">Strips</option>
              </optgroup>
            </select>
          </div>

          <div class="pf">
            <label for="pMetric">Metric</label>
            <select id="pMetric"><option value="">&mdash;</option></select>
          </div>

          <div class="pf">
            <label for="pAnalysis">Analysis</label>
            <select id="pAnalysis"><option value="">&mdash;</option></select>
          </div>

          <div class="pacts">
            <button type="button" class="pbtn ghost" onclick="clearPreset()">Clear</button>
            <button type="button" class="pbtn" onclick="buildPrompt()">Build prompt</button>
          </div>

        </div>

        <div class="periodrow" id="periodRow">
          <span class="ptag">Period A</span>
          <div class="pf"><label for="pA1">From</label>
            <input type="date" id="pA1" title="Date the recording was made"></div>
          <div class="pf"><label for="pA2">To</label>
            <input type="date" id="pA2" title="Date the recording was made"></div>
          <span class="ptag">Period B</span>
          <div class="pf"><label for="pB1">From</label>
            <input type="date" id="pB1" title="Date the recording was made"></div>
          <div class="pf"><label for="pB2">To</label>
            <input type="date" id="pB2" title="Date the recording was made"></div>
        </div>

      </div>
      <!-- ===== end preset bar ===== -->

      <div class="lbl">Analysis agent</div>
      <div class="msgs" id="msgs"></div>
      <div class="composer">
        <textarea id="input" placeholder="Ask the agent to analyze the individual or compare groups..."></textarea>
        <button class="send" id="sendBtn" onclick="send()">&uarr;</button>
      </div>
      <div class="hint">Preview: the agent is connected. Live statistics and figures are being added next.</div>
    </div>

  </div>
</div>

<div class="overlay" id="rosterOverlay" onclick="if(event.target===this)closeRoster()">
  <div class="rosterpanel">
    <div class="rosterhd">
      <span class="title" id="rosterTitle">Subject roster</span>
      <span class="acts">
        <button class="prbtn" onclick="window.print()">Print</button>
        <button class="clbtn" onclick="closeRoster()">&times;</button>
      </span>
    </div>
    <div class="rostermeta" id="rosterMeta"></div>
    <div class="rosterbody">
      <table class="rostertbl">
        <thead><tr><th>SE code</th><th>Email</th><th>Label</th><th>Enrolled</th></tr></thead>
        <tbody id="rosterRows"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const boxes = { individual: [], group1: [], group2: [] };
let selected = null;
const conversation = [];

document.getElementById('who').textContent = PROVIDER + (PROVNAME ? '  \u00b7  ' + PROVNAME : '');
document.getElementById('pcField').value = PROVIDER;
document.getElementById('keyField').value = KEY;

function knownCode(code) { return SUBJECTS.some(function(s){ return s.code === code; }); }

function renderList() {
  const el = document.getElementById('subjlist');
  el.innerHTML = '';
  SUBJECTS.forEach(function(s) {
    const d = document.createElement('div');
    d.className = 'subj' + (selected === s.code ? ' sel' : '');
    d.textContent = s.code;
    d.title = s.label || '';
    d.onclick = function(){ selected = (selected === s.code ? null : s.code); renderList(); };
    el.appendChild(d);
  });
}

function inOtherBox(code, box) {
  return Object.keys(boxes).some(function(b){ return b !== box && boxes[b].indexOf(code) !== -1; });
}

function addCode(box, code) {
  code = (code || '').trim().toUpperCase();
  if (!code || !knownCode(code)) return false;
  Object.keys(boxes).forEach(function(b){
    const i = boxes[b].indexOf(code);
    if (i !== -1) boxes[b].splice(i, 1);
  });
  if (box === 'individual') boxes.individual = [code];
  else if (boxes[box].indexOf(code) === -1) boxes[box].push(code);
  return true;
}

function addSel(box) {
  if (!selected) return;
  addCode(box, selected);
  selected = null;
  renderAll();
}

function removeCode(box, code) {
  const i = boxes[box].indexOf(code);
  if (i !== -1) boxes[box].splice(i, 1);
  renderAll();
}

function clearBox(box) { boxes[box] = []; renderAll(); }

function parsePaste(text) {
  return (text || '').split(/[\s,;]+/).map(function(x){ return x.trim().toUpperCase(); }).filter(Boolean);
}

function renderBox(box) {
  const el = document.getElementById(box);
  el.innerHTML = '';
  boxes[box].forEach(function(code) {
    const c = document.createElement('div');
    c.className = 'chip';
    const span = document.createElement('span');
    span.textContent = code;
    const rm = document.createElement('span');
    rm.className = 'rm';
    rm.textContent = '\u00d7';
    rm.onclick = function(){ removeCode(box, code); };
    c.appendChild(span); c.appendChild(rm);
    el.appendChild(c);
  });
}

function renderAll() { renderList(); ['individual','group1','group2'].forEach(renderBox); }

document.querySelectorAll('[data-box]').forEach(function(d) {
  d.addEventListener('paste', function(e) {
    e.preventDefault();
    const box = d.getAttribute('data-box');
    const text = (e.clipboardData || window.clipboardData).getData('text');
    parsePaste(text).forEach(function(code){ addCode(box, code); });
    renderAll();
  });
});

function groupsPayload() {
  return { individual: boxes.individual.slice(), group1: boxes.group1.slice(), group2: boxes.group2.slice() };
}

function esc(s) { return String(s == null ? '' : s).replace(/[&<>]/g, function(c){ return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c]; }); }

function openRoster() {
  const rows = document.getElementById('rosterRows');
  rows.innerHTML = '';
  SUBJECTS.forEach(function(s) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td class="mono">' + esc(s.code) + '</td><td>' + esc(s.email || '\u2014') +
                   '</td><td>' + esc(s.label || '\u2014') + '</td><td>' + esc(s.enrolled || '\u2014') + '</td>';
    rows.appendChild(tr);
  });
  const today = new Date().toISOString().slice(0, 10);
  document.getElementById('rosterTitle').textContent = 'Subject roster \u00b7 ' + SUBJECTS.length + ' subjects';
  document.getElementById('rosterMeta').textContent =
    PROVIDER + (PROVNAME ? ' \u00b7 ' + PROVNAME : '') + ' \u00b7 generated ' + today;
  document.getElementById('rosterOverlay').classList.add('show');
}

function closeRoster() { document.getElementById('rosterOverlay').classList.remove('show'); }

function addMsg(role, text, cls) {
  const m = document.createElement('div');
  m.className = 'msg ' + (cls || role);
  m.textContent = text;
  document.getElementById('msgs').appendChild(m);
  const box = document.getElementById('msgs');
  box.scrollTop = box.scrollHeight;
  return m;
}

function addFigure(fileId) {
  const wrap = document.createElement('div');
  wrap.className = 'msg bot figwrap';
  const img = document.createElement('img');
  img.className = 'figimg';
  img.alt = 'analysis figure';
  // Fetch the figure with auth, show as blob.
  fetch('/portal/agent/figure', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider_code: PROVIDER, key: KEY, file_id: fileId })
  }).then(function(r){ return r.ok ? r.blob() : null; })
    .then(function(b){ if (b) img.src = URL.createObjectURL(b); else wrap.remove(); })
    .catch(function(){ wrap.remove(); });
  wrap.appendChild(img);
  document.getElementById('msgs').appendChild(wrap);
  const box = document.getElementById('msgs');
  box.scrollTop = box.scrollHeight;
}

function addSummaryButton(replyText, figureIds) {
  const row = document.createElement('div');
  row.className = 'msg bot dlrow';
  const btn = document.createElement('button');
  btn.className = 'dlbtn';
  btn.textContent = 'Download summary (PDF)';
  btn.onclick = function() {
    btn.disabled = true; btn.textContent = 'Preparing...';
    fetch('/portal/agent/summary', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider_code: PROVIDER, key: KEY,
                             title: 'Vagis analysis summary', text: replyText, figure_ids: figureIds })
    }).then(function(r){ return r.ok ? r.blob() : null; })
      .then(function(b){
        btn.disabled = false; btn.textContent = 'Download summary (PDF)';
        if (!b) return;
        const url = URL.createObjectURL(b);
        const a = document.createElement('a');
        a.href = url; a.download = 'vagis_analysis_summary.pdf'; a.click();
        URL.revokeObjectURL(url);
      }).catch(function(){ btn.disabled = false; btn.textContent = 'Download summary (PDF)'; });
  };
  row.appendChild(btn);
  document.getElementById('msgs').appendChild(row);
  const box = document.getElementById('msgs');
  box.scrollTop = box.scrollHeight;
}

async function send() {
  const inp = document.getElementById('input');
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';
  const btn = document.getElementById('sendBtn');
  btn.disabled = true;
  addMsg('user', msg);
  conversation.push({ role: 'user', content: msg });
  const thinking = addMsg('bot', 'Analyzing...', 'bot think');
  let secs = 0;
  const ticker = setInterval(function(){
    secs += 5;
    thinking.textContent = 'Analyzing... (' + secs + 's — larger analyses can take a minute or two)';
  }, 5000);
  try {
    const res = await fetch('/portal/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider_code: PROVIDER, key: KEY, message: msg,
                             history: conversation, groups: groupsPayload(),
                             mode: document.getElementById('pMode').value })
    });
    const data = await res.json();
    clearInterval(ticker); thinking.remove();
    const reply = (data && data.reply) ? data.reply : (data && data.detail ? data.detail : 'No response.');
    addMsg('bot', reply);
    conversation.push({ role: 'assistant', content: reply });
    const figs = (data && data.figures) ? data.figures : [];
    figs.forEach(addFigure);
    if (figs.length) addSummaryButton(reply, figs);
  } catch (e) {
    clearInterval(ticker); thinking.remove();
    addMsg('bot', 'Could not reach the agent: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById('input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

/* ===================== analysis preset bar ===================== */

/* Metric options per mode, from each master CSV's columns. Timestamp and
   identifier columns are omitted — they drive the date filter, they are not
   things to analyse. */
const METRICS = {
  sleep: ["duration_min","deep_min","rem_min","nrem_min","awake_min","total_sleep_min",
          "mean_hr","resp_rate_brpm","adi_score","rem_vlf_pct","deep_vlf_pct",
          "night_mean_vlf_pct","position_per_hour","pwad_score","PPI_PWA_Coup"],
  rest: ["duration_min","mean_hr","avg_ibi_ms","resp_rate_brpm","avg_pwa","coherence",
         "phase_deg","breath_rate_cpm","hrdb_bpm","ei_ratio","band1_endothelial",
         "band2_neurogenic","band3_myogenic","band4_respiratory"],
  stand: ["duration_s","motion_valid","cue_time_s","stand_onset_s","beats_supine",
          "beats_stand","beats_t3_supine","beats_t3_stand","supine_hr","supine_pwa",
          "supine_dc","supine_ttp_ms","supine_ttp_frac","supine_rise_steep",
          "supine_resp_brpm","supine_resp_conf","stand_hr","stand_peak_hr","stand_pwa",
          "stand_dc","stand_ttp_ms","stand_ttp_frac","stand_rise_steep","delta_hr",
          "delta_pwa_pct","delta_dc_pct","delta_ttp_frac","beats_return","return_hr",
          "return_pwa","return_dc","return_ttp_ms","return_ttp_frac","return_sdnn",
          "delta_return_hr","delta_return_pwa_pct","delta_return_dc_pct",
          "delta_return_ttp_frac","ratio_3015","ratio_3015_min_ppi","ratio_3015_max_ppi",
          "pots_rise_1min","pots_rise_2min","pots_met","hr_time_to_peak_s",
          "hr_rise_rate_bpm_s","hr_onset_latency_s","hr_peak_to_plateau",
          "pwa_onset_latency_s","pwa_settle_time_s","dc_onset_latency_s",
          "dc_settle_time_s","venoarteriolar_ratio"],
  breathwork: ["duration_min","mean_hr","ln_lf","rsa","coherence","phase_deg",
               "breath_rate_cpm","hrdb_bpm","ei_ratio","pwv_sd"],
  circadian: ["coupling_slope","coupling_intercept","coupling_r","cardiac_cost",
              "still_pct","light_pct","mod_pct","hr_still","hr_light","hr_mod",
              "hr_mean","hr_min","hr_max","hr_range",
              "recovery_tau_sec","recovery_hr_drop","recovery_confidence"],
  circadian_rhythm: [], circadian_episodes: [],
  circadian_strips: ["ppg_green"],
  /* night_duty_pct is the burden measure — episode count saturates once
     episodes cover much of the night */
  sleep_pwr: ["n_episodes","episode_min_total","night_duty_pct","control_duty_pct"],
  sleep_pwr_episodes: [],
  sleep_pwr_strips: ["pwa","accel"]
};

/* Log-style files: the agent renders the whole file as a table, so there is no
   single metric to pick. */
const TABLE_MODES = ["sleep_pwr_episodes", "circadian_rhythm", "circadian_episodes"];

const pModeEl = document.getElementById('pMode');
const pMetricEl = document.getElementById('pMetric');
const pScopeEl = document.getElementById('pScope');
let allDates = true;

function fillMetrics() {
  if (TABLE_MODES.indexOf(pModeEl.value) !== -1) {
    pMetricEl.innerHTML = '<option value="all">All columns (table)</option>';
    pMetricEl.disabled = true;
    return;
  }
  const list = METRICS[pModeEl.value] || [];
  pMetricEl.innerHTML = '<option value="">\u2014</option>';
  list.forEach(function(m) {
    const o = document.createElement('option');
    o.value = m; o.textContent = m;
    pMetricEl.appendChild(o);
  });
  pMetricEl.disabled = list.length === 0;
}
pModeEl.addEventListener('change', fillMetrics);

function applyAllDates() {
  const st = document.getElementById('pStart');
  const en = document.getElementById('pEnd');
  document.getElementById('pAll').classList.toggle('on', allDates);
  st.disabled = allDates; en.disabled = allDates;
  if (allDates) { st.value = ''; en.value = ''; }
}
function toggleAllDates() { allDates = !allDates; applyAllDates(); }
['pStart','pEnd'].forEach(function(id) {
  document.getElementById(id).addEventListener('input', function() {
    if (allDates) { allDates = false; applyAllDates(); }
  });
});

/* Compare-periods scopes swap the single window for Period A / Period B. */
function applyScope() {
  const cmp = (pScopeEl.value === 'indcmp' || pScopeEl.value === 'g1cmp');
  document.getElementById('periodRow').classList.toggle('on', cmp);
  ['pStart','pEnd'].forEach(function(id) {
    document.getElementById(id).closest('.pf').style.display = cmp ? 'none' : '';
  });
  document.querySelector('.alltog').style.display = cmp ? 'none' : '';
}
pScopeEl.addEventListener('change', applyScope);

function clearPreset() {
  pScopeEl.value = '';
  allDates = true; applyAllDates();
  document.getElementById('pStart').value = '';
  document.getElementById('pEnd').value = '';
  ['pA1','pA2','pB1','pB2'].forEach(function(id) {
    document.getElementById(id).value = '';
  });
  pModeEl.value = ''; fillMetrics();
  document.getElementById('pAnalysis').value = '';
  applyScope();
  document.getElementById('input').value = '';
}

/* ---------------- preset prompt text ----------------
   Written so the same selection produces the same analysis every time. The
   agent starts each session with no context, so everything it needs to read
   these files correctly — and every caveat it must not overstate — lives here. */

const PWR_CONTEXT = [
  "BACKGROUND — read before analysing.",
  "Pulse Wave Rhythms measures how much of the night pulse wave amplitude spends",
  "suppressed. Amplitude is averaged onto a 1-second grid from clean beats only; a",
  "second counts as SUPPRESSED when it sits more than 70% below its running baseline.",
  "DUTY is the percentage of suppressed seconds over a rolling 10 minutes. An EPISODE",
  "is a contiguous stretch where duty stays at or above 1%, lasting at least 5 minutes.",
  "",
  "Nothing here counts discrete events, deliberately. Counting drops needed both a",
  "depth threshold and a minimum duration, and under-counted badly — a strip with a",
  "dozen visible dips reported six. Duty needs neither, and does not care whether",
  "suppression arrives as several clean falls or one long trough. So do not report a",
  "number of drops, an events-per-hour index, or a drop depth: they are not in the data.",
  "",
  "Suppression indexes sympathetic AROUSALS. It is not oxygen desaturation and picks up",
  "arousals that never desaturate, which oximetry cannot see. That is the point of the",
  "measure, not a shortcoming. Never describe it as apnea, SDB, hypopnea or any",
  "breathing diagnosis, and never present duty as a severity score.",
  "",
  "night_duty_pct is the best night-to-night burden measure. Episode count saturates",
  "once episodes cover much of the night, so lead with duty and episode minutes.",
  "",
  "Two strips are exported. The DISTURBANCE strip is the highest-duty 10 minutes",
  "inside whichever of the night's three longest episodes offers it — episode length",
  "tracks disturbance better than duty alone. The CONTROL strip is the quietest 10",
  "minutes of the night, chosen with a movement gate so that low duty means settled",
  "rather than merely noisy. The control is the subject's own baseline, which makes",
  "the comparison within-subject and free of any assumption about normal amplitude.",
  "Neither strip is the worst stretch of the night; do not present either as",
  "representative of the whole recording.",
  "",
  "TIMESTAMPS — get this exactly right. Every timestamp column (start, end, iso_ts,",
  "strip_start) is ALREADY local clock time, written with a trailing UTC offset such as",
  "2026-07-27T22:40:15-07:00. The clock part is what you want. Do NOT use timezone",
  "conversion of any kind: no utc=True, no tz_convert, no tz_localize. Instead take the",
  "first 19 characters of the string and parse that, e.g.",
  "    pd.to_datetime(df['start'].str.slice(0, 19))",
  "That yields naive local time and cannot be shifted by anything. A 22:40 recording must",
  "read 22:40 in every table and on every axis. Verify one value before you finish.",
  "algo_version records the detection thresholds behind each row. Use only rows whose",
  "algo_version matches the most recent recording's, and state which version that is."
].join("\n");

const CIRC_CONTEXT = [
  "BACKGROUND — read before analysing.",
  "This is the irregular rhythm detector from circadian recordings. The recording is",
  "cut into 30-second windows on a 15-second hop. Each window first passes a MOTION",
  "GATE based on beat quality and movement; windows that fail are still logged, with",
  "gate_pass 0 and the detector columns left blank, so the record is complete.",
  "",
  "A gated window is then judged on TWO independent axes, and both must agree before",
  "it is flagged:",
  "  calm_fraction        — how much of the window has settled beat-to-beat timing",
  "  coupling_concordance — whether beat interval and pulse amplitude move together",
  "confidence is the distance from the decision boundary, 0 to 1, not a probability.",
  "Consecutive flagged windows are grouped into episodes.",
  "",
  "NAMING — call this irregular rhythm and nothing else, in text, table headers,",
  "figure titles and axis labels, however the question is phrased. Never substitute a",
  "named cardiac condition or its abbreviation, and never restate this instruction in",
  "your answer.",
  "",
  "This flags a region for human review and would need ECG confirmation to mean",
  "anything clinically. Do not describe it as detecting or diagnosing a condition, and",
  "do not present confidence as a likelihood of disease.",
  "",
  "gated_pct is a data-quality figure, not a finding. A recording with a low gated_pct",
  "had a lot of movement and its flagged counts mean less; say so when it is low.",
  "",
  "TIMESTAMPS — get this exactly right. Every timestamp column is ALREADY local clock",
  "time, written with a trailing UTC offset such as 2026-07-27T22:40:15-07:00. Do NOT",
  "use timezone conversion of any kind: no utc=True, no tz_convert, no tz_localize.",
  "Take the first 19 characters of the string and parse that, e.g.",
  "    pd.to_datetime(df['episode_start'].str.slice(0, 19))",
  "That yields naive local time and cannot be shifted by anything. Verify one value",
  "before you finish."
].join("\n");

const CIRC_PRESETS = {
  circadian_rhythm: [
    "TASK — irregular rhythm summary across recordings. Produce a TABLE ONLY.",
    "Use the circadian_rhythm file. One row per recording.",
    "",
    "One row per recording, ordered by date:",
    "  recording_date, start_time, duration_sec, gated_pct, flagged_windows,",
    "  episode_count, flagged_minutes, max_confidence",
    "",
    "gated_pct stays in the table because a low value means the flagged counts on",
    "that row carry less weight. Print the number; do not comment on it.",
    "",
    "No figure. No commentary, no interpretation, no summary paragraph, no caveats.",
    "Output the table and nothing else."
  ].join("\n"),

  circadian_episodes: [
    "TASK — irregular rhythm episodes. Produce a TABLE ONLY.",
    "Use the circadian_episodes file. One row per episode.",
    "",
    "One row per episode, ordered by start:",
    "  recording_date, episode_id, start (HH:MM), duration_min, n_windows,",
    "  mean_calm_fraction, mean_coupling_concordance, mean_confidence",
    "Finish with a total row: number of episodes and total episode minutes.",
    "",
    "No figure. No commentary, no interpretation, no summary paragraph, no caveats.",
    "Output the table and nothing else."
  ].join("\n"),

  circadian_strips: [
    "TASK — irregular rhythm strips, episode against sinus control.",
    "Use the circadian_strips file: recording_ts, strip_type, strip_id, rel_ms,",
    "ppg_green. strip_type is either sinus or episode. Each strip is a 30-second clip",
    "of raw green PPG at 25 Hz; rel_ms restarts at 0 within each strip. Every",
    "recording contributes one sinus control strip and one strip per episode.",
    "",
    "PREPARE THE SIGNAL — the samples are raw and unusable as they arrive:",
    "  1. INVERT (multiply by -1). The sensor value falls as blood volume rises, so",
    "     without this every pulse points downward.",
    "  2. High-pass at 0.5 Hz to remove baseline wander — a 2nd-order Butterworth,",
    "     applied forwards and backwards so the beats are not shifted in time.",
    "Do this before plotting anything. Do not smooth further; the beat-to-beat shape",
    "is the whole point.",
    "",
    "PANEL LIST — build it once, explicitly. One panel per unique strip_id, and one",
    "figure in total:",
    "    df  = df.drop_duplicates(subset=['strip_id', 'rel_ms'])",
    "    ids = [i for i in dict.fromkeys(df['strip_id']) if is_sinus(i)] \\",
    "        + [i for i in dict.fromkeys(df['strip_id']) if not is_sinus(i)]",
    "Create exactly len(ids) axes and draw ids[i] on axes[i]. Sinus control comes",
    "first, then episode strips in time order. Each strip_id is drawn ONCE — do not",
    "place the sinus strip first and then also iterate over every strip, which draws",
    "it twice. A recording contributes exactly one sinus control strip.",
    "",
    "Each x-axis runs 0 to 30 seconds. Y label 'PPG (filtered)', x label 'Seconds",
    "within strip'. One subplots call, one savefig, one figure in the response.",
    "",
    "PANEL TITLES — large and readable, not a caption. Bold, fontsize 14, left",
    "aligned, naming the strip and the recording it came from:",
    "    Sinus control — 2026-07-22 12:47",
    "    Irregular rhythm episode 1 — 2026-07-22 12:47",
    "Put the raw strip_id in the panel's top-right corner instead, as a small grey",
    "annotation at fontsize 7, colour '#888888', for traceability only.",
    "",
    "Y-AXIS — scale each panel to its OWN trace. Panels do NOT share a y-scale, so a",
    "clean low-amplitude strip fills its panel instead of sitting as a flat line in a",
    "range set by a noisier strip. For each panel, from that strip's filtered y:",
    "    lim = 1.15 * float(np.percentile(np.abs(y), 99.5))",
    "    lim = max(lim, 1000.0)",
    "    ax.set_ylim(-lim, lim)",
    "The 99.5th percentile stops a few motion spikes setting the scale for the whole",
    "panel; those spikes run off the top and bottom, which is intended. The floor of",
    "1000 stops an almost flat strip being blown up into pure noise.",
    "",
    "WHEN THERE ARE NO EPISODE STRIPS — a sinus control strip with nothing beneath it",
    "is the normal, expected result, and the figure must say so rather than leave the",
    "reader wondering whether panels are missing. Draw the sinus panel, then one line",
    "of text in the figure directly below it, centred, fontsize 11, colour '#444444':",
    "    No irregular rhythm episode strips in this selection.",
    "Use fig.text below the sinus axes and leave layout room so it is not clipped.",
    "No box, no explanation, nothing else on that line.",
    "",
    "Grid on at alpha 0.2. Draw nothing else: no markers, no shading, no vertical",
    "lines.",
    "",
    "OUTPUT THE FIGURE AND NOTHING ELSE. No commentary, no interpretation, no",
    "description of the traces, no comparison between strips, no summary paragraph,",
    "no caveats, no next steps. Do not say what the strips show. The researcher reads",
    "the figure; if they want it interpreted they will ask, and you answer then.",
    "",
    "The only text permitted alongside the figure is a bare note naming any recording",
    "in the selection that produced no sinus control strip, e.g. 'No sinus control",
    "strip for 2026-07-24.' State it and stop; do not explain why."
  ].join("\n")
};

const PWR_PRESETS = {
  sleep_pwr_strips: [
    "TASK — Pulse Wave Rhythms strips, disturbance against control.",
    "Use the sleep_pwr_strips file (recording_ts, episode_id, strip_id, rel_ms, pwa,",
    "accel, tier). Each recording holds TWO 10-minute strips: the DISTURBANCE strip,",
    "whose episode_id is an episode number (E1, E2, ...), and the CONTROL strip, whose",
    "episode_id is the literal string CONTROL. Split on episode_id.",
    "Take the disturbance strip's clock start from sleep_pwr_episodes (strip_start on",
    "the matching episode_id) and the control's from sleep_pwr (control_start).",
    "",
    "FIGURE — four panels stacked, built in one pass with a single subplots call:",
    "  1. DISTURBANCE pwa, black trace",
    "  2. DISTURBANCE accel, black trace, about one third the height of panel 1",
    "  3. CONTROL pwa, black trace",
    "  4. CONTROL accel, black trace, about one third the height of panel 3",
    "Label the two pwa panels 'Disturbance' and 'Control' with their clock windows.",
    "",
    "Y-AXIS RANGES — identical on both strips, so the contrast is real and not an",
    "artefact of scaling:",
    "  pwa   : ALWAYS 0 to 50000. Do not autoscale. Beats above 50000 are motion and",
    "          run off the top; draw the full trace clipped at the axis top and say",
    "          how many beats went above it in each strip.",
    "  accel : ALWAYS 0 to 10, even when a trace sits near 1.",
    "",
    "Build each x-axis by adding rel_ms to that strip's own start, labelled 'Time of",
    "day' as HH:MM. Y labels 'Pulse Wave Amplitude' and 'Accel'.",
    "Title: Pulse Wave Rhythms - <subject> - <recording date>.",
    "accel is ALREADY time-aligned to pwa — do not shift it.",
    "Draw nothing else: no shading, no markers, no vertical lines, no arrows.",
    "",
    "REPORT briefly: the recording date, which episode the disturbance strip came from",
    "and both clock windows, then a two-row table read straight from the files — do not",
    "recompute anything:",
    "  disturbance: strip_duty_pct, strip_move_sec, strip_peak_accel",
    "  control    : control_duty_pct, control_move_sec, control_peak_accel",
    "Finish with one or two sentences contrasting the two traces. Never call either",
    "accel trace quiet, calm or flat without quoting its move and peak numbers — a",
    "strip can look flat at this scale and still hold a 15-second burst peaking above",
    "30. The control is there to show what this subject's amplitude looks like when",
    "undisturbed; say plainly whether the disturbance strip differs from it.",
    "Nothing else. No extra analysis, no per-second scanning of the accel channel."
  ].join("\n"),

  sleep_pwr_episodes: [
    "TASK — Pulse Wave Rhythms episodes. Produce a TABLE ONLY.",
    "Use the sleep_pwr_episodes file. One row per episode; episodes never overlap.",
    "",
    "One row per episode, ordered by start:",
    "  recording_date, episode_id, start (HH:MM), end (HH:MM), dur_min,",
    "  duty_mean_pct, duty_max_pct",
    "Finish with a total row: number of episodes and total episode minutes.",
    "",
    "No figure. No commentary, no interpretation, no summary paragraph, no caveats.",
    "Output the table and nothing else."
  ].join("\n"),

  sleep_pwr: [
    "TASK — Pulse Wave Rhythms summary. Produce a TABLE ONLY.",
    "Use the sleep_pwr file. One row per recording.",
    "",
    "One row per recording, ordered by date:",
    "  recording_date, n_episodes, episode_min_total, night_duty_pct,",
    "  strip_start (HH:MM), strip_duty_pct",
    "",
    "No figure. No commentary, no interpretation, no summary paragraph, no caveats.",
    "Output the table and nothing else."
  ].join("\n")
};

function scopeLine() {
  const v = pScopeEl.value;
  if (v === 'individual') return "SCOPE: the single selected subject.";
  if (v === 'group1')     return "SCOPE: the subjects in Group 1, analysed together.";
  if (v === 'group2')     return "SCOPE: the subjects in Group 2, analysed together.";
  if (v === 'groupcmp')   return "SCOPE: compare Group 1 against Group 2. Report each group separately, then the difference between them.";
  if (v === 'indcmp' || v === 'g1cmp') {
    const a1 = document.getElementById('pA1').value, a2 = document.getElementById('pA2').value;
    const b1 = document.getElementById('pB1').value, b2 = document.getElementById('pB2').value;
    const who = (v === 'indcmp') ? "the single selected subject" : "the subjects in Group 1";
    if (!a1 || !a2 || !b1 || !b2) return "SCOPE: " + who + ", comparing two periods (dates not set — ask before analysing).";
    return "SCOPE: " + who + ", split into two periods by recording date and compared.\n"
         + "  Period A: " + a1 + " to " + a2 + "\n"
         + "  Period B: " + b1 + " to " + b2 + "\n"
         + "Report each period separately, then the difference between them.";
  }
  return "SCOPE: the current selection.";
}

function dateLine() {
  if (pScopeEl.value === 'indcmp' || pScopeEl.value === 'g1cmp') return "";
  if (allDates) return "DATES: use every recording available.";
  const a = document.getElementById('pStart').value, b = document.getElementById('pEnd').value;
  if (!a && !b) return "DATES: use every recording available.";
  if (a && b)   return "DATES: use recordings made from " + a + " to " + b + " inclusive.";
  if (a)        return "DATES: use recordings made on or after " + a + ".";
  return "DATES: use recordings made on or before " + b + ".";
}

function buildPrompt() {
  const mode = pModeEl.value;
  const box = document.getElementById('input');
  if (!mode) { box.value = "Choose a Mode first."; box.focus(); return; }
  const isCirc = (CIRC_PRESETS[mode] !== undefined);
  const preset = isCirc ? CIRC_PRESETS[mode] : PWR_PRESETS[mode];
  if (!preset) {
    box.value = "No preset is written for this mode yet — type your request instead.";
    box.focus(); return;
  }
  const metric = pMetricEl.value;
  const parts = [preset, "", scopeLine()];
  const dl = dateLine();
  if (dl) parts.push(dl);
  if (metric && metric !== 'all') parts.push("FOCUS on the metric: " + metric + ".");
  const tableOnly = (mode === 'sleep_pwr' || mode === 'sleep_pwr_episodes');
  parts.push("", isCirc ? CIRC_CONTEXT : PWR_CONTEXT, "",
             tableOnly ? "One code execution. Table only — no figure."
                       : "One code execution, one figure.");
  box.value = parts.join("\n");
  box.focus();
}

fillMetrics();
applyAllDates();
applyScope();

renderAll();
</script>
</body></html>
"""

def _research_dashboard(prov: dict, key: str, cur, banner: str = "") -> str:
    """Rich research analysis workspace: thin subject list, Individual/Group1/Group2
    boxes with click-to-add + paste, and a live agent chat panel."""
    import json as _json
    cur.execute("SELECT person_code, label, email, created_at FROM persons "
                "WHERE provider_code = %s AND kind='research' ORDER BY person_seq;",
                (prov["provider_code"],))
    rows = cur.fetchall()
    subjects = [{"code": r[0], "label": r[1] or "", "email": r[2] or "",
                 "enrolled": r[3].isoformat()[:10] if r[3] else ""} for r in rows]
    subjects_json = _json.dumps(subjects)
    prov_code = prov["provider_code"]
    name = prov.get("name") or ""

    head = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vagis Research Portal</title>{_research_style()}</head><body>
<script>
const SUBJECTS = {subjects_json};
const PROVIDER = {_json.dumps(prov_code)};
const KEY = {_json.dumps(key)};
const PROVNAME = {_json.dumps(name)};
</script>
"""
    notice = ""
    if banner:
        notice = (
            '<div class="notice" id="notice">'
            '<button class="noticex" onclick="document.getElementById(\'notice\').remove()">&times;</button>'
            + banner + '</div>'
        )
    return head + notice + _RESEARCH_BODY


def _portal_dashboard(prov: dict, key: str, cur, banner: str = "") -> str:
    """Dispatch: research providers get the rich analysis workspace;
    clinical providers keep the plain dashboard for now."""
    if prov["kind"] == "research":
        return _research_dashboard(prov, key, cur, banner)
    return _plain_dashboard(prov, key, cur, banner)


def _plain_dashboard(prov: dict, key: str, cur, banner: str = "") -> str:
    kind = prov["kind"]
    is_research = kind == "research"
    word = "subject" if is_research else "patient"
    words = "subjects" if is_research else "patients"

    cur.execute("SELECT person_code, person_seq, label, email, created_at FROM persons "
                "WHERE provider_code = %s ORDER BY person_seq;", (prov["provider_code"],))
    people = cur.fetchall()

    # Which modes each person has, and (clinical) whether data is currently held.
    if is_research:
        cur.execute("SELECT person_code, mode FROM research_uploads WHERE person_code IN "
                    "(SELECT person_code FROM persons WHERE provider_code = %s);", (prov["provider_code"],))
    else:
        purge_expired(cur)
        cur.execute("SELECT person_code, mode FROM clinical_holds WHERE person_code IN "
                    "(SELECT person_code FROM persons WHERE provider_code = %s);", (prov["provider_code"],))
    modes_by: dict[str, list[str]] = {}
    for pc, m in cur.fetchall():
        modes_by.setdefault(pc, []).append(m)

    if people:
        rows = ""
        for pcode, seq, label, email, created in people:
            has = sorted(modes_by.get(pcode, []))
            if has:
                if is_research:
                    marks = "".join(f'<span class="pill">{_esc(_mode_label(m))}</span>' for m in has)
                else:
                    marks = ('<span class="flag">NEW DATA</span>'
                             + "".join(f'<span class="pill">{_esc(_mode_label(m))}</span>' for m in has))
            else:
                marks = '<span class="muted" style="margin-left:6px">no data</span>'
            meta = " &middot; ".join(x for x in [_esc(label) if label else "", _esc(email) if email else ""] if x)
            rows += (
                f'<div class="subrow"><div><span class="mono">{_esc(pcode)}</span>'
                f'{(" &middot; " + meta) if meta else ""}{marks}'
                f'<div class="muted">issued {created.isoformat()[:10] if created else ""}</div></div>'
                f'<form class="inline" method="post" action="/portal/ui/person">'
                f'{_hidden(prov["provider_code"], key)}'
                f'<input type="hidden" name="person_code" value="{_esc(pcode)}">'
                f'<button class="small" type="submit">View</button></form></div>'
            )
        people_block = rows
    else:
        people_block = f'<p class="muted">No {words} yet. Issue a code below to add your first.</p>'

    badge = "res" if is_research else "phy"
    retention_note = ("Study data is stored persistently for your protocol."
                      if is_research else
                      f"Patient data is held only briefly and auto-deletes {CLINICAL_HOLD_HOURS}h after the patient sends it.")

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vagis Portal</title>{_style()}</head><body>
  <h1>Vagis Portal <span class="badge {badge}">{_esc(kind)}</span></h1>
  <p class="sub">Signed in as <span class="mono">{_esc(prov["provider_code"])}</span>{(" &middot; " + _esc(prov["name"])) if prov.get("name") else ""} &middot; {retention_note}</p>
  {banner}
  <div class="card">
    <h2>Issue a new {word} code</h2>
    <p class="muted">Creates the next code under your account. Email it to the {word} to enrol them.</p>
    <form method="post" action="/portal/ui/issue">
      {_hidden(prov["provider_code"], key)}
      <label>{word.capitalize()} email (required)</label>
      <input name="email" type="text" placeholder="{word}@example.com">
      <label>Label (optional, private to you)</label>
      <input name="label" type="text" placeholder="e.g. pilot {word} 1">
      <button type="submit">Issue {word} code</button>
    </form>
  </div>
  <div class="card">
    <h2>Your {words} ({len(people)})</h2>
    {people_block}
  </div>
</body></html>"""


def _auth_provider(cur, provider_code: str, key: str):
    p = authenticate_provider(cur, provider_code, key)
    if p:
        # fetch name
        cur.execute("SELECT name FROM providers WHERE provider_code = %s;", (p["provider_code"],))
        row = cur.fetchone()
        p["name"] = row[0] if row else None
    return p


@app.post("/portal/ui/dashboard", response_class=HTMLResponse)
def portal_dashboard(provider_code: str = Form(""), key: str = Form("")) -> HTMLResponse:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            p = _auth_provider(cur, provider_code, key)
            if not p:
                return HTMLResponse(_portal_login('<div class="err">Wrong Provider ID or Key.</div>'))
            page = _portal_dashboard(p, key, cur)
    finally:
        conn.close()
    return HTMLResponse(page)


@app.post("/portal/ui/issue", response_class=HTMLResponse)
def portal_issue(provider_code: str = Form(""), key: str = Form(""),
                 label: str = Form(""), email: str = Form("")) -> HTMLResponse:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            p = _auth_provider(cur, provider_code, key)
            if not p:
                return HTMLResponse(_portal_login('<div class="err">Wrong Provider ID or Key.</div>'))
            word = "subject" if p["kind"] == "research" else "patient"
            if not email.strip():
                return HTMLResponse(_portal_dashboard(p, key, cur,
                    f'<div class="err">A {word} email is required.</div>'))
            try:
                out = _issue_person(cur, p, label, email.strip())
            except HTTPException as e:
                return HTMLResponse(_portal_dashboard(p, key, cur, f'<div class="err">{_esc(e.detail)}</div>'))
            mail_body = (
                f"Hello,\n\nYou've been invited to share your Vagis data. Your enrollment code is:\n\n"
                f"{out['person_code']}\n\n"
                f"Open the Vagis app, go to Data Share, enter this code, and follow the prompts. "
                f"Only your metric summaries are shared \u2014 your raw recordings stay on your phone.\n\nThanks."
            )
            banner = (
                '<div class="result">'
                f'<div class="row"><span class="k">New {word} code</span><span class="v">{_esc(out["person_code"])}</span></div>'
                f'<div class="row"><span class="k">Email</span><span class="v">{_esc(email.strip())}</span></div>'
                f'<div class="warn">Send this code to the {word}. They enter it in the app to share their data with you.</div>'
                + _mailto(email.strip(), "Your Vagis enrollment code", mail_body, f"Email this {word}")
                + '</div>'
            )
            page = _portal_dashboard(p, key, cur, banner)
    finally:
        conn.close()
    return HTMLResponse(page)


@app.post("/portal/ui/person", response_class=HTMLResponse)
def portal_person(provider_code: str = Form(""), key: str = Form(""),
                  person_code: str = Form("")) -> HTMLResponse:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            p = _auth_provider(cur, provider_code, key)
            if not p:
                return HTMLResponse(_portal_login('<div class="err">Wrong Provider ID or Key.</div>'))
            pc = (person_code or "").strip().upper()
            cur.execute("SELECT provider_code, label FROM persons WHERE person_code = %s;", (pc,))
            row = cur.fetchone()
            if not row or row[0] != p["provider_code"]:
                word = "subject" if p["kind"] == "research" else "patient"
                return HTMLResponse(_portal_login(f'<div class="err">{word.capitalize()} not found under your account.</div>'))
            label = row[1]
            if p["kind"] == "research":
                cur.execute("SELECT mode, row_count, uploaded_at FROM research_uploads "
                            "WHERE person_code = %s ORDER BY mode;", (pc,))
                extra = ""
            else:
                purge_expired(cur)
                cur.execute("SELECT mode, row_count, uploaded_at, expires_at FROM clinical_holds "
                            "WHERE person_code = %s ORDER BY mode;", (pc,))
            uploads = cur.fetchall()

            if uploads:
                items = ""
                for u in uploads:
                    mode, rc, up = u[0], u[1], u[2]
                    up_s = up.isoformat()[:16].replace("T", " ") if up else ""
                    exp_note = ""
                    if p["kind"] == "clinical":
                        exp = u[3]
                        exp_s = exp.isoformat()[:16].replace("T", " ") if exp else ""
                        exp_note = f' &middot; auto-deletes {exp_s}'
                    items += (
                        f'<div class="subrow"><div><b>{_esc(_mode_label(mode))}</b>'
                        f'<div class="muted">{rc if rc is not None else "?"} sessions &middot; updated {up_s}{exp_note}</div></div>'
                        f'<form class="inline" method="post" action="/portal/ui/view">'
                        f'{_hidden(p["provider_code"], key)}'
                        f'<input type="hidden" name="person_code" value="{_esc(pc)}">'
                        f'<input type="hidden" name="mode" value="{_esc(mode)}">'
                        f'<button class="small" type="submit">Open</button></form></div>'
                    )
                block = items
            else:
                block = '<p class="muted">No data shared yet' + ('' if p["kind"] == "research" else ' (or it has expired)') + '.</p>'

            back = (f'<form class="inline" method="post" action="/portal/ui/dashboard">'
                    f'{_hidden(p["provider_code"], key)}'
                    f'<button class="backbtn" type="submit">&larr; Back</button></form>')
            word = "Subject" if p["kind"] == "research" else "Patient"
            page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{word} {_esc(pc)}</title>{_style()}</head><body>
  <h1>{word} <span class="mono">{_esc(pc)}</span></h1>
  <p class="sub">{_esc(label) if label else "No label"} &middot; {_esc(p["provider_code"])}</p>
  {back}
  <div class="card"><h2>Shared data</h2>{block}</div>
</body></html>"""
    finally:
        conn.close()
    return HTMLResponse(page)


@app.post("/portal/ui/view", response_class=HTMLResponse)
def portal_view(provider_code: str = Form(""), key: str = Form(""),
                person_code: str = Form(""), mode: str = Form("")) -> HTMLResponse:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            p = _auth_provider(cur, provider_code, key)
            if not p:
                return HTMLResponse(_portal_login('<div class="err">Wrong Provider ID or Key.</div>'))
            pc = (person_code or "").strip().upper()
            md = (mode or "").strip().lower()
            cur.execute("SELECT provider_code FROM persons WHERE person_code = %s;", (pc,))
            own = cur.fetchone()
            if not own or own[0] != p["provider_code"]:
                return HTMLResponse(_portal_login('<div class="err">Not found under your account.</div>'))
            if p["kind"] == "research":
                cur.execute("SELECT csv_text FROM research_uploads WHERE person_code=%s AND mode=%s;", (pc, md))
            else:
                purge_expired(cur)
                cur.execute("SELECT csv_text FROM clinical_holds WHERE person_code=%s AND mode=%s;", (pc, md))
            row = cur.fetchone()
    finally:
        conn.close()

    back = (f'<form class="inline" method="post" action="/portal/ui/person">'
            f'{_hidden(provider_code, key)}'
            f'<input type="hidden" name="person_code" value="{_esc(person_code)}">'
            f'<button class="backbtn" type="submit">&larr; Back</button></form>')

    if not row:
        table = '<p class="muted">No data for this mode (it may have expired).</p>'
    else:
        reader = csv.reader(io.StringIO(row[0]))
        all_rows = list(reader)
        if not all_rows:
            table = '<p class="muted">File is empty.</p>'
        else:
            header, body_rows = all_rows[0], all_rows[1:]
            thead = "<tr>" + "".join(f"<th>{_esc(h)}</th>" for h in header) + "</tr>"
            tbody = "".join("<tr>" + "".join(f'<td class="mono">{_esc(c)}</td>' for c in r) + "</tr>" for r in body_rows)
            table = (f'<p class="muted">{len(body_rows)} sessions &middot; {len(header)} metrics &middot; view only</p>'
                     f'<div class="tablewrap"><table><thead>{thead}</thead><tbody>{tbody}</tbody></table></div>')

    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(_mode_label(md))}</title>{_style()}</head><body>
  <h1>{_esc(_mode_label(md))} data</h1>
  <p class="sub"><span class="mono">{_esc(person_code)}</span></p>
  {back}
  <div class="card">{table}</div>
</body></html>"""
    return HTMLResponse(page)


# --------------------------------------------------------------------------
# Research analysis agent  (Stage 1: live agent pipe, no code execution yet)
# --------------------------------------------------------------------------
class AgentChatRequest(BaseModel):
    provider_code: str
    key: str
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    mode: str = ""            # group-comparison mode (sleep/rest/stand/breathwork)


# Rough budget for how much subject data to hand the agent as raw rows before
# falling back to computed summaries. ~4 chars/token; keep well under context.
DATA_CHAR_BUDGET = 220_000
# Per-recording metric master modes: bundled into an Individual analysis and
# offered for group comparison. One compact row per recording each.
RESEARCH_MODES = ["sleep", "rest", "stand", "breathwork", "circadian", "circadian_rhythm",
                  "circadian_episodes", "sleep_pwr", "sleep_pwr_episodes"]

# Waveform strip data (raw PPG for sinus-control + episode strips). NOT a metric
# master — it's long-format samples, pulled on demand when the researcher asks to
# see rhythm strips, so it never bloats a routine analysis.
STRIP_MODES = ["circadian_strips", "sleep_pwr_strips"]


def _fetch_csv(cur, person_code: str, mode: str) -> Optional[str]:
    cur.execute("SELECT csv_text FROM research_uploads WHERE person_code=%s AND mode=%s;",
                (person_code, mode))
    row = cur.fetchone()
    return row[0] if row else None


def _summarize_csv(csv_text: str) -> str:
    """Compact per-column summary (n, mean, sd, min, max) for numeric columns —
    the graceful fallback when raw rows exceed the budget."""
    import csv as _csv, io as _io, math
    reader = list(_csv.reader(_io.StringIO(csv_text)))
    if len(reader) < 2:
        return "(no data rows)"
    header, rows = reader[0], reader[1:]
    lines = [f"n_sessions={len(rows)}"]
    for ci, col in enumerate(header):
        vals = []
        for r in rows:
            if ci < len(r):
                try:
                    vals.append(float(r[ci]))
                except (ValueError, TypeError):
                    pass
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            lines.append(f"{col}: n={len(vals)} mean={mean:.3g} sd={sd:.3g} "
                         f"min={min(vals):.3g} max={max(vals):.3g}")
    return "\n".join(lines)


def _mode_guide(md: str) -> str:
    """One line telling the agent what a mode's columns mean."""
    if md == "sleep_pwr_strips":
        return ("Pulse Wave Rhythms strips. Columns: recording_ts, episode_id, strip_id, "
                "rel_ms, pwa, accel, tier, algo_version. Each recording contributes TWO "
                "continuous 10-minute strips: the DISTURBANCE strip, taken from one of the "
                "night's longest episodes, where episode_id is that episode's id (E1, E2, "
                "...); and the CONTROL strip, the quietest 10 minutes of the night, where "
                "episode_id is the literal string CONTROL. Split on episode_id and plot "
                "both on identical axes — the contrast is the point. rel_ms runs 0 to "
                "600000 within each strip. accel is exported as recorded and is already "
                "time-aligned to pwa — do not shift it. tier is beat quality (T1 is clean). "
                "Join to sleep_pwr_episodes for the disturbance strip's clock start and to "
                "sleep_pwr for the control's.")
    if md == "sleep_pwr":
        return ("Pulse Wave Rhythms summary: one row per recording. Columns: n_episodes, "
                "episode_min_total, night_duty_pct, control_start, control_duty_pct, "
                "control_move_sec, control_peak_accel, algo_version. DUTY is the percentage "
                "of time spent with pulse wave amplitude more than 70% below its running "
                "baseline; night_duty_pct is that figure for the whole recording and is the "
                "best night-to-night burden measure. The control_* columns describe the "
                "quietest 10 minutes of the night, which the control strip is drawn from. "
                "algo_version records the settings behind the row; rows with different "
                "algo_version values are not directly comparable.")
    if md == "sleep_pwr_episodes":
        return ("Pulse Wave Rhythms episode log: one row PER EPISODE, so a recording "
                "contributes several rows. Columns: recording_ts, episode_id, start, end, "
                "dur_min, duty_mean_pct, duty_max_pct, strip_start, strip_duty_pct, "
                "strip_move_sec, strip_peak_accel, algo_version. An episode is a contiguous "
                "stretch where duty stays at or above 1%, lasting at least 5 minutes; "
                "episodes never overlap and cannot run through movement. The strip_* "
                "columns are filled only for the one episode the disturbance strip came "
                "from and are BLANK for the rest. strip_move_sec is the number of seconds "
                "in that strip with accel above 3 and strip_peak_accel its highest accel; "
                "read them before calling a strip clean. Times are local with a UTC offset "
                "— keep them local. Nothing here counts discrete events: duty measures how "
                "much of the time amplitude is suppressed, which is why there is no drop "
                "count or depth. These index arousals, which is why they can exceed what "
                "oximetry would show; do not describe them as apnea, SDB or a breathing "
                "diagnosis, and do not treat duty as a severity measure.")
    return f"{md} session metrics: one row per recording."


def _selected_files(cur, groups: dict[str, list[str]], mode: str) -> list[dict[str, Any]]:
    """Which stored CSVs this request needs. Individual -> all metric modes plus
    its strip files; group comparison -> the chosen mode for each subject."""
    ind = groups.get("individual", []) or []
    g1 = groups.get("group1", []) or []
    g2 = groups.get("group2", []) or []
    mode = (mode or "").strip().lower()

    wanted: list[tuple[str, str]] = []   # (person_code, mode)
    for code in ind:
        wanted += [(code, md) for md in RESEARCH_MODES]
        wanted += [(code, sm) for sm in STRIP_MODES]
    if g1 or g2:
        if mode in RESEARCH_MODES:
            wanted += [(code, mode) for code in g1 + g2]

    def label(code: str) -> str:
        where = []
        if code in ind: where.append("Individual")
        if code in g1:  where.append("Group 1")
        if code in g2:  where.append("Group 2")
        return f"{code} [{', '.join(where)}]" if where else code

    out = []
    seen = set()
    for code, md in wanted:
        if (code, md) in seen:
            continue
        seen.add((code, md))
        cur.execute("SELECT id, csv_text, row_count, uploaded_at, anthropic_file_id, "
                    "file_uploaded_at FROM research_uploads WHERE person_code=%s AND mode=%s;",
                    (code, md))
        row = cur.fetchone()
        if not row or not row[1]:
            continue
        out.append(dict(row_id=row[0], code=code, mode=md, csv_text=row[1],
                        row_count=row[2], uploaded_at=row[3],
                        file_id=row[4], file_uploaded_at=row[5],
                        label=label(code),
                        filename=f"{code}_{md}.csv"))
    return out


def _ensure_files_uploaded(cur, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Push each CSV to the Anthropic Files API once and cache its id, so the
    agent's sandbox reads it from disk rather than having it inlined in the
    prompt. Re-uploads only when the stored CSV is newer than the cached file."""
    import io as _io
    ready = []
    for f in files:
        fid = f.get("file_id")
        stale = (f.get("file_uploaded_at") is None or f.get("uploaded_at") is None
                 or f["file_uploaded_at"] < f["uploaded_at"])
        if not fid or stale:
            try:
                up = client.beta.files.upload(
                    file=(f["filename"], _io.BytesIO(f["csv_text"].encode("utf-8")), "text/csv"))
                fid = getattr(up, "id", None)
                if fid:
                    cur.execute("UPDATE research_uploads SET anthropic_file_id=%s, "
                                "file_uploaded_at=now() WHERE id=%s;", (fid, f["row_id"]))
            except Exception:
                fid = None
        if fid:
            f["file_id"] = fid
            ready.append(f)
    return ready


def _files_manifest(files: list[dict[str, Any]]) -> str:
    """Human-readable listing of what's mounted in the sandbox."""
    if not files:
        return ""
    lines = []
    for f in files:
        rc = f"{f['row_count']} rows" if f.get("row_count") else "unknown size"
        lines.append(f"- {f['filename']}  ({f['label']} · {rc})\n    {_mode_guide(f['mode'])}")
    return "\n".join(lines)


def _research_agent_system(groups: dict[str, list[str]], mode: str,
                           manifest: str) -> str:
    ind = groups.get("individual", []) or []
    g1 = groups.get("group1", []) or []
    g2 = groups.get("group2", []) or []
    sel = []
    if ind: sel.append(f"Individual: {', '.join(ind)}")
    if g1:  sel.append(f"Group 1: {', '.join(g1)}")
    if g2:  sel.append(f"Group 2: {', '.join(g2)}")
    if (g1 or g2) and mode:
        sel.append(f"Group comparison mode: {mode}")
    sel_block = "\n".join(sel) if sel else "No subjects are selected yet."

    base = (
        "You are the Vagis research analysis assistant, helping a researcher analyze "
        "autonomic-metric data collected from study subjects via a smart ring. You speak "
        "to a professional researcher, so be technical and precise.\n\n"
        "The researcher has currently selected:\n" + sel_block + "\n\n"
        "The researcher is responsible for which subject codes belong to which group.\n\n"
    )

    if manifest:
        return base + (
            "YOU HAVE A PYTHON CODE-EXECUTION TOOL, and the selected subjects' CSV files are "
            "already mounted in your sandbox working directory. READ THEM FROM DISK with "
            "pandas (e.g. pd.read_csv('FILENAME.csv')) — never retype the data into your "
            "script, and never ask for it to be pasted.\n\n"
            "Files available:\n" + manifest + "\n\n"
            "SPEED IS CRITICAL — WORK IN ONE PASS:\n"
            "- Do the ENTIRE analysis in a SINGLE code execution: load the file(s), compute, "
            "and generate the figure(s) in one script. Do not run exploratory code first.\n"
            "- Do NOT narrate what you are about to do. Run the script, then report.\n"
            "- Make ONE focused figure unless the researcher asks for more.\n"
            "- Make reasonable assumptions inside the script and state them in your summary "
            "rather than stopping to ask.\n\n"
            "HOW TO REPORT:\n"
            "- The researcher wants RESULTS, not code. NEVER show, print, or describe the code "
            "you ran. Give only a clear, plain-language summary.\n"
            "- Report the real computed numbers: test used, n per group, means \u00b1 SD, the "
            "statistic, p-value, and an effect size where appropriate.\n"
            "- Save each figure as a PNG. Put KEY STATISTICS ON THE FIGURE where sensible "
            "(p-value, group means, error bars, n), with clear axis labels and a short title.\n"
            "- Do NOT fabricate. Only report what the computation produced. If the data can't "
            "support the requested test, say so plainly.\n"
        )
    return base + (
        "No subject data is loaded for this request (nothing selected, or the selected "
        "subjects have no uploaded data for the chosen mode). You can discuss study design "
        "and which tests would fit — but do NOT run code or fabricate numbers."
    )


CODE_EXEC_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}
AGENT_BETAS = ["code-execution-2025-08-25", "files-api-2025-04-14"]
AGENT_MAX_TOKENS = int(os.environ.get("VAGIS_AGENT_MAX_TOKENS", "16000"))


def _extract_text(content) -> str:
    return "".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


def _extract_figure_ids(content) -> list[str]:
    """Pull file_ids for any files the code execution created (e.g. saved PNGs)."""
    ids = []
    for b in content:
        if getattr(b, "type", None) == "bash_code_execution_tool_result":
            inner = getattr(b, "content", None)
            files = getattr(inner, "content", None) if inner is not None else None
            if files:
                for fb in files:
                    fid = getattr(fb, "file_id", None)
                    if fid:
                        ids.append(fid)
    return ids


@app.post("/portal/agent/chat")
def research_agent_chat(req: AgentChatRequest) -> dict[str, Any]:
    """Research analysis agent. Authenticated by provider_code+key. Stage 3: the
    agent runs real Python (code execution) on the selected subjects' data to
    compute prescribed statistics and generate figures. It reports plain-language
    results only (never code); figures are returned as downloadable images."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic key not configured.")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message.")

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            prov = authenticate_provider(cur, req.provider_code, req.key)
            if not prov or prov["kind"] != "research":
                raise HTTPException(status_code=401, detail="Not authorized for the research agent.")
            owned = _owned_selection(cur, prov["provider_code"], req.groups)
            wanted = _selected_files(cur, owned, req.mode)
            files = _ensure_files_uploaded(cur, wanted)
    finally:
        conn.close()

    manifest = _files_manifest(files)
    system = _research_agent_system(owned, req.mode, manifest)

    messages = []
    for turn in req.history[-20:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    if not messages:
        messages = [{"role": "user", "content": req.message}]

    # Mount the CSVs in the sandbox by attaching them to the latest user turn.
    # The agent reads them from disk instead of the data being inlined here.
    if files:
        blocks: list[dict[str, Any]] = []
        last = messages[-1]
        if last.get("role") == "user":
            text = last.get("content")
            if isinstance(text, str):
                blocks.append({"type": "text", "text": text})
            elif isinstance(text, list):
                blocks.extend(text)
        else:
            blocks.append({"type": "text", "text": req.message})
            messages.append({"role": "user", "content": blocks})
            last = messages[-1]
        for f in files:
            blocks.append({"type": "container_upload", "file_id": f["file_id"]})
        last["content"] = blocks

    figure_ids: list[str] = []
    container_id = None
    text_parts: list[str] = []
    CONTINUE_REASONS = {"pause_turn", "tool_use"}

    # Tool-use loop: code execution runs server-side on Anthropic's side. A turn
    # that is still working reports stop_reason "pause_turn" or "tool_use"; we feed
    # the assistant turn back and continue until it reaches a final stop reason and
    # produces its closing text. Each call carries a timeout so nothing hangs.
    hit_token_limit = False
    try:
        for _i in range(10):  # safety bound on continuations
            kwargs = dict(model=MODEL, max_tokens=AGENT_MAX_TOKENS,
                          system=system, messages=messages, tools=[CODE_EXEC_TOOL],
                          betas=AGENT_BETAS)
            if container_id:
                kwargs["container"] = container_id
            m = client.with_options(timeout=170.0).beta.messages.create(**kwargs)

            if getattr(m, "container", None):
                container_id = m.container.id
            figure_ids += _extract_figure_ids(m.content)
            t = _extract_text(m.content)
            if t:
                text_parts.append(t)

            if getattr(m, "stop_reason", None) == "max_tokens":
                hit_token_limit = True
                break
            if getattr(m, "stop_reason", None) in CONTINUE_REASONS:
                # Feed the assistant turn back verbatim so it can continue.
                messages.append({"role": "assistant", "content": m.content})
                continue
            break
    except anthropic.APITimeoutError:
        partial = "\n\n".join(p for p in text_parts if p).strip()
        note = ("The analysis is taking longer than expected and timed out. This can "
                "happen with larger multi-step analyses. Please try again, or ask for a "
                "more specific single test (e.g. name the exact metric and test).")
        reply = (partial + "\n\n" + note) if partial else note
        return {"reply": reply, "figures": figure_ids,
                "container": container_id, "has_analysis": bool(figure_ids), "timed_out": True}
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic error: {e.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {type(e).__name__}")

    reply = "\n\n".join(p for p in text_parts if p).strip()
    if not reply:
        if hit_token_limit:
            reply = ("The analysis was too large to complete in one response. Please ask for "
                     "a more focused analysis — for example, one specific metric and one test "
                     "at a time.")
        elif figure_ids:
            reply = ("The analysis ran but didn't return a written summary. "
                     "Please try again, or rephrase the request more specifically.")
        else:
            reply = "(no reply)"
    return {"reply": reply, "figures": figure_ids,
            "container": container_id, "has_analysis": bool(figure_ids)}


def _owned_selection(cur, provider_code: str, groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Filter the selected codes down to subjects that truly belong to this provider,
    so a tampered request can't pull another researcher's data."""
    cur.execute("SELECT person_code FROM persons WHERE provider_code=%s AND kind='research';",
                (provider_code,))
    mine = {r[0] for r in cur.fetchall()}
    out = {}
    for box in ("individual", "group1", "group2"):
        codes = [c.strip().upper() for c in (groups.get(box) or [])]
        out[box] = [c for c in codes if c in mine]
    return out


def _provider_ok(provider_code: str, key: str) -> bool:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            prov = authenticate_provider(cur, provider_code, key)
            return bool(prov and prov["kind"] == "research")
    finally:
        conn.close()


class FigureRequest(BaseModel):
    provider_code: str
    key: str
    file_id: str


@app.post("/portal/agent/figure")
def research_agent_figure(req: FigureRequest):
    """Stream a figure the code execution produced, by its Files API id.
    Authenticated by provider key so figures aren't world-readable."""
    from fastapi.responses import Response
    if not _provider_ok(req.provider_code, req.key):
        raise HTTPException(status_code=401, detail="Not authorized.")
    try:
        data = client.beta.files.download(req.file_id)
        raw = data.read() if hasattr(data, "read") else bytes(data)
    except Exception:
        raise HTTPException(status_code=404, detail="Figure not available.")
    return Response(content=raw, media_type="image/png")


class SummaryRequest(BaseModel):
    provider_code: str
    key: str
    title: str = "Vagis analysis summary"
    text: str = ""
    figure_ids: list[str] = Field(default_factory=list)


@app.post("/portal/agent/summary")
def research_agent_summary(req: SummaryRequest):
    """Assemble a PDF summary (plain-language results + figures) for download.
    Built server-side from the agent's reply text and the figures it generated."""
    from fastapi.responses import Response
    if not _provider_ok(req.provider_code, req.key):
        raise HTTPException(status_code=401, detail="Not authorized.")

    imgs = []
    for fid in req.figure_ids[:12]:
        try:
            d = client.beta.files.download(fid)
            imgs.append(d.read() if hasattr(d, "read") else bytes(d))
        except Exception:
            pass

    try:
        pdf_bytes = _build_summary_pdf(req.title, req.text, imgs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not build summary: {type(e).__name__}")

    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="vagis_analysis_summary.pdf"'})


def _build_summary_pdf(title: str, text: str, images: list[bytes]) -> bytes:
    """Compose a simple PDF: title, plain-language results, then figures."""
    import io as _io
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as _canvas
    from datetime import datetime as _dt

    buf = _io.BytesIO()
    c = _canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    margin = 0.9 * inch
    y = H - margin

    c.setFont("Helvetica-Bold", 15)
    c.drawString(margin, y, title[:90]); y -= 20
    c.setFont("Helvetica", 9)
    c.setFillGray(0.4)
    c.drawString(margin, y, "Generated by Vagis · " + _dt.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    c.setFillGray(0); y -= 22

    c.setFont("Helvetica", 10.5)
    max_w = W - 2 * margin
    for para in (text or "").split("\n"):
        para = para.replace("**", "").rstrip()
        if not para:
            y -= 6; continue
        words = para.split(" ")
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, "Helvetica", 10.5) > max_w:
                c.drawString(margin, y, line); y -= 14; line = w
                if y < margin + 40:
                    c.showPage(); y = H - margin; c.setFont("Helvetica", 10.5)
            else:
                line = test
        if line:
            c.drawString(margin, y, line); y -= 14
            if y < margin + 40:
                c.showPage(); y = H - margin; c.setFont("Helvetica", 10.5)

    for img in images:
        try:
            ir = ImageReader(_io.BytesIO(img))
            iw, ih = ir.getSize()
            disp_w = max_w
            disp_h = disp_w * ih / iw
            if disp_h > H - 2 * margin:
                disp_h = H - 2 * margin
                disp_w = disp_h * iw / ih
            if y - disp_h < margin:
                c.showPage(); y = H - margin
            c.drawImage(ir, margin, y - disp_h, width=disp_w, height=disp_h,
                        preserveAspectRatio=True, mask="auto")
            y -= disp_h + 18
        except Exception:
            pass

    c.showPage(); c.save()
    return buf.getvalue()


# ==========================================================================
# AI CONNECT — the Vagis connector for Claude, ChatGPT and other chat agents
# ==========================================================================
# A Model Context Protocol (MCP) server over plain HTTP, built into this app.
#
# HOW A USER CONNECTS (testing stage)
#   1. Admin page -> "AI Connect key": enter the person's SE code. The server
#      makes a private connector address:
#          https://vagis-server.onrender.com/mcp/<private key>
#   2. In Claude: Customize > Connectors > + > Add custom connector, paste
#      that address, name it Vagis.
#   3. The agent can now read that person's Session History and the guide.
#
# The private key in the address is the login for now. Anyone with the
# address can read that person's metrics, so it is treated like a password.
# Making a new key for the same person retires the old one. A proper login
# (OAuth) replaces this before public release.
#
# WHAT THE AGENT CAN DO (tools)
#   get_vagis_guide      the overview, rules and mode guides (GUIDE_TEXT below)
#   list_my_data         which modes have Session History on the server
#   get_session_history  one mode's Session History (CSV, one row per recording)
#   save_note / get_notes  short notes carried between conversations
#
# Only Session History metrics (what the app uploads via Data Share) are ever
# served. No raw signal exists on the server.
#
# ADAPTS TO APP CHANGES. Session History is served exactly as uploaded, with
# whatever columns it has, and any mode name the app sends is listed. New or
# renamed metrics and new modes need no change here — only the guide text.
# --------------------------------------------------------------------------
import json as _json
from pathlib import Path as _Path

from fastapi import Request
from fastapi.responses import JSONResponse, Response

GUIDE_DIR = _Path(__file__).parent / "guide"
GUIDE_SECTIONS = {
    "overview":         "00_overview_and_rules",
    "sleep":            "sleep",
    "stand":            "stand",
    "exertion":         "exertion",
    "load":             "load",
    "breathwork":       "breathwork",
    "quick_check":      "quick_check",
    "load_and_reserve": "load_and_reserve",
    "background_data":  "background_data",
}
MCP_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
AI_NOTE_MAX_CHARS = 4000
AI_NOTES_RETURNED = 30

CREATE_AI_KEYS_SQL = """
CREATE TABLE IF NOT EXISTS ai_connect_keys (
    key          TEXT PRIMARY KEY,
    person_code  TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked      BOOLEAN NOT NULL DEFAULT FALSE
);
"""

CREATE_AI_NOTES_SQL = """
CREATE TABLE IF NOT EXISTS ai_notes (
    id           SERIAL PRIMARY KEY,
    person_code  TEXT NOT NULL,
    note         TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _ai_ensure_tables(cur) -> None:
    ensure_tables(cur)
    cur.execute(CREATE_AI_KEYS_SQL)
    cur.execute(CREATE_AI_NOTES_SQL)


# ---- Keys ----------------------------------------------------------------
def _ai_issue_key(cur, person_code: str) -> str:
    """New private key for a person; any earlier key stops working."""
    cur.execute("UPDATE ai_connect_keys SET revoked = TRUE WHERE person_code = %s;",
                (person_code,))
    key = secrets.token_urlsafe(24)
    cur.execute("INSERT INTO ai_connect_keys (key, person_code) VALUES (%s, %s);",
                (key, person_code))
    return key


def _ai_person_for_key(key: str) -> Optional[str]:
    if not key or not DATABASE_URL:
        return None
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _ai_ensure_tables(cur)
            cur.execute("SELECT person_code FROM ai_connect_keys "
                        "WHERE key = %s AND NOT revoked;", (key,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


# ---- Tools ---------------------------------------------------------------
# The Vagis guide. Built in here so main.py is the only file to update.
# A guide/<name>.md file next to main.py, if present, takes priority.
GUIDE_TEXT: dict[str, str] = {
    "overview": r"""# Vagis Guide — Overview and Rules

## What Vagis is
Vagis Health is an iPhone app that works with the 2301B smart ring. The ring measures the pulse at the finger with light (PPG) and records movement with an accelerometer. The app turns those signals into metrics about heart rate, heart rate variability, the pulse wave and movement.

The app has three homepage sections:
- **Sessions** — recording modes: Sleep, Stand, Exertion, Load, Breathwork, Quick Check.
- **Analysis** — views that combine modes: Load & Reserve, Data Share, AI Connect.
- **Background** — information the user adds: Genomics, Questionnaires, Calendar.

Each recording mode has its own guide file. Every mode keeps a **Session History**: one row of metrics per recording, newest first. Session History is the only recording data you receive. Raw signals never leave the phone.

## Who uses Vagis
Many users live with ME/CFS, Long COVID, hEDS or other conditions that affect the autonomic nervous system. A recording may be a hard effort or a housebound person doing very little; both are equally valid.

## Rules for answering
1. **Research use only.** Vagis metrics are not diagnostic measurements. Never diagnose, and never state that a user has or does not have a condition. Suggest discussing notable findings with their clinician.
2. **Label by what was measured.** Use the metric names exactly as the app shows them. Do not add severity words (mild, moderate, severe, abnormal, normal) and do not convert Vagis metrics into clinical scores such as AHI, RDI or blood pressure.
3. **Compare the user with themselves.** The most useful comparison is against the user's own earlier recordings of the same mode, taken the same way. Avoid population "normal ranges" unless the user asks, and say they are general.
4. **Do not explain how metrics are calculated.** Describe what a metric means and what it reflects, never its formula, thresholds or algorithm. Some methods are protected (for example Vascular Tone). If asked, say the method is proprietary.
5. **Do not grade effort or recovery.** Never praise higher numbers or treat lower ones as underperformance, and do not predict crashes, flares or post-exertional malaise.
6. **Frequency language.** Describe rhythms by frequency in Hz or by the named band (for example "Neurogenic band, 0.021–0.052 Hz"), not as cycle length in seconds.
7. **Be plain and specific.** Use the user's actual numbers and dates. Keep answers short unless more detail is asked for.

## Data you may receive
- Session History for each recording mode.
- Saved genomic variants (selected variants only; the full genome file never leaves the phone).
- Questionnaire answers and calendar notes.
- Lab and test results the user chose to save (values and dates only).
- Notes saved from earlier conversations.
""",
    "sleep": r"""# Vagis Guide — Sleep

## How to use it
1. Open Sleep and tap Start before bed, with the ring on and connected.
2. Sleep normally. The phone can stay nearby; the recording resumes on its own if the connection drops.
3. Tap Stop in the morning. Analysis needs at least 2 hours of recording and takes 1–2 minutes.

Sleep uses the ring's red/infrared light. Stopping a recording analyses that night only; earlier nights are read from Session History.

## Tabs and graphs
- **Stages** — a hypnogram of Awake, REM, Light and Deep sleep across the night, with a motion strip underneath. Stages come from heart rate, pulse and movement, not brain waves, so they are an estimate.
- **Heart Rate** — heart rate across the night with movement underneath.
- **Cycling** — "Pulse Wave Cycling": overlapping curves showing how many cycling episodes occurred in each 5 minutes of the night. Blue is PWA Cycling, dark red is PWA-HR Cycling.
- **Deep** — two traces from a 5-minute stretch of Deep sleep: the time between beats (RR interval, blue) above the pulse wave amplitude (PWA, orange).
- **Bands** — a bar for each vascular rhythm band, measured on a 30-minute stretch of non-REM sleep (at least 20 minutes of it light, no REM or waking). Each bar is how much pulse size swings in that band, as a percentage of its own mean. Bands: Endothelial 0.005–0.0095 Hz, Endothelial-NO 0.0095–0.021 Hz, Neurogenic 0.021–0.052 Hz, Myogenic 0.052–0.145 Hz.
- **Export** — CSV files for each recording and the cumulative metrics file.

## Key terms
- **PWA (pulse wave amplitude)** — the size of each pulse at the finger. It shrinks when finger vessels tighten and grows when they relax.
- **Deep sleep as the reference** — Deep sleep is the calmest part of the night, so several metrics use the user's own Deep sleep as their personal baseline.

## Session History metrics
### Stages
- **Sleep Time** (hours) — time asleep.
- **Deep %** and **REM %** — share of the recording in Deep and REM sleep.
- **Deep Sleep Time** and **REM Sleep Time** (min).
- **Autonomic Disturbance Index (ADI)** (%) — how much of the night's heart rate variability sits in the slowest rhythms. Higher means more slow autonomic disturbance across the night.
- **Breathing Rate** (breaths/min) — average overnight breathing rate.
- **Δ Temp** (°C) — change in skin temperature measured by the ring. Ring temperature is noisy, so treat small changes with caution.
- **Motion per Hour** — movements per hour of recording.

### Heart Rate
- **Average Heart Rate** (bpm).
- **Sustained Nadir** and **Sustained Peak** (bpm) — the lowest and highest heart rate held for a sustained stretch, not single beats.
- **SDNN** (ms) — overall heart rate variability across the night.
- **RMSSD** (ms) — beat-to-beat variability, mainly reflecting vagal (rest-and-digest) activity.
- **Overnight HR Dip** (%) — how far heart rate falls during the night.

### Cycling
Slow, repeating rises and falls in pulse size during sleep fall into three groups:
- **Vaso-cycling** — cycling within the depth the user shows in their own Deep sleep; their own normal vessel rhythm.
- **PWA Cycling** — cycling deeper than the user's own Deep sleep range, without a heart-rate surge.
- **PWA-HR Cycling** — cycling with a heart-rate surge at the same time.

Each group is reported per hour of sleep and as total minutes. **Total Obstruction (PWA + PWA-HR)** is PWA Cycling plus PWA-HR Cycling per hour of sleep.

These are measured pulse-wave patterns. Breathing disruption can occur without a fall in oxygen, so these patterns are not expected to match oximetry. Do not call them apneas or convert them to AHI.

### Deep
Measured on the user's Deep sleep:
- **Deep HR Swing** (bpm) — typical size of heart-rate swings in Deep sleep; the user's calm-state reactivity.
- **Deep SDNN** and **Deep RMSSD** (ms) — heart rate variability in Deep sleep.
- **Pulse Wave Variability** (%) — how much pulse size varies in Deep sleep.
- **Vasomotor Power** — strength of slow vessel-tone rhythms in pulse size during Deep sleep.
- **Respiratory DC Modulation** (% of DC) — how much the finger's blood volume level moves with each breath.
- **Perfusion Index** (% of DC) — how strongly blood pulses into the finger relative to its steady level.

## Notes for interpretation
- The app does not measure lymphatic or glymphatic clearance; do not claim it does.
- Compare nights against the user's own earlier nights.
""",
    "stand": r"""# Vagis Guide — Stand

## How to use it
Choose a test mode first: **Short** or **Long**.
1. **Lie down, 90 s.** Stay still and breathe normally. The ring settles over the first 20 s.
2. **Stand on the cue.** A spoken countdown plays; stand and stay still with your arm at your side.
3. **Stand.** Short test: 75 s, and the test ends there. Long test: 2.5 min. Standing still matters more than standing straight.
4. **Long test only — lie down on the cue.** Stay down for the last 2.5 min so recovery is recorded.

Phases: **Supine 1** (lying before standing), **Standing**, and **Supine 2** (lying after, long test only). Every heart rate value comes from the ring's own heart rate.

## Tabs and graphs
- **HR** — heart rate through the test. "HR Rise While Standing" shows the rise above lying heart rate with a +30 bpm reference line used in research criteria. Research only, not a diagnosis.
- **Pulse** — "Pulse Amplitude & Venous Pooling": two stacked dot panels sharing the phase axis, pulse amplitude above venous pooling, with the change shown between phases.
- **Waveform** — "Pulse Shape": the average pulse shape for each phase, in ms from the pulse foot, with Time to Peak tiles per phase.
- **Export** — CSV files and the metrics master.

## Key terms
- **Pulse Strength / Pulse Amplitude** — height of each pulse, adjusted for light reaching the sensor. It usually falls on standing as finger vessels tighten.
- **Rise Timing** — share of each beat spent rising to the pulse peak.
- **Blood Pooling / Venous Pooling** — blood sitting in the finger, with lying down = 100.

## Session History metrics
### Heart Rate
- **Supine Heart Rate** (bpm) — lying heart rate before standing.
- **Peak Standing Heart Rate** (bpm).
- **Change on Standing** (bpm) — standing heart rate minus lying heart rate.
- **Rise at 1 Minute** and **Rise at 2 Minutes** (bpm) — heart rate rise above lying at those times.
- **Heart Rate After Lying Back Down** (bpm) — long test.
- **Recovery Dip** (bpm) — how far heart rate dips after lying back down (long test).

### Pulse
- **Pulse Amplitude — Supine 1 / Standing / Supine 2** (counts).
- **Pulse Amplitude — Supine 1 to Standing** and **— Standing to Supine 2** (% of supine) — change between phases.
- **Venous Pooling — Supine 1 to Standing** and **— Standing to Supine 2** (index points) — change in blood sitting in the finger.

### Waveform
- **Rise Timing — Supine 1 / Standing / Supine 2** (% of beat), and the change between phases (points).
- **Time to Peak — Supine 1 / Standing / Supine 2** (ms) — time from the start of the pulse to its peak.

## Notes for interpretation
- Short and long tests are only comparable with tests of the same mode.
- Do not diagnose POTS or any condition; describe the measured rise and suggest the user discuss it with their clinician if it concerns them.
""",
    "exertion": r"""# Vagis Guide — Exertion

## How to use it
1. Enter age, sex and weight once. They stay on the phone and are never uploaded. Sex only selects the right VO2max equation.
2. Tap Start. The ring stores heart rate on its own, so the phone is not needed during the recording.
3. Go about the activity or the day. Record at least 30 minutes; under an hour, peak values may not represent the day.
4. Tap Finish to download the heart rate from the ring and analyse it.

Resting heart rate comes from the user's Quick Check if they did one in the last 14 days. Otherwise it is estimated from the recording itself, and the app shows a note saying so.

Exertion is a **monitor**, not a fitness test and not a diagnostic. It describes what the heart did.

## Tabs and graphs
- **HR** — heart rate across the recording with the resting level marked, plus time spent in heart-rate ranges.
- **Energy** — estimated METs and energy use across the recording.
- **Export** — the metrics workbook (one row per session) and the 5-second heart rate samples.

## Session History metrics
### Heart rate
- **Time** (min) — recording length.
- **Resting Heart Rate** (bpm) — from Quick Check, or estimated from the recording.
- **Average Heart Rate** and **Median Heart Rate** (bpm).
- **Peak Heart Rate** (bpm) — highest heart rate in the recording.
- **Heart Rate Range** (bpm).
- **Time Above Rest** (%) — share of the recording above resting heart rate.

### Demand
- **Maximum Heart Rate** (bpm) — the highest heart rate actually recorded.
- **Heart Rate Reserve** (bpm) — maximum minus resting heart rate.
- **Resting % of Max**, **Mean % of Max**, **Peak % of Max** (%) — heart rate as a share of maximum.
- **VO2max** (mL/kg/min) — an estimate of aerobic capacity from maximum and resting heart rate. It updates as those values change.

### Energy
- **Mean METs** and **Peak METs** — estimated energy cost relative to rest.
- **Energy Rate** (kcal/h), **Energy Above Rest** (kcal/h), **Total Energy** (kcal).

## Notes for interpretation
- METs, calories and VO2max are estimates that assume typical physiology. In people whose heart rate does not rise normally with activity, they can be off.
- The ring's stored heart rate is smoothed, so brief true peaks may read lower.
- Peak heart rate and total energy grow with recording length; average, median and per-hour values compare better between recordings.
- Never grade a session or predict post-exertional malaise. Compare against the user's own earlier recordings.
""",
    "load": r"""# Vagis Guide — Load

## How to use it
1. **Start and carry on.** No protocol and no posture to hold; the point is an ordinary stretch of the day.
2. **Wear it for a few hours.** Longer recordings hold more still, clean stretches.
3. **Move normally.** Everything is measured against movement, so a recording with no activity has nothing to measure from.
4. **Stop when done.** Everything is computed on Stop.

Enter age once; it sets the heart rate ceiling. Load uses the ring's own heart rate and the accelerometer only.

Purpose: tracking how heart rate responds to everyday movement over long periods, to help spot changes and possible triggers.

## Motion bands
- **Still** — under 0.02 g
- **Light** — 0.02–0.07 g
- **Moderate** — 0.07–0.15 g
- **High** — over 0.15 g

## Tabs and graphs
- **Timeline** — "Heart Rate & Motion" across the whole recording, with an orange line marking the heart rate ceiling (age-predicted).
- **Motion** — "Time by motion": time spent in each motion band.
- **Response** — "Heart rate by motion": median heart rate in each motion band.
- **Export** — CSV files and the metrics master.

## Session History metrics
### Timeline
- **Recording Length** (min).
- **Lowest Heart Rate**, **Average Heart Rate**, **Maximum Heart Rate** (bpm).
- **Heart Rate Ceiling** (bpm) — an age-predicted level used as a pacing reference.
- **Time Above Ceiling** — time spent above the ceiling.

### Motion
- **Share of Recording Still / in Light / in Moderate / in High Motion** (%).
- **Longest Still Stretch** (min), **Moving Bouts per Hour**, **Longest Moving Bout** (min).
- **Average Motion** (g).

### Response
- **HR While Still / in Light / in Moderate / in High Motion** (median bpm).
- **Still to High HR Difference** (bpm) — heart rate in high motion minus heart rate while still.
- **Heart Rate per Motion** — how much heart rate rises for a given amount of movement, scaled to the user's own still heart rate in that recording.
- **Heart Rate Response Delay** (s) — how long heart rate takes to follow movement.
- **Heart Rate Range While Still** (bpm).

## Notes for interpretation
- The ceiling is a reference line, not a limit or a diagnosis.
- Recordings differ in what the user did, so compare like with like and look at trends over many recordings.
""",
    "breathwork": r"""# Vagis Guide — Breathwork

## How to use it
1. Choose a pace in breaths per minute (5.0, 5.5, 6.0, 6.5, 7.0, 8, 10, 12, 14, 16, 18 or 20), Box breathing, or Free breathing.
2. Tap Start, stay still, and follow the guide. Movement affects signal quality.
3. Sessions must last at least 2 minutes for results.

An option saves the full raw pulse trace for later analysis.

## Graphs
- **HRV Spectrum** — how heart rate variability is spread across frequencies; paced breathing produces a peak at the breathing frequency.
- **Heart Rate** — heart rate through the session, rising and falling with each breath.
- **RSA** — the swing in heart rate with each breath.
- **Accel** — movement, to check the session was still.

## Session History metrics
- **Pace / mode** — the pace or mode used.
- **Duration** (min).
- **Average Heart Rate** (bpm).
- **ln LF** — strength of heart rate variability in the low-frequency range (0.04–0.15 Hz), on a log scale. Slow paced breathing increases it.
- **RSA Amplitude** (ms) — how much the time between beats swings with each breath; a marker of vagal response.
- **Coherence** (0–1) — how closely heart rate and pulse size move together at the breathing frequency. Higher means they are more tightly linked.
- **Pulse Wave Variability** — how much pulse size varies during the session.

## Notes for interpretation
- Sessions are most comparable at the same pace.
""",
    "quick_check": r"""# Vagis Guide — Quick Check

## How to use it
1. **Sit comfortably** — back supported, feet flat, legs uncrossed.
2. **Rest the ring hand** lightly on the upper chest, just below the collarbone. Don't press.
3. **Settle for 30 seconds** during the countdown.
4. **Keep still.** The check needs 30 seconds of stillness. If the hand moves, a voice says "Please keep still" and the 30 seconds restart. If a steady reading isn't possible within 3 minutes, the check stops and asks to try again.
5. **Listen for the tone** that marks the end.

Optionally, enter a cuff blood pressure reading taken with the check; it is saved with that check's metrics.

## Tabs and graphs
- **Check** — the guided check.
- **Results** — the 30-second finger pulse trace, the result tiles and the cuff entry box.
- **Export** — CSV files and the metrics master.

## Session History metrics
- **Heart Rate** (bpm).
- **HRV** (ms) — beat-to-beat heart rate variability (RMSSD).
- **Breathing Rate** (breaths/min).
- **Vascular Tone** — a Vagis index of peripheral vessel tone, measured from the pulse at the finger. It has no unit. Higher means more relaxed vessels; lower means more constricted.
- **Cuff Systolic** and **Cuff Diastolic** (mmHg) — entered by the user, if any.

## Notes for interpretation
- **Vascular Tone is not blood pressure.** Never convert it to mmHg. It often moves with blood pressure, but they are different measures and can diverge.
- **The Vascular Tone method is proprietary.** Do not describe, guess or speculate how it is calculated; say only that it comes from the pulse measured by the ring.
- Cuff readings are entered by the user and may contain errors.
- Checks are comparable only when taken the same way (seated, hand on chest). Resting heart rate from Quick Check is also used by Exertion.
""",
    "load_and_reserve": r"""# Vagis Guide — Load & Reserve (Analysis)

## What it shows
A cross-mode view plotting each day's **Load** (what the day spent: daytime activity and exertion) against its **Reserve** (recovery capacity from sleep, breathwork, stand and resting measures). Each sits on a 0–100 scale relative to the user's own history.

The screen shows the graph of days, how many reserve and load elements had data, and a list of the elements with their current band. The more modes the user records, the more complete it gets.

## Notes for interpretation
- Use only the elements and values present. If an element has no data, say so.
- Do not invent a single composite score.
- Describe whether reserve is keeping pace with load and which way recent days have moved, without grading.
""",
    "background_data": r"""# Vagis Guide — Background Data

## Genomics
The user loads their own genome file (VCF) into the app. The file never leaves the phone. The user searches for genes or variants (for example COMT, or rs4680) and saves the ones they want; only saved variants are shared.

Each saved variant includes the rsID, gene, the user's genotype, and an optional note on why it matters to them. Saved variants can be grouped.

Notes: single variants usually have small effects. Explain what a variant is generally associated with, without predicting disease or giving risk numbers, and suggest a genetic counsellor or clinician for health decisions.

## Questionnaires
- **DSQ-SF** — the DePaul Symptom Questionnaire short form, used in ME/CFS.
- **Daily VAS** — five 0–10 sliders: fatigue, PEM, brain fog, pain and sleep disturbance. Higher means worse for all five.

Each submission is dated, so symptoms can be compared with ring metrics from the same days.

## Calendar
Day notes the user writes (History & Notes), plus which modes were recorded on each day. Notes are the user's own words and can give context for changes in the metrics.

## Saved results
Lab or test results the user chose to save from a chat: test name, value, unit, reference range and date only. No documents are stored.
""",
}


def _guide_text(section: str) -> Optional[str]:
    name = GUIDE_SECTIONS.get(section)
    if not name:
        return None
    path = GUIDE_DIR / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return GUIDE_TEXT.get(section)


def _tool_get_vagis_guide(person: str, args: dict) -> str:
    section = (args.get("section") or "").strip().lower()
    if section and section not in GUIDE_SECTIONS:
        return ("Unknown section. Choose one of: " + ", ".join(GUIDE_SECTIONS) + ".")
    wanted = ["overview"] if not section else (
        [section] if section == "overview" else ["overview", section])
    parts = [t for t in (_guide_text(s) for s in wanted) if t]
    if not parts:
        return "The Vagis guide is not available on the server yet."
    if not section:
        parts.append("Other guide sections: " +
                     ", ".join(s for s in GUIDE_SECTIONS if s != "overview") +
                     ". Call get_vagis_guide with a section for a mode's guide.")
    return "\n\n----------------------------------------\n\n".join(parts)


def _tool_list_my_data(person: str, args: dict) -> str:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _ai_ensure_tables(cur)
            cur.execute("SELECT mode, row_count, uploaded_at, csv_text FROM research_uploads "
                        "WHERE person_code = %s ORDER BY mode;", (person,))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return ("No Session History on the server yet. In the Vagis app, open "
                "Analysis > Data Share and tap Send my data.")
    lines = ["Session History on the server (one row per recording):"]
    for mode, n, up, text in rows:
        header = next(csv.reader(io.StringIO(text)), [])
        lines.append(f"- {mode}: {n} recording(s), last sent "
                     f"{up.strftime('%Y-%m-%d %H:%M UTC') if up else 'unknown'}; "
                     f"columns: {', '.join(header)}")
    return "\n".join(lines)


def _tool_get_session_history(person: str, args: dict) -> str:
    mode = (args.get("mode") or "").strip().lower()
    if not mode:
        return "Give a mode, e.g. sleep, stand, exertion, load, breathwork or quick_check."
    last_n = args.get("last_n")
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _ai_ensure_tables(cur)
            cur.execute("SELECT csv_text, uploaded_at FROM research_uploads "
                        "WHERE person_code = %s AND mode = %s;", (person, mode))
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return (f"No {mode} Session History on the server. Call list_my_data to see "
                "which modes are available.")
    text, up = row
    lines = text.strip().splitlines()
    if isinstance(last_n, int) and last_n > 0 and len(lines) > last_n + 1:
        lines = [lines[0]] + lines[-last_n:]
    sent = up.strftime("%Y-%m-%d %H:%M UTC") if up else "unknown"
    return (f"{mode} Session History (CSV, one row per recording; last sent from the "
            f"app {sent}). Column names are the app's file headers; the guide gives "
            f"the names the user sees.\n\n" + "\n".join(lines))


def _tool_save_note(person: str, args: dict) -> str:
    note = (args.get("note") or "").strip()
    if not note:
        return "Nothing to save — the note was empty."
    note = note[:AI_NOTE_MAX_CHARS]
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _ai_ensure_tables(cur)
            cur.execute("INSERT INTO ai_notes (person_code, note) VALUES (%s, %s);",
                        (person, note))
    finally:
        conn.close()
    return "Note saved. It will be available in future conversations."


def _tool_get_notes(person: str, args: dict) -> str:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _ai_ensure_tables(cur)
            cur.execute("SELECT note, created_at FROM ai_notes WHERE person_code = %s "
                        "ORDER BY created_at DESC LIMIT %s;", (person, AI_NOTES_RETURNED))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return "No notes saved from earlier conversations yet."
    return "Notes from earlier conversations, newest first:\n\n" + "\n\n".join(
        f"[{c.strftime('%Y-%m-%d')}] {n}" for n, c in rows)


AI_TOOLS: dict[str, dict[str, Any]] = {
    "get_vagis_guide": {
        "fn": _tool_get_vagis_guide,
        "description": ("The Vagis guide: what the app is, the rules for answering, and "
                        "for each mode how it is used, what its graphs show and what each "
                        "metric means. Call with no section first for the overview and "
                        "rules, then with a mode's section before discussing that mode."),
        "schema": {"type": "object", "properties": {
            "section": {"type": "string", "enum": list(GUIDE_SECTIONS),
                        "description": "Guide section. Omit for the overview and rules."}},
            "additionalProperties": False},
    },
    "list_my_data": {
        "fn": _tool_list_my_data,
        "description": ("Which Vagis modes have Session History on the server, how many "
                        "recordings each holds, when they were last sent, and the columns."),
        "schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "get_session_history": {
        "fn": _tool_get_session_history,
        "description": ("One mode's Session History as CSV: one row per recording with that "
                        "recording's metrics. Use it to answer questions, compare recordings, "
                        "graph trends or run statistics."),
        "schema": {"type": "object", "properties": {
            "mode": {"type": "string",
                     "description": "Mode name as listed by list_my_data, e.g. sleep, stand, "
                                    "exertion, load, breathwork, quick_check."},
            "last_n": {"type": "integer", "minimum": 1,
                       "description": "Optional: only the most recent N recordings."}},
            "required": ["mode"], "additionalProperties": False},
    },
    "save_note": {
        "fn": _tool_save_note,
        "description": ("Save a short note about this conversation (what was looked at, "
                        "what the user wants to follow) so future conversations can pick up "
                        "where this one left off. Ask the user before saving. Metric "
                        "observations only — no names or identifying details."),
        "schema": {"type": "object", "properties": {
            "note": {"type": "string", "description": "The note, a few sentences."}},
            "required": ["note"], "additionalProperties": False},
    },
    "get_notes": {
        "fn": _tool_get_notes,
        "description": "Notes saved from earlier conversations with this user, newest first.",
        "schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}

AI_INSTRUCTIONS = (
    "Vagis connector: the user's own Vagis smart-ring metrics. At the start of a "
    "conversation call get_vagis_guide (overview and rules) and get_notes. Before "
    "discussing a mode, read its guide section. Follow the guide's rules: research "
    "use only, no diagnosis, label metrics by what was measured, compare the user "
    "with their own history, and never describe how metrics are calculated. Use "
    "get_session_history for the data. Offer to save a short note at the end of a "
    "useful conversation."
)


# ---- MCP over HTTP (JSON-RPC, stateless, JSON responses) -----------------
def _rpc_result(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _rpc_error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _mcp_handle(person: str, msg: Any) -> Optional[dict]:
    if not isinstance(msg, dict):
        return _rpc_error(None, -32600, "Invalid request")
    method = msg.get("method")
    msg_id = msg.get("id")
    if msg_id is None:            # a notification: no reply
        return None
    params = msg.get("params") or {}

    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in MCP_PROTOCOL_VERSIONS else MCP_PROTOCOL_VERSIONS[0]
        return _rpc_result(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "vagis", "title": "Vagis", "version": "1.0.0"},
            "instructions": AI_INSTRUCTIONS,
        })
    if method == "ping":
        return _rpc_result(msg_id, {})
    if method == "tools/list":
        return _rpc_result(msg_id, {"tools": [
            {"name": n, "description": t["description"], "inputSchema": t["schema"]}
            for n, t in AI_TOOLS.items()]})
    if method == "tools/call":
        name = params.get("name")
        tool = AI_TOOLS.get(name)
        if not tool:
            return _rpc_error(msg_id, -32602, f"Unknown tool: {name}")
        args = params.get("arguments") or {}
        try:
            text = tool["fn"](person, args if isinstance(args, dict) else {})
            return _rpc_result(msg_id, {"content": [{"type": "text", "text": text}],
                                        "isError": False})
        except Exception as e:
            print(f"[ai-connect] tool {name} failed: {type(e).__name__}: {e}")
            return _rpc_result(msg_id, {"content": [{"type": "text",
                               "text": "The Vagis server could not complete that request."}],
                               "isError": True})
    return _rpc_error(msg_id, -32601, f"Method not found: {method}")


@app.post("/mcp/{key}")
async def mcp_endpoint(key: str, request: Request):
    person = await asyncio.to_thread(_ai_person_for_key, key)
    if not person:
        return JSONResponse(status_code=404,
                            content={"error": "Unknown or retired Vagis connector address."})
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content=_rpc_error(None, -32700, "Parse error"))

    if isinstance(body, list):
        replies = [r for r in [await asyncio.to_thread(_mcp_handle, person, m) for m in body] if r]
        return JSONResponse(replies) if replies else Response(status_code=202)
    reply = await asyncio.to_thread(_mcp_handle, person, body)
    return JSONResponse(reply) if reply else Response(status_code=202)


@app.get("/mcp/{key}")
def mcp_get(key: str):
    # No server-initiated stream; clients fall back to plain POST replies.
    return Response(status_code=405, headers={"Allow": "POST"})


@app.delete("/mcp/{key}")
def mcp_delete(key: str):
    return Response(status_code=405, headers={"Allow": "POST"})


# ---- Admin: make a connector address ------------------------------------
@app.post("/admin/ui/aiconnect", response_class=HTMLResponse)
def admin_ui_aiconnect(request: Request, token: str = Form(""),
                       person_code: str = Form("")) -> HTMLResponse:
    if token.strip() != (VAGIS_ADMIN_TOKEN or "").strip() or not VAGIS_ADMIN_TOKEN:
        return HTMLResponse(_admin_page(token, '<div class="err">Admin token did not match.</div>'))
    parsed = parse_person_code(person_code)
    if not parsed or parsed["kind"] != "research":
        return HTMLResponse(_admin_page(token,
            '<div class="err">Enter a valid SE code (AI Connect uses SE codes for now).</div>'))
    code = parsed["person_code"]
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _ai_ensure_tables(cur)
            if not person_exists(cur, code, "research"):
                return HTMLResponse(_admin_page(token,
                    '<div class="err">That SE code has not been issued.</div>'))
            key = _ai_issue_key(cur, code)
    finally:
        conn.close()
    base = str(request.base_url).rstrip("/").replace("http://", "https://")
    url = f"{base}/mcp/{key}"
    banner = (
        '<div class="result">'
        f'<div class="row"><span class="k">Person</span><span class="v">{_esc(code)}</span></div>'
        f'<div class="row"><span class="k">Connector address</span>'
        f'<span class="v" style="word-break:break-all">{_esc(url)}</span></div>'
        '<div class="warn">Paste this into Claude: Customize &gt; Connectors &gt; + &gt; '
        'Add custom connector. Keep it private &mdash; anyone with it can read this '
        "person's metrics. Making a new one retires this one. Not shown again.</div>"
        '</div>'
    )
    return HTMLResponse(_admin_page(token, banner))
