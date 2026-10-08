"""
Vagis backend.

ONE code per person, for life: VG-XXXX-XXXX (see VAGIS CODES below). People
are made on /admin with a Free or Premium tier. All of a person's data lives
under their code.

Endpoints:
  GET  /health                 -- status
  POST /ingest                 -- app uploads a mode's Session History CSV
  POST /ingest/timeline        -- app uploads one recording's graph data
  POST /portal/validate        -- app checks a VG code is real (path kept for the app)
  POST /mcp/{key}              -- personal Claude/ChatGPT connector (Premium only)
  POST /help/chat              -- in-app Vagis Help (guide only, no user data)
  POST /chat                   -- old in-app agent relay (no longer used by the app)
  GET  /admin                  -- add people, set tiers, connector addresses, studies
  POST /mcp/study/{key}        -- a researcher's study connector (all subjects in the study)
  GET  /study/{key}            -- a researcher's download page
  GET  /study/{key}/download   -- zip of the study's data
"""

from __future__ import annotations

import asyncio
import csv
import io
import os
import secrets
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
# VAGIS CODES — one permanent code per person
# --------------------------------------------------------------------------
# Format VG-XXXX-XXXX: 8 random characters from an alphabet with no look-alikes
# (no O/0, I/1/L). The code carries no meaning: not a study, not a group.
# What a person can use is decided by their tier, stored next to the code.
# Their Session History lives in research_uploads under the code.
VG_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
VG_TIERS = ("free", "premium")

CREATE_PEOPLE_SQL = """
CREATE TABLE IF NOT EXISTS vagis_people (
    code         TEXT PRIMARY KEY,
    name         TEXT,
    email        TEXT,
    tier         TEXT NOT NULL DEFAULT 'free',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


# Graph data: one small time series per recording per graph (e.g. the sleep
# hypnogram's stage per 30 s), so a connected agent can redraw the app's graphs
# without raw data. GENERIC: the app names the mode and graph; the server
# stores whatever arrives and the connector lists whatever is there, so a new
# graph later is an app change plus a GRAPH_STYLES entry, nothing else.
CREATE_TIMELINES_SQL = """
CREATE TABLE IF NOT EXISTS vagis_timelines (
    id           SERIAL PRIMARY KEY,
    person_code  TEXT NOT NULL,
    mode         TEXT NOT NULL,
    graph        TEXT NOT NULL,
    local_start  TEXT NOT NULL,
    csv_text     TEXT NOT NULL,
    row_count    INTEGER,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (person_code, mode, graph, local_start)
);
"""


def make_vg_code() -> str:
    body = "".join(secrets.choice(VG_ALPHABET) for _ in range(8))
    return f"VG-{body[:4]}-{body[4:]}"


def parse_vg_code(raw: str) -> Optional[str]:
    """Canonical VG-XXXX-XXXX, or None. Accepts any case, spaces or dashes."""
    s = "".join(ch for ch in (raw or "").upper() if ch.isalnum())
    if len(s) == 10 and s.startswith("VG") and all(c in VG_ALPHABET for c in s[2:]):
        return f"VG-{s[2:6]}-{s[6:]}"
    return None


def vg_person(cur, code: str) -> Optional[dict[str, Any]]:
    cur.execute("SELECT code, name, email, tier, created_at "
                "FROM vagis_people WHERE code = %s;", (code,))
    r = cur.fetchone()
    if not r:
        return None
    return {"code": r[0], "name": r[1], "email": r[2], "tier": r[3],
            "created_at": r[4]}


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

UPSERT_RESEARCH_SQL = """
INSERT INTO research_uploads (person_code, mode, filename, csv_text, row_count, uploaded_at)
VALUES (%s, %s, %s, %s, %s, now())
ON CONFLICT (person_code, mode)
DO UPDATE SET filename=EXCLUDED.filename, csv_text=EXCLUDED.csv_text,
              row_count=EXCLUDED.row_count, uploaded_at=now()
RETURNING uploaded_at;
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
    cur.execute(CREATE_RESEARCH_UPLOADS_SQL)
    cur.execute(CREATE_AGENT_USAGE_SQL)
    cur.execute(CREATE_PEOPLE_SQL)
    cur.execute(CREATE_TIMELINES_SQL)


@app.on_event("startup")
def init_db() -> None:
    if not DATABASE_URL:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
        conn.close()
    except Exception as e:
        print(f"[startup] db init failed: {type(e).__name__}: {e}")


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


@app.post("/ingest")
async def ingest(
    enrollment_code: str = Form(...),
    mode: str = Form(...),
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Store one mode's cumulative Session History CSV under a VG code."""
    check_app_auth(authorization)
    vg = parse_vg_code(enrollment_code)
    if not vg:
        raise HTTPException(status_code=400, detail="That isn't a valid Vagis code.")
    return await _ingest_vg(vg, mode, file)


async def _ingest_vg(code: str, mode: str, file: UploadFile) -> dict[str, Any]:
    """Store one Session History CSV under a VG code (persistent)."""
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
            if not vg_person(cur, code):
                raise HTTPException(status_code=404, detail="That Vagis code isn't recognized.")
            cur.execute(UPSERT_RESEARCH_SQL, (code, mode_clean, file.filename, csv_text, row_count))
            uploaded_at = cur.fetchone()[0]
    finally:
        conn.close()
    return {"status": "ok", "enrollment_code": code, "mode": mode_clean,
            "row_count": row_count, "uploaded_at": uploaded_at.isoformat()}


@app.post("/ingest/timeline")
async def ingest_timeline(
    enrollment_code: str = Form(...),
    mode: str = Form(...),
    graph: str = Form(...),
    local_start: str = Form(...),
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Store one recording's graph data. local_start is the recording's start
    in the phone's local time, ISO 8601 with offset (2026-10-03T23:12:00-07:00).
    Sending the same mode/graph/start again replaces it."""
    check_app_auth(authorization)
    code = parse_vg_code(enrollment_code)
    if not code:
        raise HTTPException(status_code=400, detail="Graph data needs a Vagis (VG) code.")
    mode_c = (mode or "").strip().lower()
    graph_c = (graph or "").strip().lower()
    start = (local_start or "").strip()
    if not mode_c or not graph_c or len(start) < 16:
        raise HTTPException(status_code=400, detail="mode, graph and local_start are required.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text CSV.")
    n = count_csv_rows(text)
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            if not vg_person(cur, code):
                raise HTTPException(status_code=404, detail="That Vagis code isn't recognized.")
            cur.execute("""
                INSERT INTO vagis_timelines (person_code, mode, graph, local_start, csv_text, row_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (person_code, mode, graph, local_start)
                DO UPDATE SET csv_text = EXCLUDED.csv_text, row_count = EXCLUDED.row_count,
                              uploaded_at = now();""", (code, mode_c, graph_c, start, text, n))
    finally:
        conn.close()
    return {"status": "ok", "mode": mode_c, "graph": graph_c, "local_start": start, "row_count": n}


class ValidateRequest(BaseModel):
    enrollment_code: str


@app.post("/portal/validate")
def validate_person(req: ValidateRequest,
                    authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """App checks a VG code is well-formed AND issued. (Path kept for the app.)"""
    check_app_auth(authorization)
    vg = parse_vg_code(req.enrollment_code)
    if not vg:
        return {"valid": False, "reason": "malformed"}
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            person = vg_person(cur, vg)
    finally:
        conn.close()
    if not person:
        return {"valid": False, "reason": "not_issued"}
    return {"valid": True, "enrollment_code": vg, "tier": person["tier"]}


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


def _mailto(email: str, subject: str, body: str, text: str) -> str:
    if not email:
        return ""
    href = f"mailto:{_q(email)}?subject={_q(subject)}&body={_q(body)}"
    return (f'<a href="{href}" style="display:inline-block;margin-top:10px;padding:9px 16px;'
            f'background:#0f6e56;color:#fff;border-radius:8px;font-size:14px;font-weight:500;'
            f'text-decoration:none;">{_esc(text)}</a>')


# ---- Admin page ----------------------------------------------------------
def _admin_page(token: str = "", banner: str = "", people_html: str = "",
                studies_html: str = "") -> str:
    tok = _esc(token)
    tier_opts = "".join(f'<option value="{t}">{t.capitalize()}</option>' for t in VG_TIERS)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vagis Admin</title>{_style()}</head><body>
  <h1>Vagis Admin</h1>
  <p class="sub">One Vagis code per person, for life.</p>
  {banner}
  <div class="card">
    <h2>Add a person</h2>
    <p class="sub">Makes their Vagis code and their personal Claude/ChatGPT connector address.</p>
    <form method="post" action="/admin/ui/person/add">
      <label>Admin token</label>
      <input name="token" type="password" placeholder="Your VAGIS_ADMIN_TOKEN" value="{tok}" autocomplete="off">
      <label>Name</label>
      <input name="name" type="text" placeholder="Tess">
      <label>Email</label>
      <input name="email" type="text" placeholder="tess@example.com">
      <label>Tier</label>
      <select name="tier">{tier_opts}</select>
      <button type="submit">Add person</button>
    </form>
  </div>
  <div class="card">
    <h2>People</h2>
    <form method="post" action="/admin/ui/people">
      <label>Admin token</label>
      <input name="token" type="password" placeholder="Your VAGIS_ADMIN_TOKEN" value="{tok}" autocomplete="off">
      <button type="submit" class="secondary">Show people</button>
    </form>
    {people_html}
  </div>
  <div class="card">
    <h2>Studies</h2>
    <p class="sub">A study groups Vagis codes. Each researcher gets a study connector address
    (shows everyone in the study in Claude/ChatGPT) and a download page.</p>
    <form method="post" action="/admin/ui/study/add">
      <label>Admin token</label>
      <input name="token" type="password" placeholder="Your VAGIS_ADMIN_TOKEN" value="{tok}" autocomplete="off">
      <label>New study name</label>
      <input name="name" type="text" placeholder="Jason research">
      <button type="submit">Add study</button>
      <button type="submit" formaction="/admin/ui/studies" class="secondary">Show studies</button>
    </form>
    {studies_html}
  </div>
</body></html>"""


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> HTMLResponse:
    return HTMLResponse(_admin_page())


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
#   get_saved_variants   genomic variants the user saved in the app's Genomics
#   save_note / get_notes  short notes carried between conversations
#
# Only Session History metrics (what the app uploads via Data Share) are ever
# served. No raw signal exists on the server.
#
# ADAPTS TO APP CHANGES. Session History is served exactly as uploaded, with
# whatever columns it has, and any mode name the app sends is listed. New or
# renamed metrics and new modes need no change here — only the guide text.
# --------------------------------------------------------------------------
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
- **Cycling** — "Pulse Wave Cycling": three lanes across the night, one each for PWA-HR cycling (sky blue), PWA-cycling (strong blue) and Vaso-cycling (light cyan). The strength of the shading shows how much of each 5-minute period was spent in that state; there is no vertical scale. The number at top right is PWA-cycling plus PWA-HR cycling per hour of sleep. Below the graph, a card for each shows Per hour and Time (min).
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
Every minute of sleep is placed in exactly one of three states, so the three times add up to the night's total:
- **Vaso-cycling** — the slow, resting rhythm of the finger's blood vessels, no stronger than the person shows in their own Deep sleep. A resting state, not an event.
- **PWA-cycling** — slow swings in pulse size stronger than the person's own Deep-sleep level, without a heart-rate surge.
- **PWA-HR cycling** — the same, with a heart-rate surge larger than the person's own Deep-sleep heart-rate swing.

Each is shown as **Per hour** (per hour of sleep) and **Time** (min). **Total** is PWA-cycling plus PWA-HR cycling; it does not include Vaso-cycling. In Session History the trend choices are PWA-HR, PWA, Vaso and Total, plus each one's minutes.

These are measured pulse-wave patterns, not breathing events. They are not expected to match oximetry. Do not call them apneas or convert them to AHI.

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

Each saved variant includes the rsID, gene, the user's genotype, and an optional note on why it matters to them. Saved variants can be grouped. Read them with get_saved_variants; they reach the server when the user taps Send my data in Data Share. If a variant the user asks about isn't saved, ask them to search it in Genomics, save it, and send again.

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
                "Analysis > Data Share and tap Send my data." + _graph_data_summary(person))
    lines = ["Session History on the server (one row per recording):"]
    for mode, n, up, text in rows:
        if mode == GENOMICS_MODE:
            continue
        header = next(csv.reader(io.StringIO(text)), [])
        lines.append(f"- {mode}: {n} recording(s), last sent "
                     f"{up.strftime('%Y-%m-%d %H:%M UTC') if up else 'unknown'}; "
                     f"columns: {', '.join(header)}")
    gen = next((r for r in rows if r[0] == GENOMICS_MODE), None)
    if gen and gen[1]:
        lines.append(f"\nSaved genomic variants: {gen[1]}, last sent "
                     f"{gen[2].strftime('%Y-%m-%d %H:%M UTC') if gen[2] else 'unknown'}. "
                     "Read them with get_saved_variants.")
    else:
        lines.append("\nNo saved genomic variants on the server.")
    lines.append(_graph_data_summary(person))
    return "\n".join(lines)


GENOMICS_MODE = "genomics"


# ---- Graph data (redrawing the app's graphs) -----------------------------
# How each graph looks in the app, so the agent's redraw matches it. Keyed
# "mode/graph". A graph with no entry here is still listed and returned; the
# agent then just plots its columns against time.
GRAPH_STYLE_COMMON = (
    "Draw on a black background with white/grey text, no chart junk. The time column "
    "is local clock time (graphs with a freq_hz, ms or t_s column use that instead). Label the x-axis with whole hours (hour number only for "
    "overnight graphs). When the user asks for several recordings, stack them one "
    "above the other sharing the same x-axis (align by clock time for sleep, by "
    "elapsed time otherwise) or place them side by side, as the user prefers, each "
    "titled with its date. Never draw raw data; these are the app's derived values."
)
GRAPH_STYLES: dict[str, str] = {
    "sleep/stages": (
        "Sleep Stages hypnogram. Columns: time, stage (Awake, REM, Light, Deep), one row "
        "per 30 s. Draw it as the app does: four horizontal lanes, top to bottom Awake, "
        "REM, Light, Deep, with a filled bar in a stage's lane for every stretch spent in "
        "it (thin connectors between lanes are optional). Colours: Awake #FFFFFF, REM "
        "#5AA6EE, Light #3554C9, Deep #4FDBFF. Beside each lane label you may show that "
        "stage's percentage of the night. Under the lanes the app draws a motion strip: "
        "take it from the same night's sleep/heart_rate graph data (motion column)."),
    "sleep/heart_rate": (
        "Overnight heart rate. Columns: time, hr (bpm), motion, beats_pct; one row per "
        "30 s. Line #3B82F6, y-axis in bpm. Under it draw a thin motion strip: one bar "
        "per row, height = motion (the largest movement score in that 30 s; under 2 is "
        "still, 2-10 some movement, 10 or more large movement), colour #8E8E93. "
        "beats_pct is signal quality: how much of the 30 s was covered by detected "
        "beats (100 = every beat found). Lower values mean beats were missed there, so "
        "treat hr in those rows with caution; when the user asks about data quality, "
        "use it and motion together. motion and beats_pct are blank for nights whose "
        "beat file is no longer on the phone."),
    "sleep/cycling": (
        "Pulse Wave Cycling. Columns: time (start of each 5-min bin), pwa_hr, pwa, vaso "
        "(counts in that bin). Draw THREE LANES, not overlaid curves, top to bottom: "
        "PWA-HR cycling #5AA6EE, PWA-cycling #3554C9, Vaso-cycling #4FDBFF. In each lane "
        "fill a block for every bin with a count above zero; opacity 0.35 + 0.65 x "
        "(count / the night's largest bin count across all three). No y-axis scale. "
        "Never add Vaso-cycling to the other two as a total."),
    "load/timeline": (
        "Load Timeline. Columns: time, hr (bpm), motion_g (g). Heart rate as a line "
        "#3B82F6 in an upper panel; motion below it as a filled area sharing the time "
        "axis. Motion bands: Still < 0.02 g, Light 0.02-0.07, Moderate 0.07-0.15, "
        "High > 0.15 g."),
    "exertion/heart_rate": (
        "Exertion heart rate. Columns: time, hr (bpm), from the ring's stored HR. "
        "Line #3B82F6, y-axis in bpm."),
    "breathwork/heart_rate": (
        "Breathwork heart rate (tachogram). Columns: time, hr (bpm), one row per beat. "
        "Line #3B82F6; paced breathing shows as regular waves in HR."),
    "breathwork/spectrum": (
        "Breathwork breathing spectrum. Columns: freq_hz, power. The x-axis is "
        "frequency in Hz (not time), 0 to 0.5 Hz; draw power as a filled curve #3B82F6. "
        "Mark the LF band 0.04-0.15 Hz and HF band 0.15-0.40 Hz; the tallest peak is "
        "the breathing frequency. Several sessions can be overlaid in one plot."),
    "stand/waveform": (
        "Stand Waveform Shape. Columns: phase (Supine or Supine 1, Standing, Supine 2), "
        "ms (time from the pulse foot), shape (the averaged beat scaled 0-1), ttp_ms "
        "(that phase's Time to Peak, blank if no clear peak). Overlay the phases on one "
        "plot, x = ms from pulse foot, y = Pulse Shape 0-1 with no numbers on the axis. "
        "Colours: Supine 1 #5AA6EE, Standing #3554C9 drawn thicker and on top, "
        "Supine 2 #9AA7B4. Mark each phase's Time to Peak with a faint dashed drop line "
        "in its colour. Legend names only; put Time to Peak values beside the graph."),
    "quick_check/pulse": (
        "Quick Check Finger Pulse. Given compactly: start_s, sample_rate_hz, pulse (evenly "
        "spaced samples of the filtered pulse signal the app draws, already oriented; "
        "sample i is at start_s + i / sample_rate_hz seconds) and beat_start_s (times "
        "where beats begin). Interpolate smoothly (cubic) between samples. Draw as a single smooth line #5AA6EE on a "
        "black background with a small dot at each beat start. Each pulse's steeper "
        "slope must be its leading (upstroke) slope; the app has already oriented it "
        "that way, so do not flip it. No y-axis numbers; x-axis in seconds."),
    "stand/heart_rate": (
        "Stand test heart rate and motion. Columns: time, hr (bpm), phase (Supine or "
        "Supine 1, Standing, Supine 2), motion (the app's accel score). HR as a line "
        "#3B82F6 in an upper panel, motion as a thin trace in a short panel below "
        "sharing the time axis; shade or label each phase band behind both; the "
        "standing cue is where phase changes to Standing. Exertion has no motion data: "
        "the ring's stored HR carries none."),
}


def _graph_data_summary(person: str) -> str:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            cur.execute("SELECT mode, graph, COUNT(*), MIN(local_start), MAX(local_start) "
                        "FROM vagis_timelines WHERE person_code = %s "
                        "GROUP BY mode, graph ORDER BY mode, graph;", (person,))
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return "\nNo graph data on the server yet."
    out = ["\nGraph data for redrawing the app's graphs (read with get_graph_data):"]
    for mode, graph, n, lo, hi in rows:
        out.append(f"- {mode}/{graph}: {n} recording(s), {lo[:10]} to {hi[:10]}")
    return "\n".join(out)


def _tool_get_graph_data(person: str, args: dict) -> str:
    mode = (args.get("mode") or "").strip().lower()
    graph = (args.get("graph") or "").strip().lower()
    dates = [str(d)[:10] for d in (args.get("dates") or []) if d]
    latest = max(1, min(int(args.get("latest") or 1), 14))
    if not mode or not graph:
        return "Give mode and graph, e.g. mode=sleep, graph=stages. list_my_data lists them."
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            if dates:
                cur.execute("SELECT local_start, csv_text FROM vagis_timelines "
                            "WHERE person_code = %s AND mode = %s AND graph = %s "
                            "AND LEFT(local_start, 10) = ANY(%s) ORDER BY local_start;",
                            (person, mode, graph, dates[:14]))
            else:
                cur.execute("SELECT local_start, csv_text FROM vagis_timelines "
                            "WHERE person_code = %s AND mode = %s AND graph = %s "
                            "ORDER BY local_start DESC LIMIT %s;", (person, mode, graph, latest))
            rows = cur.fetchall()
            if not dates:
                rows = rows[::-1]
            cur.execute("SELECT DISTINCT LEFT(local_start, 10) FROM vagis_timelines "
                        "WHERE person_code = %s AND mode = %s AND graph = %s "
                        "ORDER BY 1 DESC LIMIT 60;", (person, mode, graph))
            available = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    if not rows:
        avail = ", ".join(available) if available else "none"
        return (f"No {mode}/{graph} graph data for that request. Dates available: {avail}. "
                "A sleep recording is dated by the evening it started.")
    style = GRAPH_STYLES.get(f"{mode}/{graph}",
                             "No style on file for this graph: plot its columns against time.")
    parts = [f"{mode}/{graph}: {len(rows)} recording(s).", "", "HOW TO DRAW IT: " + style,
             GRAPH_STYLE_COMMON, ""]
    compact = COMPACT_GRAPHS.get(f"{mode}/{graph}")
    for start, text in rows:
        body = compact(text) if compact else text.strip()
        parts += [f"=== Recording started {start} ===", body, ""]
    return "\n".join(parts)


def _compact_pulse(text: str, target_hz: float = 25.0) -> str:
    """Quick Check pulse in a short form an agent can copy straight into its
    plotting code: evenly spaced samples on one line (reduced to ~25 Hz, which
    keeps the waveform's shape, notch included) plus the beat-start times at
    full resolution."""
    recs = list(csv.DictReader(io.StringIO(text)))
    ts, ys, beats = [], [], []
    for r in recs:
        try:
            t, y = float(r["t_s"]), float(r["pulse"])
        except (KeyError, ValueError, TypeError):
            continue
        ts.append(t); ys.append(y)
        if (r.get("beat_start") or "0").strip() == "1":
            beats.append(t)
    if len(ts) < 2:
        return text.strip()
    dt = (ts[-1] - ts[0]) / (len(ts) - 1)
    step = max(1, round((1.0 / target_hz) / dt)) if dt > 0 else 1
    hz = 1.0 / (dt * step) if dt > 0 else target_hz
    vals = ",".join(str(int(round(v))) for v in ys[::step])
    return (f"start_s: {ts[0]:.2f}\nsample_rate_hz: {hz:g}\n"
            f"pulse: {vals}\n"
            f"beat_start_s: {','.join(f'{b:.2f}' for b in beats)}")


# Graphs returned in a compact form instead of their raw CSV.
COMPACT_GRAPHS = {"quick_check/pulse": _compact_pulse}


def _tool_get_saved_variants(person: str, args: dict) -> str:
    """Saved variants as uploaded by Data Share (mode "genomics"), optionally
    filtered by gene or group name."""
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _ai_ensure_tables(cur)
            cur.execute("SELECT csv_text, uploaded_at FROM research_uploads "
                        "WHERE person_code = %s AND mode = %s;", (person, GENOMICS_MODE))
            row = cur.fetchone()
    finally:
        conn.close()
    none_msg = ("No saved genomic variants on the server. In the Vagis app, search and "
                "save variants in Genomics, then open Analysis > Data Share and tap "
                "Send my data.")
    if not row:
        return none_msg
    text, up = row
    recs = list(csv.DictReader(io.StringIO(text)))
    if not recs:
        return none_msg
    gene = (args.get("gene") or "").strip().upper()
    group = (args.get("group") or "").strip().lower()
    if gene:
        recs = [r for r in recs if (r.get("gene") or "").upper() == gene]
    if group:
        recs = [r for r in recs if (r.get("group") or "").lower() == group]
    if not recs:
        return "No saved variants match that filter."
    cols = list(recs[0].keys())
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(recs)
    sent = up.strftime("%Y-%m-%d %H:%M UTC") if up else "unknown"
    return (f"Saved genomic variants (CSV; last sent from the app {sent}). "
            "source = saved (picked individually) or group (from an imported study "
            "list, named in group). genotype = the user's bases; genotype_raw = the "
            "VCF call (0 = reference allele, 1 = first alt). gene is blank when the "
            "variant isn't in the app's gene list. Positions are as in the user's "
            "VCF file. call = vcf (read from the user's VCF) or inferred_reference "
            "(not in the VCF; a sequencing VCF lists only positions that differ from "
            "the reference, so the user most likely has two reference copies — say it "
            "is inferred, not measured).\n\n" + out.getvalue())


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
    "get_saved_variants": {
        "fn": _tool_get_saved_variants,
        "description": ("Genomic variants the user saved in the Vagis app's Genomics "
                        "section: rsID, gene, position, alleles, the user's genotype and "
                        "their note. Only variants the user chose to save; the genome "
                        "file never leaves the phone. Read the background_data guide "
                        "section before discussing them."),
        "schema": {"type": "object", "properties": {
            "gene": {"type": "string", "description": "Optional: only this gene, e.g. COMT."},
            "group": {"type": "string",
                      "description": "Optional: only variants from this imported group."}},
            "additionalProperties": False},
    },
    "get_graph_data": {
        "fn": _tool_get_graph_data,
        "description": ("Data for redrawing one of the app's graphs (e.g. the sleep "
                        "hypnogram) for one or more recordings, with instructions for how "
                        "the app draws it. Use it whenever the user wants to see an app "
                        "graph, or several nights stacked or side by side. list_my_data "
                        "shows which graphs and dates exist."),
        "schema": {"type": "object", "properties": {
            "mode": {"type": "string", "description": "e.g. sleep, load, exertion, stand."},
            "graph": {"type": "string",
                      "description": "e.g. stages, heart_rate, cycling, timeline."},
            "dates": {"type": "array", "items": {"type": "string"},
                      "description": "Optional YYYY-MM-DD dates (up to 14). A sleep "
                                     "recording is dated by the evening it started."},
            "latest": {"type": "integer",
                       "description": "If no dates: how many most recent recordings (1-14)."}},
            "required": ["mode", "graph"], "additionalProperties": False},
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
    "get_session_history for the data, get_graph_data to redraw the app's graphs, and get_saved_variants for saved genomic variants. Offer to save a short note at the end of a "
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
    if person.startswith("VG-") and await asyncio.to_thread(_vg_tier, person) != "premium":
        return JSONResponse(status_code=403,
                            content={"error": "This Vagis connection needs a Premium plan."})
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


# ==========================================================================
# VAGIS HELP — the in-app help chat (all users)
# ==========================================================================
# Answers questions about how to use the app and what its graphs and metrics
# mean, using only the Vagis guide (GUIDE_TEXT above). It receives no user
# data, has no internet and runs no statistics.
#
# Uses a small, inexpensive model with the guide cached between questions.
# Shares the per-phone daily/monthly question caps with /chat.
# --------------------------------------------------------------------------
HELP_MODEL = os.environ.get("VAGIS_HELP_MODEL", "claude-haiku-4-5")
HELP_MAX_TOKENS = int(os.environ.get("VAGIS_HELP_MAX_TOKENS", "700"))
HELP_MAX_TURNS = 12   # most recent messages sent with each question

HELP_RULES = """You are Vagis Help, the help assistant inside the Vagis app.

Answer questions about how to use the Vagis app and what its modes, graphs and
metrics mean, using ONLY the Vagis guide below.

- You have no access to the user's data. If the user types in numbers or
  describes results, you may discuss them, following the guide's rules: research
  use only, no diagnosis, no severity words, compare with the user's own history.
- If the guide does not cover something, say so plainly rather than guessing.
- Never describe how any metric is calculated.
- You can only discuss numbers the user types in. Do not offer to compare
  recordings unless the user gives you the numbers.
- For deeper questions about their own data, users can tap the share icon on a
  graph and choose "Send to AI". That sends the graph and that mode's Session
  History to their own chat assistant (Claude, ChatGPT or another). It covers
  that one mode only, not all of their data.
- Keep answers short and plain. Use the metric names exactly as the app shows them.

THE VAGIS GUIDE
"""


class HelpTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class HelpRequest(BaseModel):
    conversation: list[HelpTurn]


class HelpResponse(BaseModel):
    reply: str


def _help_system() -> list[dict]:
    guide = "\n\n----------------------------------------\n\n".join(
        _guide_text(s) or "" for s in GUIDE_SECTIONS)
    return [{"type": "text", "text": HELP_RULES + guide,
             "cache_control": {"type": "ephemeral"}}]


@app.post("/help/chat", response_model=HelpResponse)
def help_chat(req: HelpRequest, authorization: str | None = Header(default=None),
              x_vagis_device: str | None = Header(default=None)) -> HelpResponse:
    check_app_auth(authorization)
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Help is not configured on the server.")
    turns = req.conversation[-HELP_MAX_TURNS:]
    while turns and turns[0].role != "user":
        turns = turns[1:]
    if not turns or turns[-1].role != "user":
        raise HTTPException(status_code=400, detail="No question provided.")

    device_id = (x_vagis_device or "").strip()[:64]
    if device_id and device_id not in AGENT_EXEMPT_DEVICES:
        _check_agent_cap(device_id)

    try:
        message = client.messages.create(
            model=HELP_MODEL, max_tokens=HELP_MAX_TOKENS, system=_help_system(),
            messages=[{"role": t.role, "content": t.content} for t in turns])
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Help is unavailable right now ({e.status_code}).")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Help is unavailable right now ({type(e).__name__}).")

    reply = "".join(b.text for b in message.content if getattr(b, "type", None) == "text").strip()
    if not reply:
        raise HTTPException(status_code=502, detail="Help returned an empty answer. Please try again.")
    if device_id:
        _agent_record(device_id)
    return HelpResponse(reply=reply)


# ==========================================================================
# ADMIN — PEOPLE (VG codes)
# ==========================================================================
def _vg_tier(code: str) -> Optional[str]:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            p = vg_person(cur, code)
            return p["tier"] if p else None
    finally:
        conn.close()


def _admin_ok(token: str) -> bool:
    return bool(VAGIS_ADMIN_TOKEN) and token.strip() == VAGIS_ADMIN_TOKEN.strip()


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/").replace("http://", "https://")


def _vg_active_key(cur, code: str) -> Optional[str]:
    cur.execute("SELECT key FROM ai_connect_keys WHERE person_code = %s AND NOT revoked "
                "ORDER BY created_at DESC LIMIT 1;", (code,))
    r = cur.fetchone()
    return r[0] if r else None


def _vg_email_button(person: dict[str, Any], url: Optional[str]) -> str:
    """mailto button with the person's code (and connector address if Premium)."""
    if not person.get("email"):
        return ""
    name = person.get("name") or ""
    lines = [f"Hi {name}," if name else "Hi,", "",
             f"Your Vagis code: {person['code']}", "",
             "Enter it in the Vagis app: Analysis > Data Share > type the code > Check > Start sharing.",
             "After that your data is sent automatically after each recording."]
    if person["tier"] == "premium" and url:
        lines += ["", "Your private Claude / ChatGPT connector address "
                  "(keep it private, like a password):", url, "",
                  "In Claude: Customize > Connectors > + > Add custom connector, "
                  "name it Vagis and paste the address. Full steps are in the attached instructions."]
    lines += ["", "Jason"]
    return _mailto(person["email"], "Your Vagis code", "\n".join(lines), "Email this person")


def _vg_result(person: dict[str, Any], url: Optional[str], extra: str = "") -> str:
    rows = [("Name", _esc(person.get("name") or "") or "&mdash;"),
            ("Vagis code", _esc(person["code"])),
            ("Tier", person["tier"].capitalize())]
    html = '<div class="result">' + "".join(
        f'<div class="row"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in rows)
    if url:
        html += ('<div class="row"><span class="k">Connector address</span>'
                 f'<span class="v" style="word-break:break-all">{_esc(url)}</span></div>')
        if person["tier"] != "premium":
            html += '<div class="warn">The connector address only works once the tier is Premium.</div>'
    html += extra + _vg_email_button(person, url) + "</div>"
    return html


@app.post("/admin/ui/person/add", response_class=HTMLResponse)
def admin_person_add(request: Request, token: str = Form(""), name: str = Form(""),
                     email: str = Form(""), tier: str = Form("free")) -> HTMLResponse:
    if not _admin_ok(token):
        return HTMLResponse(_admin_page(token, '<div class="err">Admin token did not match.</div>'))
    tier = tier if tier in VG_TIERS else "free"
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _ai_ensure_tables(cur)
            code = make_vg_code()
            while vg_person(cur, code):
                code = make_vg_code()
            cur.execute("INSERT INTO vagis_people (code, name, email, tier) "
                        "VALUES (%s, %s, %s, %s);",
                        (code, name.strip() or None, email.strip() or None, tier))
            key = _ai_issue_key(cur, code)
            person = vg_person(cur, code)
    finally:
        conn.close()
    url = f"{_base_url(request)}/mcp/{key}"
    return HTMLResponse(_admin_page(token, _vg_result(person, url)))


def _people_table(request: Request, token: str) -> str:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _ai_ensure_tables(cur)
            cur.execute("SELECT code FROM vagis_people ORDER BY created_at DESC;")
            codes = [r[0] for r in cur.fetchall()]
            people = []
            for c in codes:
                p = vg_person(cur, c)
                cur.execute("SELECT mode, uploaded_at FROM research_uploads WHERE person_code = %s "
                            "ORDER BY mode;", (c,))
                ups = cur.fetchall()
                people.append((p, _vg_active_key(cur, c), ups))
    finally:
        conn.close()
    if not people:
        return '<p class="sub" style="margin-top:12px">No people yet.</p>'
    tok = _esc(token)
    rows = []
    for p, _key, ups in people:
        last = max((u[1] for u in ups), default=None)
        data = (f"{len(ups)} mode(s), last sent {last.strftime('%Y-%m-%d')}" if ups else "no data yet")
        opts = "".join(f'<option value="{t}"{" selected" if t == p["tier"] else ""}>{t.capitalize()}</option>'
                       for t in VG_TIERS)
        rows.append(f"""<tr>
  <td><b>{_esc(p['code'])}</b><br><span class="sub">{_esc(p.get('name') or '')}</span></td>
  <td>{_esc(p.get('email') or '')}<br><span class="sub">{data}</span></td>
  <td>
    <form method="post" action="/admin/ui/person/update" style="display:flex;gap:6px;align-items:center">
      <input type="hidden" name="token" value="{tok}">
      <input type="hidden" name="code" value="{_esc(p['code'])}">
      <select name="tier" style="margin:0">{opts}</select>
      <button type="submit" name="action" value="tier" class="secondary" style="margin:0;padding:6px 10px">Save</button>
    </form>
  </td>
  <td>
    <form method="post" action="/admin/ui/person/update">
      <input type="hidden" name="token" value="{tok}">
      <input type="hidden" name="code" value="{_esc(p['code'])}">
      <button type="submit" name="action" value="show" class="secondary" style="margin:0;padding:6px 10px">Show / email</button>
      <button type="submit" name="action" value="newkey" class="secondary" style="margin:4px 0 0;padding:6px 10px">New connector address</button>
    </form>
  </td>
</tr>""")
    return ('<div class="tablewrap" style="margin-top:12px"><table>'
            '<tr><th>Code</th><th>Email / data</th><th>Tier</th><th></th></tr>'
            + "".join(rows) + "</table></div>")


@app.post("/admin/ui/people", response_class=HTMLResponse)
def admin_people(request: Request, token: str = Form("")) -> HTMLResponse:
    if not _admin_ok(token):
        return HTMLResponse(_admin_page(token, '<div class="err">Admin token did not match.</div>'))
    return HTMLResponse(_admin_page(token, people_html=_people_table(request, token)))


@app.post("/admin/ui/person/update", response_class=HTMLResponse)
def admin_person_update(request: Request, token: str = Form(""), code: str = Form(""),
                        action: str = Form(""), tier: str = Form("")) -> HTMLResponse:
    if not _admin_ok(token):
        return HTMLResponse(_admin_page(token, '<div class="err">Admin token did not match.</div>'))
    vg = parse_vg_code(code)
    extra = ""
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _ai_ensure_tables(cur)
            person = vg_person(cur, vg) if vg else None
            if not person:
                return HTMLResponse(_admin_page(token, '<div class="err">Unknown code.</div>'))
            if action == "tier" and tier in VG_TIERS:
                cur.execute("UPDATE vagis_people SET tier = %s WHERE code = %s;", (tier, vg))
                person["tier"] = tier
                extra = f'<div class="warn">Tier set to {tier.capitalize()}.</div>'
            if action == "newkey":
                key = _ai_issue_key(cur, vg)
                extra = '<div class="warn">New connector address made. The previous one no longer works.</div>'
            else:
                key = _vg_active_key(cur, vg) or _ai_issue_key(cur, vg)
    finally:
        conn.close()
    url = f"{_base_url(request)}/mcp/{key}"
    return HTMLResponse(_admin_page(token, _vg_result(person, url, extra),
                                    people_html=_people_table(request, token)))


# ==========================================================================
# STUDIES — researchers see the data of every code linked to their study
# ==========================================================================
# A study is a name plus a list of linked Vagis codes. A subject needs nothing
# but their code with sharing on in the app (Free is fine). Each researcher on
# a study gets one private key, used for:
#     POST /mcp/study/<key>       study connector for Claude / ChatGPT
#     GET  /study/<key>           download page
#     GET  /study/<key>/download  zip of all the study's data
# A subject linked to several studies is visible to each. Unlinking removes
# them from that study at once; their data stays under their own code.
import zipfile as _zipfile

CREATE_STUDIES_SQL = """
CREATE TABLE IF NOT EXISTS vagis_studies (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS vagis_study_members (
    study_id     INTEGER NOT NULL,
    person_code  TEXT NOT NULL,
    label        TEXT,
    added_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (study_id, person_code)
);
CREATE TABLE IF NOT EXISTS vagis_study_access (
    key          TEXT PRIMARY KEY,
    study_id     INTEGER NOT NULL,
    researcher   TEXT,
    email        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked      BOOLEAN NOT NULL DEFAULT FALSE
);
"""


def _study_ensure(cur) -> None:
    _ai_ensure_tables(cur)
    cur.execute(CREATE_STUDIES_SQL)


def _study_for_key(key: str) -> Optional[dict[str, Any]]:
    if not key or not DATABASE_URL:
        return None
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _study_ensure(cur)
            cur.execute("SELECT s.id, s.name, a.researcher FROM vagis_study_access a "
                        "JOIN vagis_studies s ON s.id = a.study_id "
                        "WHERE a.key = %s AND NOT a.revoked;", (key,))
            r = cur.fetchone()
            return {"id": r[0], "name": r[1], "researcher": r[2]} if r else None
    finally:
        conn.close()


def _study_members(cur, study_id: int) -> list[tuple[str, str]]:
    """[(code, display name)] — display name is the label, else the code."""
    cur.execute("SELECT person_code, label FROM vagis_study_members WHERE study_id = %s "
                "ORDER BY added_at, person_code;", (study_id,))
    return [(c, (l or "").strip() or c) for c, l in cur.fetchall()]


def _subject_id(m: tuple[str, str]) -> str:
    code, disp = m
    return code if disp == code else f"{disp} ({code})"


def _study_find(cur, study_id: int, subject: str) -> Optional[str]:
    """Subject given as code or label -> code, if in this study."""
    want = (subject or "").strip()
    vg = parse_vg_code(want)
    for code, disp in _study_members(cur, study_id):
        if code == vg or disp.lower() == want.lower():
            return code
    return None


def _merge_csvs(parts: list[tuple[str, str]]) -> tuple[str, int]:
    """[(subject, csv_text)] -> one CSV with a leading subject column. Columns are
    the union across subjects (app versions can differ), in first-seen order."""
    cols: list[str] = []
    rows: list[dict] = []
    for subj, text in parts:
        rd = csv.DictReader(io.StringIO(text))
        for c in rd.fieldnames or []:
            if c not in cols:
                cols.append(c)
        for r in rd:
            r = {k: v for k, v in r.items() if k is not None}
            r["subject"] = subj
            rows.append(r)
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=["subject"] + cols, lineterminator="\n",
                       extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue(), len(rows)


# ---- Study connector tools -----------------------------------------------
def _st_list_subjects(study: dict, args: dict) -> str:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _study_ensure(cur)
            members = _study_members(cur, study["id"])
            lines = [f"Study: {study['name']}. {len(members)} subject(s)."]
            for m in members:
                code = m[0]
                cur.execute("SELECT mode, row_count, uploaded_at FROM research_uploads "
                            "WHERE person_code = %s AND mode <> %s ORDER BY mode;",
                            (code, GENOMICS_MODE))
                ups = cur.fetchall()
                cur.execute("SELECT mode, graph, COUNT(*), MIN(local_start), MAX(local_start) "
                            "FROM vagis_timelines WHERE person_code = %s "
                            "GROUP BY mode, graph ORDER BY mode, graph;", (code,))
                gr = cur.fetchall()
                lines.append(f"\n## {_subject_id(m)}")
                if not ups and not gr:
                    lines.append("No data yet (sharing not started in the app).")
                    continue
                for mode, n, up in ups:
                    lines.append(f"- {mode}: {n} recording(s), last sent "
                                 f"{up.strftime('%Y-%m-%d %H:%M UTC') if up else '?'}")
                if gr:
                    lines.append("  graph data: " + "; ".join(
                        f"{mo}/{g} {n} ({lo[:10]} to {hi[:10]})" for mo, g, n, lo, hi in gr))
    finally:
        conn.close()
    return "\n".join(lines)


def _st_subject_history(study: dict, args: dict) -> str:
    mode = (args.get("mode") or "").strip().lower()
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _study_ensure(cur)
            code = _study_find(cur, study["id"], args.get("subject") or "")
            if not code:
                return "That subject is not in this study. Call list_subjects."
            if not mode:
                return "Give a mode, e.g. sleep, stand, exertion, load, breathwork or quick_check."
            return _tool_get_session_history(code, args)
    finally:
        conn.close()


def _st_group_history(study: dict, args: dict) -> str:
    mode = (args.get("mode") or "").strip().lower()
    if not mode:
        return "Give a mode, e.g. sleep, stand, exertion, load, breathwork or quick_check."
    last_n = args.get("last_n")
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _study_ensure(cur)
            parts = []
            for m in _study_members(cur, study["id"]):
                cur.execute("SELECT csv_text FROM research_uploads "
                            "WHERE person_code = %s AND mode = %s;", (m[0], mode))
                r = cur.fetchone()
                if r:
                    text = r[0]
                    if isinstance(last_n, int) and last_n > 0:
                        ls = text.strip().splitlines()
                        text = "\n".join([ls[0]] + ls[1:][-last_n:]) if ls else text
                    parts.append((m[1], text))
    finally:
        conn.close()
    if not parts:
        return f"No subject in this study has {mode} Session History yet."
    merged, n = _merge_csvs(parts)
    return (f"{mode} Session History for {len(parts)} subject(s), {n} recording(s) "
            "(CSV, one row per recording; the subject column says whose). Column names "
            "are the app's file headers; the guide gives the names the user sees.\n\n" + merged)


def _st_graph_data(study: dict, args: dict) -> str:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _study_ensure(cur)
            code = _study_find(cur, study["id"], args.get("subject") or "")
    finally:
        conn.close()
    if not code:
        return "That subject is not in this study. Call list_subjects."
    return _tool_get_graph_data(code, args)


def _st_guide(study: dict, args: dict) -> str:
    return _tool_get_vagis_guide("", args)


def _st_save_note(study: dict, args: dict) -> str:
    return _tool_save_note(f"STUDY-{study['id']}", args)


def _st_get_notes(study: dict, args: dict) -> str:
    return _tool_get_notes(f"STUDY-{study['id']}", args)


_SUBJ = {"type": "string", "description": "Subject name or Vagis code, as list_subjects shows it."}
STUDY_TOOLS: dict[str, dict[str, Any]] = {
    "get_vagis_guide": {**AI_TOOLS["get_vagis_guide"], "fn": _st_guide},
    "list_subjects": {
        "fn": _st_list_subjects,
        "description": ("Everyone in this study, and for each: which modes have Session "
                        "History, how many recordings, when last sent, and which graph "
                        "data exists. Call this first."),
        "schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "get_subject_history": {
        "fn": _st_subject_history,
        "description": "One subject's Session History for one mode, as CSV (one row per recording).",
        "schema": {"type": "object", "properties": {
            "subject": _SUBJ,
            "mode": {"type": "string", "description": "e.g. sleep, stand, exertion, load, breathwork, quick_check."},
            "last_n": {"type": "integer", "minimum": 1, "description": "Optional: only the most recent N."}},
            "required": ["subject", "mode"], "additionalProperties": False},
    },
    "get_group_history": {
        "fn": _st_group_history,
        "description": ("One mode's Session History for every subject in the study, as one "
                        "CSV with a subject column. Use it to compare subjects or run group "
                        "statistics."),
        "schema": {"type": "object", "properties": {
            "mode": {"type": "string", "description": "e.g. sleep, stand, exertion, load, breathwork, quick_check."},
            "last_n": {"type": "integer", "minimum": 1,
                       "description": "Optional: only each subject's most recent N."}},
            "required": ["mode"], "additionalProperties": False},
    },
    "get_graph_data": {
        "fn": _st_graph_data,
        "description": ("Data for redrawing one of the app's graphs (e.g. the sleep "
                        "hypnogram) for one subject, one or more recordings, with "
                        "instructions for how the app draws it. Call once per subject "
                        "to show several subjects."),
        "schema": {"type": "object", "properties": {
            "subject": _SUBJ,
            **AI_TOOLS["get_graph_data"]["schema"]["properties"]},
            "required": ["subject", "mode", "graph"], "additionalProperties": False},
    },
    "save_note": {**AI_TOOLS["save_note"], "fn": _st_save_note,
                  "description": ("Save a short note about this study conversation so later "
                                  "conversations can pick up from it. Ask before saving.")},
    "get_notes": {**AI_TOOLS["get_notes"], "fn": _st_get_notes,
                  "description": "Notes saved from earlier conversations about this study, newest first."},
}

STUDY_INSTRUCTIONS = (
    "Vagis study connector: Vagis smart-ring metrics for every subject in one research "
    "study. At the start call get_vagis_guide (overview and rules), get_notes and "
    "list_subjects. Read a mode's guide section before discussing it. Follow the guide's "
    "rules (research use only, no diagnosis, metric names as measured, never describe how "
    "metrics are calculated). 'The user' in the guide means the subject. Use "
    "get_subject_history or get_group_history for the data and get_graph_data to redraw "
    "the app's graphs."
)


def _study_mcp_handle(study: dict, msg: Any) -> Optional[dict]:
    if not isinstance(msg, dict):
        return _rpc_error(None, -32600, "Invalid request")
    method, msg_id = msg.get("method"), msg.get("id")
    if msg_id is None:
        return None
    params = msg.get("params") or {}
    if method == "initialize":
        asked = params.get("protocolVersion")
        return _rpc_result(msg_id, {
            "protocolVersion": asked if asked in MCP_PROTOCOL_VERSIONS else MCP_PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "vagis-study", "title": f"Vagis study: {study['name']}",
                           "version": "1.0.0"},
            "instructions": STUDY_INSTRUCTIONS,
        })
    if method == "ping":
        return _rpc_result(msg_id, {})
    if method == "tools/list":
        return _rpc_result(msg_id, {"tools": [
            {"name": n, "description": t["description"], "inputSchema": t["schema"]}
            for n, t in STUDY_TOOLS.items()]})
    if method == "tools/call":
        name = params.get("name")
        tool = STUDY_TOOLS.get(name)
        if not tool:
            return _rpc_error(msg_id, -32602, f"Unknown tool: {name}")
        args = params.get("arguments") or {}
        try:
            text = tool["fn"](study, args if isinstance(args, dict) else {})
            return _rpc_result(msg_id, {"content": [{"type": "text", "text": text}],
                                        "isError": False})
        except Exception as e:
            print(f"[study] tool {name} failed: {type(e).__name__}: {e}")
            return _rpc_result(msg_id, {"content": [{"type": "text",
                               "text": "The Vagis server could not complete that request."}],
                               "isError": True})
    return _rpc_error(msg_id, -32601, f"Method not found: {method}")


@app.post("/mcp/study/{key}")
async def study_mcp_endpoint(key: str, request: Request):
    study = await asyncio.to_thread(_study_for_key, key)
    if not study:
        return JSONResponse(status_code=404,
                            content={"error": "Unknown or retired Vagis study address."})
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content=_rpc_error(None, -32700, "Parse error"))
    if isinstance(body, list):
        replies = [r for r in [await asyncio.to_thread(_study_mcp_handle, study, m) for m in body] if r]
        return JSONResponse(replies) if replies else Response(status_code=202)
    reply = await asyncio.to_thread(_study_mcp_handle, study, body)
    return JSONResponse(reply) if reply else Response(status_code=202)


@app.get("/mcp/study/{key}")
def study_mcp_get(key: str):
    return Response(status_code=405, headers={"Allow": "POST"})


# ---- Plain download --------------------------------------------------------
def _study_zip(study: dict) -> bytes:
    """session_history/<mode>.csv  — all subjects, subject column first
       graph_data/<subject>/<mode>__<graph>__<start>.csv — one file per recording
       subjects.csv"""
    buf = io.BytesIO()
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur, _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as z:
            _study_ensure(cur)
            members = _study_members(cur, study["id"])
            by_mode: dict[str, list[tuple[str, str]]] = {}
            subj_rows = ["subject,vagis_code"]
            for code, disp in members:
                subj_rows.append(f'"{disp}",{code}')
                cur.execute("SELECT mode, csv_text FROM research_uploads "
                            "WHERE person_code = %s AND mode <> %s;", (code, GENOMICS_MODE))
                for mode, text in cur.fetchall():
                    by_mode.setdefault(mode, []).append((disp, text))
                cur.execute("SELECT mode, graph, local_start, csv_text FROM vagis_timelines "
                            "WHERE person_code = %s ORDER BY local_start;", (code,))
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in disp)
                for mode, graph, start, text in cur.fetchall():
                    st = "".join(ch for ch in start if ch.isalnum() or ch in "-T")[:30]
                    z.writestr(f"graph_data/{safe}/{mode}__{graph}__{st}.csv", text)
            z.writestr("subjects.csv", "\n".join(subj_rows) + "\n")
            for mode, parts in sorted(by_mode.items()):
                z.writestr(f"session_history/{mode}.csv", _merge_csvs(parts)[0])
    finally:
        conn.close()
    return buf.getvalue()


@app.get("/study/{key}/download")
def study_download(key: str):
    study = _study_for_key(key)
    if not study:
        return HTMLResponse("<p>Unknown or retired study link.</p>", status_code=404)
    fname = "".join(ch if ch.isalnum() else "_" for ch in study["name"]).strip("_") or "study"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Response(_study_zip(study), media_type="application/zip",
                    headers={"Content-Disposition":
                             f'attachment; filename="vagis_{fname}_{stamp}.zip"'})


@app.get("/study/{key}", response_class=HTMLResponse)
def study_page(key: str) -> HTMLResponse:
    study = _study_for_key(key)
    if not study:
        return HTMLResponse("<p>Unknown or retired study link.</p>", status_code=404)
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _study_ensure(cur)
            rows = []
            for code, disp in _study_members(cur, study["id"]):
                cur.execute("SELECT mode, row_count, uploaded_at FROM research_uploads "
                            "WHERE person_code = %s AND mode <> %s ORDER BY mode;",
                            (code, GENOMICS_MODE))
                ups = cur.fetchall()
                last = max((u[2] for u in ups if u[2]), default=None)
                modes = ", ".join(f"{m} ({n})" for m, n, _ in ups) or "no data yet"
                rows.append(f"<tr><td><b>{_esc(disp)}</b><br><span class='muted'>{_esc(code)}</span></td>"
                            f"<td style='white-space:normal'>{_esc(modes)}</td>"
                            f"<td>{last.strftime('%Y-%m-%d') if last else '&mdash;'}</td></tr>")
    finally:
        conn.close()
    table = ("<div class='tablewrap'><table><tr><th>Subject</th><th>Recordings</th><th>Last sent</th></tr>"
             + "".join(rows) + "</table></div>") if rows else "<p class='sub'>No subjects linked yet.</p>"
    return HTMLResponse(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vagis study</title>{_style()}</head><body>
  <h1>{_esc(study['name'])}</h1>
  <p class="sub">Vagis study data{(' &middot; ' + _esc(study['researcher'])) if study['researcher'] else ''}</p>
  <div class="card">
    <h2>Download</h2>
    <p class="sub">One zip: a Session History CSV per mode with every subject (subject column first),
    plus each recording's graph data in graph_data/&lt;subject&gt;/.</p>
    <a href="/study/{_esc(key)}/download" style="display:inline-block;padding:10px 18px;background:#0f6e56;
       color:#fff;border-radius:8px;text-decoration:none;font-weight:500">Download all data (.zip)</a>
  </div>
  <div class="card"><h2>Subjects</h2>{table}</div>
</body></html>""")


# ---- Admin: studies --------------------------------------------------------
def _studies_html(request: Request, token: str) -> str:
    base = _base_url(request)
    tok = _esc(token)
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _study_ensure(cur)
            cur.execute("SELECT id, name FROM vagis_studies ORDER BY created_at DESC;")
            studies = cur.fetchall()
            blocks = []
            for sid, sname in studies:
                members = _study_members(cur, sid)
                cur.execute("SELECT code, name FROM vagis_people;")
                names = dict(cur.fetchall())
                cur.execute("SELECT key, researcher, email FROM vagis_study_access "
                            "WHERE study_id = %s AND NOT revoked ORDER BY created_at;", (sid,))
                access = cur.fetchall()
                hidden = (f'<input type="hidden" name="token" value="{tok}">'
                          f'<input type="hidden" name="study_id" value="{sid}">')
                mem_rows = "".join(
                    f'<div class="subrow"><span><b>{_esc(disp)}</b> '
                    f'<span class="muted mono">{_esc(code) if disp != code else ""}</span>'
                    f'<span class="muted"> &middot; in People as {_esc(names.get(code) or "(no name)")}</span></span>'
                    f'<form class="inline" method="post" action="/admin/ui/study/update">{hidden}'
                    f'<input type="hidden" name="code" value="{_esc(code)}">'
                    f'<button class="small secondary" name="action" value="unlink">Unlink</button></form></div>'
                    for code, disp in members) or '<p class="muted">No codes linked yet.</p>'
                acc_rows = ""
                for key, who, email in access:
                    mcp = f"{base}/mcp/study/{key}"
                    page = f"{base}/study/{key}"
                    body = "\n".join([
                        f"Hi {who}," if who else "Hi,", "",
                        f"Your Vagis study connection for \"{sname}\".", "",
                        "1. Study connector address (keep it private, like a password):", mcp, "",
                        "In Claude: Customize > Connectors > + > Add custom connector, name it "
                        f"\"Vagis {sname}\" and paste the address. In a chat, ask e.g. "
                        "\"list the subjects in my Vagis study\".", "",
                        "2. Download page (all data as CSV files in a zip):", page, "", "Jason"])
                    mail = _mailto(email or "", f"Vagis study: {sname}", body, "Email researcher") if email else ""
                    acc_rows += (f'<div style="padding:10px 0;border-bottom:1px solid #f0f0f0">'
                                 f'<b>{_esc(who or "Researcher")}</b> <span class="muted">{_esc(email or "")}</span>'
                                 f'<div class="mono" style="font-size:12px;word-break:break-all;margin-top:6px">'
                                 f'Connector: {_esc(mcp)}<br>Download page: {_esc(page)}</div>'
                                 f'<form class="inline" method="post" action="/admin/ui/study/update">{hidden}'
                                 f'<input type="hidden" name="key" value="{_esc(key)}">'
                                 f'<button class="small secondary" name="action" value="revoke" '
                                 f'style="margin-top:8px">Remove access</button></form> {mail}</div>')
                if not acc_rows:
                    acc_rows = '<p class="muted">No researchers yet.</p>'
                blocks.append(f"""
<div style="border:1px solid #e4e4e4;border-radius:10px;padding:16px;margin-top:16px">
  <h2 style="margin-bottom:6px">{_esc(sname)} <span class="pill">{len(members)} subject(s)</span></h2>
  <h2 style="font-size:14px;margin:14px 0 4px">Subjects</h2>
  {mem_rows}
  <form method="post" action="/admin/ui/study/update" style="display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap">
    {hidden}
    <div style="flex:1;min-width:150px"><label>Vagis code</label><input name="code" placeholder="VG-XXXX-XXXX"></div>
    <div style="flex:1;min-width:150px"><label>Name in study (optional)</label><input name="label" placeholder="Dana or S01"></div>
    <button name="action" value="link" style="margin:0">Link code</button>
  </form>
  <h2 style="font-size:14px;margin:18px 0 4px">Researchers</h2>
  {acc_rows}
  <form method="post" action="/admin/ui/study/update" style="display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap">
    {hidden}
    <div style="flex:1;min-width:150px"><label>Researcher name</label><input name="label" placeholder="Jason"></div>
    <div style="flex:1;min-width:150px"><label>Email</label><input name="email" placeholder="name@example.com"></div>
    <button name="action" value="addresearcher" style="margin:0">Add researcher</button>
  </form>
</div>""")
    finally:
        conn.close()
    return "".join(blocks) or '<p class="sub" style="margin-top:12px">No studies yet.</p>'


def _studies_page(request: Request, token: str, banner: str = "") -> HTMLResponse:
    return HTMLResponse(_admin_page(token, banner, studies_html=_studies_html(request, token)))


@app.post("/admin/ui/studies", response_class=HTMLResponse)
def admin_studies(request: Request, token: str = Form(""), name: str = Form("")) -> HTMLResponse:
    if not _admin_ok(token):
        return HTMLResponse(_admin_page(token, '<div class="err">Admin token did not match.</div>'))
    return _studies_page(request, token)


@app.post("/admin/ui/study/add", response_class=HTMLResponse)
def admin_study_add(request: Request, token: str = Form(""), name: str = Form("")) -> HTMLResponse:
    if not _admin_ok(token):
        return HTMLResponse(_admin_page(token, '<div class="err">Admin token did not match.</div>'))
    name = name.strip()
    if not name:
        return _studies_page(request, token, '<div class="err">Give the study a name.</div>')
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _study_ensure(cur)
            cur.execute("INSERT INTO vagis_studies (name) VALUES (%s);", (name,))
    finally:
        conn.close()
    return _studies_page(request, token,
                         f'<div class="result">Study &ldquo;{_esc(name)}&rdquo; added. '
                         'Link codes and add researchers below.</div>')


@app.post("/admin/ui/study/update", response_class=HTMLResponse)
def admin_study_update(request: Request, token: str = Form(""), study_id: int = Form(0),
                       action: str = Form(""), code: str = Form(""), label: str = Form(""),
                       email: str = Form(""), key: str = Form("")) -> HTMLResponse:
    if not _admin_ok(token):
        return HTMLResponse(_admin_page(token, '<div class="err">Admin token did not match.</div>'))
    banner = ""
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            _study_ensure(cur)
            if action == "link":
                vg = parse_vg_code(code)
                if not vg or not vg_person(cur, vg):
                    banner = '<div class="err">That is not a Vagis code on this server.</div>'
                else:
                    cur.execute("SELECT label FROM vagis_study_members "
                                "WHERE study_id = %s AND person_code = %s;", (study_id, vg))
                    have = cur.fetchone()
                    if have:
                        banner = (f'<div class="err">{_esc(vg)} is already in this study'
                                  f'{" as " + _esc(have[0]) if have[0] else ""}. Nothing changed. '
                                  'Check you pasted the right code.</div>')
                    else:
                        cur.execute("INSERT INTO vagis_study_members (study_id, person_code, label) "
                                    "VALUES (%s, %s, %s);", (study_id, vg, label.strip() or None))
                        banner = (f'<div class="result">{_esc(label.strip() or vg)} '
                                  f'({_esc(vg)}) linked.</div>')
            elif action == "unlink":
                cur.execute("DELETE FROM vagis_study_members WHERE study_id = %s AND person_code = %s;",
                            (study_id, code))
                banner = f'<div class="result">{_esc(code)} unlinked.</div>'
            elif action == "addresearcher":
                cur.execute("INSERT INTO vagis_study_access (key, study_id, researcher, email) "
                            "VALUES (%s, %s, %s, %s);",
                            (secrets.token_urlsafe(24), study_id, label.strip() or None,
                             email.strip() or None))
                banner = '<div class="result">Researcher added. Their addresses are below.</div>'
            elif action == "revoke":
                cur.execute("UPDATE vagis_study_access SET revoked = TRUE WHERE key = %s;", (key,))
                banner = '<div class="result">Access removed. Those addresses no longer work.</div>'
    finally:
        conn.close()
    return _studies_page(request, token, banner)
