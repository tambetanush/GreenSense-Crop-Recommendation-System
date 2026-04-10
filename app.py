# =============================================================================
# app.py — GreenSense Flask API
# =============================================================================
# Serves hybrid crop recommendations (ML ranker + cosine similarity +
# range compatibility) and persists new crop registrations and
# recommendation history in SQLite.
#
# Routes
# ──────
#   GET  /                     → index.html
#   POST /predict              → hybrid recommendation
#   POST /register_crop        → add a new crop to the catalog
#   GET  /crops                → list all registered new crops
#   DELETE /crops/<crop_name>  → remove a registered crop
#   POST /explain              → per-dimension explanation for one crop
#   GET  /history              → past recommendation records
#   GET  /history/<int:id>     → single recommendation record
#   DELETE /history            → clear full history
#   GET  /health               → service health check
# =============================================================================
import json
import os
import sqlite3
import traceback
from contextlib import contextmanager
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, g, jsonify, render_template, request

from hybrid_recommender_v2 import HybridCropRecommenderV2
from utils import (ALL_INPUT_COLS, NUMERIC_COLS, RobustLocationImputer,
                   _null_where, clean_and_clip, feature_engineer,
                   normalise_input)

# =============================================================================
# APP SETUP
# =============================================================================

app = Flask(__name__)

DB_PATH = os.environ.get("GREENSENSE_DB",    "greensense.db")
RECOMMENDER_PATH = os.environ.get(
    "RECOMMENDER_PATH", "hybrid_recommender_v2.joblib")
CATALOG_UPDATES = os.environ.get(
    "CATALOG_UPDATES",  "crop_catalog_updates.json")

# =============================================================================
# DATABASE
# =============================================================================


def get_db() -> sqlite3.Connection:
    """Return a per-request SQLite connection (stored on Flask's g object)."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row        # rows behave like dicts
        g.db.execute("PRAGMA journal_mode=WAL")  # safe concurrent reads
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


@contextmanager
def _db_cursor():
    """Convenience context manager for write operations."""
    db = get_db()
    cur = db.cursor()
    try:
        yield cur
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        cur.close()


def init_db():
    """Create tables if they do not yet exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            -- Crops registered via API (not in the original ML training set)
            CREATE TABLE IF NOT EXISTS registered_crops (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_name     TEXT    NOT NULL UNIQUE,
                ph_min        REAL    NOT NULL,
                ph_max        REAL    NOT NULL,
                sun_min       REAL    NOT NULL,
                sun_max       REAL    NOT NULL,
                N_req         REAL    NOT NULL,
                P_req         REAL    NOT NULL,
                K_req         REAL    NOT NULL,
                moisture_min  REAL    NOT NULL,
                moisture_max  REAL    NOT NULL,
                temp_min      REAL    NOT NULL,
                temp_max      REAL    NOT NULL,
                alt_bucket    TEXT    NOT NULL,
                seasons       TEXT    NOT NULL,
                registered_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- Every recommendation the system has produced
            CREATE TABLE IF NOT EXISTS recommendation_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                requested_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                input_features  TEXT    NOT NULL,   -- JSON blob of raw input
                recommendations TEXT    NOT NULL,   -- JSON list of {rank, crop, score, source}
                top_k           INTEGER NOT NULL,
                model_version   TEXT    NOT NULL DEFAULT 'hybrid_v2'
            );
        """)
    print(f"[DB] Initialised at '{DB_PATH}'.")


# =============================================================================
# MODEL LOADING
# =============================================================================

recommender = None


def load_recommender():
    global recommender
    try:
        recommender = joblib.load(RECOMMENDER_PATH)

        # Reload any crops that were registered via the API in previous sessions
        _sync_db_crops_to_recommender()

        print(f"[MODEL] Hybrid recommender loaded from '{RECOMMENDER_PATH}'.")
        print(f"        Known ML labels : {len(recommender.known_crops)}")
        print(f"        Crops in catalog: {len(recommender.catalog)}")
    except FileNotFoundError:
        print(f"[ERROR] '{RECOMMENDER_PATH}' not found. "
              "Run the training notebook and save hybrid_recommender_v2.joblib first.")
        recommender = None


def _sync_db_crops_to_recommender():
    """
    On startup, replay every crop stored in SQLite into the recommender's
    in-memory catalog so the DB is the single source of truth.
    """
    global recommender
    if recommender is None:
        return

    # Import here to avoid circular dependency at module level
    from hybrid_recommender_v2 import register_new_crop

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM registered_crops").fetchall()

    for row in rows:
        try:
            updated = register_new_crop(
                crop_name=row["crop_name"],
                ph_min=row["ph_min"],
                ph_max=row["ph_max"],
                sun_min=row["sun_min"],
                sun_max=row["sun_max"],
                N_req=row["N_req"],
                P_req=row["P_req"],
                K_req=row["K_req"],
                moisture_min=row["moisture_min"],
                moisture_max=row["moisture_max"],
                temp_min=row["temp_min"],
                temp_max=row["temp_max"],
                alt_bucket=row["alt_bucket"],
                seasons=row["seasons"],
                catalog=recommender.catalog,
                persist_path=None,   # DB is the source of truth; skip JSON
            )
            recommender.update_catalog(updated)
        except Exception as exc:
            print(f"[WARN] Could not reload crop '{row['crop_name']}': {exc}")

    if rows:
        print(
            f"[DB] Re-synced {len(rows)} registered crop(s) into recommender.")


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def _to_serialisable(obj):
    """Recursively convert numpy types so json.dumps works."""
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None          # NaN → null in JSON
    return obj


def _save_recommendation(input_features: dict,
                         recommendations: list,
                         top_k: int) -> int:
    """Insert a recommendation record into SQLite and return its id."""
    with _db_cursor() as cur:
        cur.execute(
            """INSERT INTO recommendation_history
               (input_features, recommendations, top_k)
               VALUES (?, ?, ?)""",
            (
                json.dumps(_to_serialisable(input_features)),
                json.dumps(_to_serialisable(recommendations)),
                top_k,
            )
        )
        return cur.lastrowid


# =============================================================================
# ROUTES — PAGES
# =============================================================================

@app.route("/")
def home():
    return render_template("index.html")


# =============================================================================
# ROUTES — PREDICTION
# =============================================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST body (JSON) — raw sensor readings.
    Accepts both new canonical names (N, Env_Temp, …) and legacy names
    (N_kg_per_ha, env_temp_c, …) via normalise_input().

    Optional query param:  ?top_k=5

    Response
    --------
    {
      "status": "ok",
      "record_id": 42,
      "recommendations": [
        {"rank": 1, "crop": "Rice", "rrf_score": 0.02844, "source": "ml+cosine"},
        ...
      ]
    }
    """
    if recommender is None:
        return jsonify({"error": "Recommender not loaded. Check server logs."}), 500

    try:
        raw = request.get_json(force=True)
        top_k = int(request.args.get("top_k", 5))

        # Normalise field names and coerce types
        canonical = normalise_input(raw)

        # Run hybrid recommendation
        recs = recommender.recommend(
            raw_input=canonical,
            top_k=top_k,
            include_scores=True,
        )

        recommendations = [
            {"rank": i + 1, "crop": crop,
             "rrf_score": round(score, 6), "source": source}
            for i, (crop, score, source) in enumerate(recs)
        ]

        # Persist to history
        record_id = _save_recommendation(canonical, recommendations, top_k)

        return jsonify({
            "status":          "ok",
            "record_id":       record_id,
            "recommendations": recommendations,
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed. Check server logs."}), 400


# =============================================================================
# ROUTES — CROP REGISTRATION
# =============================================================================


@app.route("/register_crop", methods=["POST"])
def register_crop():
    """
    Register a new crop that the ML ranker was not trained on.
    The crop is immediately available for content-based recommendations
    and persisted in SQLite so it survives server restarts.

    POST body (JSON)
    ----------------
    {
      "crop_name":    "Quinoa",
      "ph_min":       6.0,  "ph_max":       7.5,
      "sun_min":  50000,    "sun_max":   90000,
      "N_req":      140,    "P_req":        30,    "K_req": 120,
      "moisture_min": 35,   "moisture_max":  60,
      "temp_min":    15,    "temp_max":      25,
      "alt_bucket": "med",
      "seasons":    "winter,summer"
    }
    """
    if recommender is None:
        return jsonify({"error": "Recommender not loaded."}), 500

    try:
        d = request.get_json(force=True)

        # Validate required fields
        required = [
            "crop_name", "ph_min", "ph_max", "sun_min", "sun_max",
            "N_req", "P_req", "K_req", "moisture_min", "moisture_max",
            "temp_min", "temp_max", "alt_bucket", "seasons",
        ]
        missing = [f for f in required if f not in d]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        valid_buckets = {"very_low", "low", "med", "high", "very_high"}
        if d["alt_bucket"] not in valid_buckets:
            return jsonify({
                "error": f"alt_bucket must be one of {sorted(valid_buckets)}"
            }), 400

        crop_name = d["crop_name"].strip()

        # ── 1. Persist to SQLite ──────────────────────────────────────────
        with _db_cursor() as cur:
            cur.execute("""
                INSERT INTO registered_crops
                    (crop_name, ph_min, ph_max, sun_min, sun_max,
                     N_req, P_req, K_req, moisture_min, moisture_max,
                     temp_min, temp_max, alt_bucket, seasons)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(crop_name) DO UPDATE SET
                    ph_min=excluded.ph_min, ph_max=excluded.ph_max,
                    sun_min=excluded.sun_min, sun_max=excluded.sun_max,
                    N_req=excluded.N_req, P_req=excluded.P_req,
                    K_req=excluded.K_req,
                    moisture_min=excluded.moisture_min,
                    moisture_max=excluded.moisture_max,
                    temp_min=excluded.temp_min, temp_max=excluded.temp_max,
                    alt_bucket=excluded.alt_bucket,
                    seasons=excluded.seasons,
                    registered_at=datetime('now')
            """, (
                crop_name,
                float(d["ph_min"]),    float(d["ph_max"]),
                float(d["sun_min"]),   float(d["sun_max"]),
                float(d["N_req"]),     float(d["P_req"]),   float(d["K_req"]),
                float(d["moisture_min"]), float(d["moisture_max"]),
                float(d["temp_min"]),  float(d["temp_max"]),
                d["alt_bucket"], d["seasons"],
            ))

        # ── 2. Update in-memory recommender catalog ───────────────────────
        from hybrid_recommender_v2 import register_new_crop
        updated_catalog = register_new_crop(
            crop_name=crop_name,
            ph_min=float(d["ph_min"]),
            ph_max=float(d["ph_max"]),
            sun_min=float(d["sun_min"]),
            sun_max=float(d["sun_max"]),
            N_req=float(d["N_req"]),
            P_req=float(d["P_req"]),
            K_req=float(d["K_req"]),
            moisture_min=float(d["moisture_min"]),
            moisture_max=float(d["moisture_max"]),
            temp_min=float(d["temp_min"]),
            temp_max=float(d["temp_max"]),
            alt_bucket=d["alt_bucket"],
            seasons=d["seasons"],
            catalog=recommender.catalog,
            persist_path=None,   # SQLite is the source of truth
        )
        recommender.update_catalog(updated_catalog)

        return jsonify({
            "status":  "ok",
            "message": f"'{crop_name}' registered successfully.",
            "catalog_size": len(recommender.catalog),
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Registration failed. Check server logs."}), 400


@app.route("/crops", methods=["GET"])
def list_crops():
    """
    Return all crops registered via the API (not including the original
    21 crops baked into the ML model).

    Optional query params:
      ?search=qui       → filter by name (case-insensitive substring)
      ?limit=50
      ?offset=0
    """
    search = request.args.get("search", "").strip().lower()
    limit = min(int(request.args.get("limit",  100)), 200)
    offset = int(request.args.get("offset", 0))

    db = get_db()
    base = "SELECT * FROM registered_crops"
    args = []

    if search:
        base += " WHERE LOWER(crop_name) LIKE ?"
        args.append(f"%{search}%")

    base += " ORDER BY registered_at DESC LIMIT ? OFFSET ?"
    args += [limit, offset]

    rows = db.execute(base, args).fetchall()
    total = db.execute(
        "SELECT COUNT(*) FROM registered_crops"
        + (" WHERE LOWER(crop_name) LIKE ?" if search else ""),
        ([f"%{search}%"] if search else [])
    ).fetchone()[0]

    return jsonify({
        "status": "ok",
        "total":  total,
        "crops":  [dict(r) for r in rows],
    })


@app.route("/crops/<crop_name>", methods=["DELETE"])
def delete_crop(crop_name: str):
    """
    Remove a registered crop from SQLite and from the recommender catalog.
    Cannot remove the 21 original ML training crops.
    """
    if recommender is None:
        return jsonify({"error": "Recommender not loaded."}), 500

    if crop_name in recommender.known_crops:
        return jsonify({
            "error": f"'{crop_name}' is a core ML-trained crop and cannot be removed."
        }), 400

    with _db_cursor() as cur:
        cur.execute(
            "DELETE FROM registered_crops WHERE crop_name = ?", (crop_name,))
        deleted = cur.rowcount

    if deleted == 0:
        return jsonify({"error": f"'{crop_name}' not found in registered crops."}), 404

    # Remove from in-memory catalog
    updated = recommender.catalog[recommender.catalog["crop"] != crop_name].copy(
    )
    recommender.update_catalog(updated)

    return jsonify({
        "status":  "ok",
        "message": f"'{crop_name}' removed.",
        "catalog_size": len(recommender.catalog),
    })


# =============================================================================
# ROUTES — EXPLANATION
# =============================================================================

@app.route("/explain", methods=["POST"])
def explain():
    """
    Return per-dimension compatibility breakdown for one crop given
    the current sensor readings.

    POST body (JSON)
    ----------------
    {
      "crop_name": "Grapes",
      <...same sensor fields as /predict...>
    }
    """
    if recommender is None:
        return jsonify({"error": "Recommender not loaded."}), 500

    try:
        body = request.get_json(force=True)
        crop_name = body.pop("crop_name", None)

        if not crop_name:
            return jsonify({"error": "'crop_name' is required."}), 400

        canonical = normalise_input(body)
        explanation = recommender.explain(canonical, crop_name)

        return jsonify(_to_serialisable(explanation))

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Explanation failed. Check server logs."}), 400


# =============================================================================
# ROUTES — RECOMMENDATION HISTORY
# =============================================================================

@app.route("/history", methods=["GET"])
def get_history():
    """
    Retrieve past recommendation records.

    Optional query params:
      ?limit=20
      ?offset=0
      ?from=2025-01-01        → ISO date lower bound
      ?to=2025-12-31          → ISO date upper bound
    """
    limit = min(int(request.args.get("limit",   20)), 200)
    offset = int(request.args.get("offset", 0))
    from_ = request.args.get("from")
    to_ = request.args.get("to")

    db = get_db()
    where = []
    args = []

    if from_:
        where.append("requested_at >= ?")
        args.append(from_)
    if to_:
        where.append("requested_at <= ?")
        args.append(to_ + " 23:59:59")

    clause = ("WHERE " + " AND ".join(where)) if where else ""
    rows = db.execute(
        f"SELECT * FROM recommendation_history {clause} "
        f"ORDER BY requested_at DESC LIMIT ? OFFSET ?",
        args + [limit, offset]
    ).fetchall()

    total = db.execute(
        f"SELECT COUNT(*) FROM recommendation_history {clause}", args
    ).fetchone()[0]

    records = []
    for r in rows:
        records.append({
            "id":              r["id"],
            "requested_at":    r["requested_at"],
            "top_k":           r["top_k"],
            "model_version":   r["model_version"],
            "input_features":  json.loads(r["input_features"]),
            "recommendations": json.loads(r["recommendations"]),
        })

    return jsonify({"status": "ok", "total": total, "records": records})


@app.route("/history/<int:record_id>", methods=["GET"])
def get_history_record(record_id: int):
    """Return a single recommendation record by id."""
    db = get_db()
    row = db.execute(
        "SELECT * FROM recommendation_history WHERE id = ?", (record_id,)
    ).fetchone()

    if row is None:
        return jsonify({"error": f"Record {record_id} not found."}), 404

    return jsonify({
        "status":          "ok",
        "id":              row["id"],
        "requested_at":    row["requested_at"],
        "top_k":           row["top_k"],
        "model_version":   row["model_version"],
        "input_features":  json.loads(row["input_features"]),
        "recommendations": json.loads(row["recommendations"]),
    })


@app.route("/history", methods=["DELETE"])
def clear_history():
    """
    Delete ALL recommendation history records.
    Requires confirmation header:  X-Confirm: yes
    """
    if request.headers.get("X-Confirm", "").lower() != "yes":
        return jsonify({
            "error": "Send header 'X-Confirm: yes' to confirm deletion."
        }), 400

    with _db_cursor() as cur:
        cur.execute("DELETE FROM recommendation_history")
        deleted = cur.rowcount

    return jsonify({
        "status":  "ok",
        "message": f"{deleted} history record(s) deleted.",
    })


# =============================================================================
# ROUTES — HEALTH CHECK
# =============================================================================

@app.route("/health", methods=["GET"])
def health():
    db_ok = False
    try:
        get_db().execute("SELECT 1")
        db_ok = True
    except Exception:
        pass

    return jsonify({
        "status":           "ok" if (recommender and db_ok) else "degraded",
        "recommender":      recommender is not None,
        "known_crops":      len(recommender.known_crops) if recommender else 0,
        "catalog_crops":    len(recommender.catalog) if recommender else 0,
        "database":         db_ok,
        "db_path":          DB_PATH,
        "recommender_path": RECOMMENDER_PATH,
        "timestamp":        datetime.utcnow().isoformat() + "Z",
    })


# =============================================================================
# STARTUP
# =============================================================================
GOOGLE_SHEET_API = "https://script.google.com/macros/s/AKfycbyTpysfpeQIB3wNvym2Gk8cx_dPQtLW0cB48RO07K9LpPoWe2hl_iRPpjvWdeVWgmk/exec"


@app.route("/latest-readings", methods=["GET"])
def latest_readings():
    try:
        device_id = request.args.get("ID", "1")
        response = requests.get(GOOGLE_SHEET_API, params={"ID": device_id})

        if response.status_code != 200:
            return jsonify({"error": "Google Script error"}), 500

        return jsonify({
            "device_id": device_id,
            "records": response.json()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


init_db()
load_recommender()

# --- Main Execution ---
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
