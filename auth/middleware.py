from functools import wraps
from flask import request, jsonify
from firebase_admin import auth
import sqlite3
import os

DB_PATH = os.environ.get("GREENSENSE_DB", "data/greensense.db")


def _get_user_record(uid: str):
    """Fetch role and status from local DB for a given Firebase UID."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT role, status FROM users WHERE uid = ?", (uid,)
            ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None


def verify_firebase_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return jsonify({"error": "No token provided"}), 401

        try:
            token = auth_header.split(" ")[1]
            decoded_token = auth.verify_id_token(token)

            # Require email verification
            if not decoded_token.get("email_verified", False):
                return jsonify({"error": "Email not verified"}), 403

            uid = decoded_token["uid"]
            db_user = _get_user_record(uid)

            if db_user is None:
                return jsonify({"error": "User record not found. Please register first."}), 403

            if db_user["status"] != "approved":
                return jsonify({"error": "Account pending admin approval"}), 403

            # Attach role/status to the decoded token on request
            decoded_token["role"] = db_user["role"]
            decoded_token["status"] = db_user["status"]
            request.user = decoded_token

        except Exception as e:
            print("Auth error:", e)
            return jsonify({"error": "Invalid or expired token"}), 401

        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = getattr(request, "user", None)

        if not user or user.get("role") != "admin":
            return jsonify({"error": "Admin only"}), 403

        return f(*args, **kwargs)

    return decorated_function