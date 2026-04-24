import sqlite3

DB_PATH = "greensense.db"
EMAIL = "ladvinayak813@gmail.com"  # change this

with sqlite3.connect(DB_PATH) as conn:
    conn.execute(
        "UPDATE users SET status = 'approved', role = 'admin' WHERE email = ?",
        (EMAIL,)
    )
    conn.commit()

    row = conn.execute(
        "SELECT uid, email, role, status FROM users WHERE email = ?",
        (EMAIL,)
    ).fetchone()
    print("Updated:", row)