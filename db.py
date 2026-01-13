import sqlite3

DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        role TEXT,
        content TEXT
    )
    """)

    conn.commit()
    conn.close()

def load_conversation(cid):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "SELECT role, content FROM chat_history WHERE conversation_id=? ORDER BY id",
        (cid,)
    )

    rows = cur.fetchall()
    conn.close()

    return [{"role": r, "content": c} for r, c in rows]

def save_message(cid, role, content):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO chat_history (conversation_id, role, content) VALUES (?, ?, ?)",
        (cid, role, content)
    )

    conn.commit()
    conn.close()