"""
view_database.py
=================
Read-only inspection script for signatures.db.
Decrypts and prints every stored user, their reference embeddings
(spatial/temporal), and the full audit log — for debugging only.
 
Run from the repository root:
    python -m tools.view_database
"""

import sqlite3
from src.database.db_manager import DatabaseManager
 
 
def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
 
 
def main():
    db = DatabaseManager()
 
    # ── Users + their enrolled references (decrypted) ──────────────
    print_section("USERS & ENROLLED REFERENCES")
 
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, created_at FROM users ORDER BY created_at")
        users = cursor.fetchall()
 
    if not users:
        print("No users enrolled yet.")
    else:
        for user_id, created_at in users:
            print(f"\nUser: {user_id}   (enrolled: {created_at})")
 
            for modality in ("spatial", "temporal"):
                refs = db.get_references(user_id, modality)
                if not refs:
                    print(f"  [{modality}] no references stored")
                    continue
                for i, vec in enumerate(refs):
                    print(
                        f"  [{modality}] reference #{i+1} — "
                        f"shape={vec.shape}, "
                        f"first 5 values={vec[:5]}"
                    )
 
    # ── Audit log (plaintext, queryable — IEEE 730) ─────────────────
    print_section("AUDIT LOG (verification events)")
 
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, modality_state, similarity_score, decision, timestamp
            FROM audit_log
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
 
    if not rows:
        print("No verification events logged yet.")
    else:
        for user_id, modality_state, score, decision, timestamp in rows:
            print(
                f"  [{timestamp}] user={user_id:15s} "
                f"modality={modality_state:10s} "
                f"score={score:.4f}  decision={decision}"
            )
 
    print("\n" + "=" * 60)
    print(f"Total users: {len(users)}   |   Total audit entries: {len(rows)}")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()
 