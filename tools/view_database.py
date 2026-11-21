"""
view_database.py
=================
Read-only inspection script for one company's signatures.db (debugging only).
Prints every client with the shape of their decrypted references, and the audit log.

The data key is wrapped by employee passwords, so the script asks for a User ID and
password (SIGVER_DB_PASSPHRASE unlocks a test database instead), after the company.
SIGVER_HOME picks the folder, as in the app.

Run from the repository root:
    python -m tools.view_database
"""

import getpass
import os
import sys

from src.database.db_manager import PASSPHRASE_ENV_VAR, DatabaseManager
from src.gui.storage import DB_FILENAME, _company_folder, list_companies


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    companies = list_companies()
    if not companies:
        sys.exit("No company yet. Run the app first, or set SIGVER_HOME.")
    path = os.path.join(_company_folder(input(f"Company ({', '.join(companies)}): ")), DB_FILENAME)
    if not os.path.exists(path):
        sys.exit(f"No database at {path}. Run the app first, or set SIGVER_HOME.")
    passphrase = os.environ.get(PASSPHRASE_ENV_VAR)
    db = DatabaseManager(path, passphrase=passphrase)
    if not db.unlocked:
        employee, _ = db.authenticate_employee(input("User ID: "), getpass.getpass("Password: "))
        if employee is None:
            sys.exit("Wrong User ID or password.")

    print_section("CLIENTS & REFERENCE SETS (decrypted)")
    clients = db.list_client_records()
    if not clients:
        print("No clients enrolled yet.")
    for rec in clients:
        print(f"\nClient: {rec['key']}   (enrolled by {rec['enrolled_by']} at {rec['enrolled_at']})")
        named_sets = [(f"spatial/{o}", s) for o, s in rec["sets"].items()] + [("temporal", rec["temporal_set"])]
        for name, found in named_sets:
            if not found:
                print(f"  [{name}] none")
                continue
            print(f"  [{name}] {len(found['embeddings'])} x {found['embeddings'][0].shape[0]}-d, "
                  f"model {', '.join(found['model_versions'])}, device {found['capture_device']}")

    print_section("AUDIT LOG (plaintext, IEEE 730)")
    events = db.get_log_events(event=None)
    if not events:
        print("No events logged yet.")
    for e in events:
        score = "" if e["confidence"] is None else f" score={e['confidence']:.1f}"
        print(f"  [{e['timestamp']}] {e['event']:16s} client={e['client_key']} "
              f"by={e['employee']} decision={e['decision']}{score}")

    print("\n" + "=" * 60)
    print(f"Total clients: {len(clients)}   |   Total audit entries: {len(events)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
