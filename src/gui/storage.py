"""
storage.py — File-based data storage layer

All reads/writes to persistent data go through this file ONLY.
Currently backed by local JSON files under data/. When the database
team finishes SQLite + AES-256-GCM encryption (rules.md §8), only
this file's function bodies change — nothing in app.py, enrollment.py,
verification.py, sidebar.py needs to change.

SECURITY NOTE: passwords are stored as plain text here — this is a
MOCK ONLY, standing in until the real encrypted DB layer replaces it.
"""

import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

EMPLOYEES_FILE = os.path.join(DATA_DIR, "employees.json")
ENROLLMENTS_FILE = os.path.join(DATA_DIR, "enrollments.json")
LOG_FILE = os.path.join(DATA_DIR, "verification_log.json")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_json(filepath, default):
    _ensure_data_dir()
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def _save_json(filepath, data):
    _ensure_data_dir()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# EMPLOYEES (login / register)
# ============================================================

def load_employees() -> dict:
    return _load_json(EMPLOYEES_FILE, {"employees": []})


def find_employee(username: str):
    data = load_employees()
    for emp in data["employees"]:
        if emp["username"] == username:
            return emp
    return None


def register_employee(full_name: str, username: str, password: str) -> tuple:
    if find_employee(username):
        return False, "This User ID already exists"

    data = load_employees()
    data["employees"].append({
        "full_name": full_name,
        "username": username,
        "password": password,  # NOTE: plain text mock only
    })
    _save_json(EMPLOYEES_FILE, data)
    return True, "Account created successfully"


def authenticate_employee(username: str, password: str) -> tuple:
    """Returns: (success, employee_dict_or_None, message)"""
    emp = find_employee(username)
    if emp is None:
        return False, None, "User ID not found"
    if emp["password"] != password:
        return False, None, "Incorrect password"
    return True, emp, ""


# ============================================================
# ENROLLMENTS (clients + their reference signatures)
# ============================================================

def load_enrollments() -> dict:
    return _load_json(ENROLLMENTS_FILE, {"clients": {}})


def client_exists(client_id: str) -> bool:
    data = load_enrollments()
    return client_id in data["clients"]


def get_client(client_id: str):
    data = load_enrollments()
    return data["clients"].get(client_id)


def list_clients() -> list:
    data = load_enrollments()
    return list(data["clients"].keys())


def save_client_enrollment(client_id: str, image_refs: list, dynamic_refs: list, employee_username: str):
    data = load_enrollments()
    data["clients"][client_id] = {
        "client_name": client_id,
        "enrolled_by": employee_username,
        "image_references": image_refs,
        "dynamic_references": dynamic_refs,
        "enrolled_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_json(ENROLLMENTS_FILE, data)


def delete_client(client_id: str):
    data = load_enrollments()
    if client_id in data["clients"]:
        del data["clients"][client_id]
        _save_json(ENROLLMENTS_FILE, data)


# ============================================================
# VERIFICATION LOG (audit trail — IEEE 730, rules.md §8)
# ============================================================

def append_verification_log(client_id: str, decision: str, confidence: float,
                              modality_state: str, employee_username: str):
    data = _load_json(LOG_FILE, {"events": []})
    data["events"].append({
        "client_id": client_id,
        "decision": decision,
        "confidence": confidence,
        "modality_state": modality_state,
        "employee": employee_username,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    _save_json(LOG_FILE, data)


def get_verification_log() -> list:
    data = _load_json(LOG_FILE, {"events": []})
    return data["events"]