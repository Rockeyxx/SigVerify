"""
storage.py — File-based data storage layer

All reads/writes to persistent data go through this file ONLY.
Client visibility is scoped per-employee (each employee sees only the
clients they personally enrolled). A future "Settings" feature could
allow sharing client access between employees — not implemented yet.

All record access uses .get() with safe fallbacks so older/partial
data files never crash the app.

Path safety: files are addressed by a fixed logical name ("employees",
"enrollments", "log"), never by a caller-supplied path, so no external
input can ever reach open().
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


def _load_json(which, default):
    _ensure_data_dir()
    if which == "employees":
        path = EMPLOYEES_FILE
    elif which == "enrollments":
        path = ENROLLMENTS_FILE
    elif which == "log":
        path = LOG_FILE
    else:
        raise ValueError(f"Unknown data file: {which}")
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def _save_json(which, data):
    _ensure_data_dir()
    if which == "employees":
        path = EMPLOYEES_FILE
    elif which == "enrollments":
        path = ENROLLMENTS_FILE
    elif which == "log":
        path = LOG_FILE
    else:
        raise ValueError(f"Unknown data file: {which}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _norm(client_id: str) -> str:
    return client_id.strip().lower()


# ============================================================
# EMPLOYEES
# ============================================================

def load_employees() -> dict:
    return _load_json("employees", {"employees": []})


def find_employee(username: str):
    data = load_employees()
    for emp in data["employees"]:
        if emp.get("username", "").lower() == username.strip().lower():
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
    _save_json("employees", data)
    return True, "Account created successfully"


def authenticate_employee(username: str, password: str) -> tuple:
    emp = find_employee(username)
    if emp is None:
        return False, None, "User ID not found"
    if emp.get("password") != password:
        return False, None, "Incorrect password"
    return True, emp, ""


def update_employee_profile(old_username: str, new_full_name: str,
                              new_username: str, new_password: str = None) -> tuple:
    data = load_employees()

    if new_username.lower() != old_username.lower():
        if find_employee(new_username):
            return False, "This User ID is already taken", None

    for emp in data["employees"]:
        if emp.get("username", "").lower() == old_username.lower():
            emp["full_name"] = new_full_name
            emp["username"] = new_username
            if new_password:
                emp["password"] = new_password
            _save_json("employees", data)
            return True, "Profile updated successfully", emp

    return False, "Employee not found", None


# ============================================================
# ENROLLMENTS (case-insensitive lookup, scoped per employee)
# ============================================================

def load_enrollments() -> dict:
    return _load_json("enrollments", {"clients": {}})


def client_exists(client_id: str, employee_username: str = None) -> bool:
    client = get_client(client_id, employee_username)
    return client is not None


def get_client(client_id: str, employee_username: str = None):
    """
    If employee_username is given, only returns the client if it was
    enrolled by that employee (ownership scoping). Pass None to bypass
    scoping (used internally, e.g. for admin-level lookups).
    """
    data = load_enrollments()
    record = data["clients"].get(_norm(client_id))
    if record is None:
        return None
    if employee_username is not None:
        if record.get("enrolled_by", "").lower() != employee_username.lower():
            return None
    return record


def list_clients(employee_username: str = None) -> list:
    """Returns list of dicts: {key, display_name, ...record}"""
    data = load_enrollments()
    results = []
    for key, record in data["clients"].items():
        if employee_username is not None:
            if record.get("enrolled_by", "").lower() != employee_username.lower():
                continue
        results.append({**record, "key": key})
    return results


def save_client_enrollment(client_id: str, image_refs: list, dynamic_refs: list,
                             employee_username: str, spatial_embeddings: list = None):
    data = load_enrollments()
    key = _norm(client_id)
    existing = data["clients"].get(key, {})
    data["clients"][key] = {
        "display_name": client_id.strip(),
        "enrolled_by": employee_username,
        "image_references": image_refs,          # kept for the View/Preview button
        "dynamic_references": dynamic_refs,
        "spatial_embeddings": spatial_embeddings or [],  # base64 — what verify() actually scores against
        "enrolled_at": existing.get("enrolled_at", datetime.now().isoformat(timespec="seconds")),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_json("enrollments", data)


def delete_client(client_id: str):
    data = load_enrollments()
    key = _norm(client_id)
    if key in data["clients"]:
        del data["clients"][key]
        _save_json("enrollments", data)


def count_clients_enrolled_this_month(employee_username: str = None) -> int:
    now = datetime.now()
    count = 0
    for record in list_clients(employee_username):
        try:
            enrolled_at = datetime.fromisoformat(record.get("enrolled_at", ""))
            if enrolled_at.year == now.year and enrolled_at.month == now.month:
                count += 1
        except ValueError:
            pass
    return count


# ============================================================
# VERIFICATION LOG (scoped per employee)
# ============================================================

def append_verification_log(client_id: str, decision: str, confidence: float,
                              modality_state: str, employee_username: str):
    data = _load_json("log", {"events": []})
    data["events"].append({
        "client_key": _norm(client_id),
        "client_display_name": client_id.strip(),
        "decision": decision,
        "confidence": confidence,
        "modality_state": modality_state,
        "employee": employee_username,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    _save_json("log", data)


def get_verification_log(employee_username: str = None) -> list:
    data = _load_json("log", {"events": []})
    events = data.get("events", [])
    if employee_username is not None:
        events = [e for e in events if e.get("employee", "").lower() == employee_username.lower()]
    return events


def get_last_verification_for_client(client_id: str, employee_username: str = None):
    events = get_verification_log(employee_username)
    key = _norm(client_id)
    matches = [e for e in events if e.get("client_key") == key]
    return matches[-1] if matches else None


def get_verification_stats(employee_username: str = None) -> dict:
    events = get_verification_log(employee_username)
    total = len(events)
    genuine = sum(1 for e in events if e.get("decision") == "Genuine")
    genuine_rate = (genuine / total * 100) if total > 0 else 0.0
    return {"total": total, "genuine": genuine, "genuine_rate": round(genuine_rate, 1)}