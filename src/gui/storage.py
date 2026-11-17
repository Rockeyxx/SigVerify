"""
storage.py - persistent data layer for the GUI (SQLite + AES-256-GCM).

All reads/writes to persistent data go through this file ONLY. The function names,
arguments, return values and dict keys are the ones the GUI pages have always used,
so no page changes. Underneath, everything lives in one SQLite database managed by
src.database.db_manager.DatabaseManager:

  * employee passwords: random salt + scrypt hash, compared in constant time
  * signature embeddings (spatial 512-d, temporal 128-d) and the image-reference
    list: AES-256-GCM, bound to the owning client with associated data
  * every save is one SQLite transaction (no half-written or wiped files)

Client visibility is scoped per employee: each employee sees only the clients they
enrolled. Ownership is stored by employee id, so renaming a username never hides
that employee's clients or log entries.

Where the data lives (never relative to the working directory):
  ~/.sigverify/signatures.db   the database (override the folder with SIGVER_HOME)
  ~/.sigverify/db.key          random passphrase for the data key, created once with
                               mode 0600. SIGVER_DB_PASSPHRASE overrides the file.
The key file sits next to the database, never inside it. If the database exists but
the key is missing, startup fails loudly; a new key is never made silently.

Only embeddings are stored. `image_references` is a list of file PATHS the employee
picked (kept for the View button), never image data; it is encrypted anyway because
file names can contain the client's name.
"""

import base64
import binascii
import os
import secrets
import threading
from datetime import datetime

from cryptography.exceptions import InvalidTag

from src.database.db_manager import DatabaseManager, PASSPHRASE_ENV_VAR

try:
    from config import settings as _settings
    MODEL_VERSION = getattr(_settings, "MODEL_VERSION", "unspecified")
except ImportError:  # pragma: no cover - only outside the repo
    MODEL_VERSION = "unspecified"

DB_FILENAME = "signatures.db"
KEY_FILENAME = "db.key"
HOME_ENV_VAR = "SIGVER_HOME"

class StorageKeyError(RuntimeError):
    """The key file / passphrase is missing or does not match the database.
    The GUI should catch this and show a dialog; a new key is never created."""


_db = None
_db_lock = threading.Lock()


# ============================================================
# Database location and key
# ============================================================

def _data_dir() -> str:
    override = os.environ.get(HOME_ENV_VAR)
    base = override if override else os.path.join("~", ".sigverify")
    return os.path.abspath(os.path.expanduser(base))


def _read_or_create_passphrase(data_dir: str, db_path: str) -> str:
    env = os.environ.get(PASSPHRASE_ENV_VAR)
    if env:
        return env
    key_path = os.path.join(data_dir, KEY_FILENAME)
    if os.path.exists(key_path):
        with open(key_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    if os.path.exists(db_path):
        raise StorageKeyError(
            f"The database {db_path} exists but its key file {key_path} is missing "
            f"(and {PASSPHRASE_ENV_VAR} is not set). Restore the key file; a new key "
            f"is not created because it could not open the stored templates."
        )
    passphrase = secrets.token_hex(32)
    try:
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # another process created it first
        with open(key_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(passphrase)
    return passphrase


def _get_db() -> DatabaseManager:
    global _db
    if _db is not None:
        return _db
    with _db_lock:
        if _db is None:
            data_dir = _data_dir()
            os.makedirs(data_dir, mode=0o700, exist_ok=True)
            db_path = os.path.join(data_dir, DB_FILENAME)
            passphrase = _read_or_create_passphrase(data_dir, db_path)
            try:
                _db = DatabaseManager(db_path, passphrase=passphrase)
            except InvalidTag:
                raise StorageKeyError(
                    f"The key does not match the database {db_path}. Check {KEY_FILENAME} "
                    f"or {PASSPHRASE_ENV_VAR}."
                ) from None
    return _db


def _reset():
    """Forget the open database handle (tests, or after SIGVER_HOME changes)."""
    global _db
    with _db_lock:
        _db = None


def _norm(client_id: str) -> str:
    return client_id.strip().lower()


def _employee_id(username):
    emp = _get_db().find_employee(username)
    return None if emp is None else emp["id"]


def _public_employee(emp: dict) -> dict:
    """What the GUI sees of an employee: never the salt or the hash."""
    return {"full_name": emp["full_name"], "username": emp["username"]}


def _b64(arr) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


# ============================================================
# EMPLOYEES
# ============================================================

def find_employee(username: str):
    emp = _get_db().find_employee(username)
    return None if emp is None else _public_employee(emp)


def register_employee(full_name: str, username: str, password: str) -> tuple:
    try:
        _get_db().create_employee(full_name, username, password)
    except ValueError as e:
        return False, str(e)
    return True, "Account created successfully"


def authenticate_employee(username: str, password: str) -> tuple:
    emp, reason = _get_db().authenticate_employee(username, password)
    if emp is None:
        if reason == "not_found":
            return False, None, "User ID not found"
        return False, None, "Incorrect password"
    return True, _public_employee(emp), ""


def update_employee_profile(old_username: str, new_full_name: str,
                            new_username: str, new_password: str = None) -> tuple:
    try:
        emp = _get_db().update_employee(old_username, new_full_name, new_username, new_password)
    except LookupError:
        return False, "Employee not found", None
    except ValueError as e:
        return False, str(e), None
    return True, "Profile updated successfully", _public_employee(emp)


# ============================================================
# ENROLLMENTS (case-insensitive lookup, scoped per employee)
# ============================================================

def _set_to_gui(found):
    if found is None:
        return None
    return {
        "embeddings": [_b64(a) for a in found["embeddings"]],
        "model_version": found["model_version"],
        "enrolled_at": found["enrolled_at"],
        "stale": found["model_version"] != MODEL_VERSION,  # other weights: ask for re-enrolment
    }


def _client_to_gui(rec: dict) -> dict:
    """
    Old keys are kept for the existing pages (spatial_embeddings / spatial_origin show the
    "scan" set, or the "render" set when there is no scan set). New keys:
      spatial_sets: {"scan": {...}, "render": {...}}  (a missing origin is absent)
      temporal_set: {...} or None
    each set = {"embeddings": [base64], "model_version", "enrolled_at", "stale"}.
    """
    return {
        "display_name": rec["display_name"],
        "enrolled_by": rec["enrolled_by"],
        "image_references": list(rec["image_references"]),
        "dynamic_references": [_b64(a) for a in rec["temporal"]],
        "spatial_embeddings": [_b64(a) for a in rec["spatial"]],
        "spatial_origin": rec["spatial_origin"],
        "spatial_sets": {o: _set_to_gui(v) for o, v in rec["sets"].items()},
        "temporal_set": _set_to_gui(rec["temporal_set"]),
        "enrolled_at": rec["enrolled_at"],
        "updated_at": rec["updated_at"],
    }


def client_exists(client_id: str, employee_username: str = None) -> bool:
    emp_id = None
    if employee_username is not None:
        emp_id = _employee_id(employee_username)
        if emp_id is None:
            return False
    return _get_db().has_client(_norm(client_id), emp_id)


def get_client(client_id: str, employee_username: str = None):
    """
    If employee_username is given, only returns the client if it was enrolled by
    that employee (ownership scoping). Pass None to look across all employees
    (the Enrollment page uses this to detect an ID owned by someone else).
    Embeddings are decrypted here; the caller gets plain base64 values.
    """
    emp_id = None
    if employee_username is not None:
        emp_id = _employee_id(employee_username)
        if emp_id is None:
            return None
    rec = _get_db().get_client_record(_norm(client_id), emp_id)
    return None if rec is None else _client_to_gui(rec)


def list_clients(employee_username: str = None) -> list:
    """Returns list of dicts: {key, display_name, ...record}"""
    emp_id = None
    if employee_username is not None:
        emp_id = _employee_id(employee_username)
        if emp_id is None:
            return []
    return [{**_client_to_gui(r), "key": r["key"]} for r in _get_db().list_client_records(emp_id)]


def _decode_embeddings(values, what: str) -> list:
    out = []
    for i, value in enumerate(values or []):
        try:
            out.append(base64.b64decode(value, validate=True))
        except (binascii.Error, ValueError, TypeError):
            raise ValueError(f"{what} #{i + 1} is not valid base64") from None
    return out


def save_client_enrollment(client_id: str, image_refs: list, dynamic_refs: list,
                           employee_username: str, spatial_embeddings: list = None,
                           spatial_origin: str = "scan") -> tuple:
    """
    spatial_embeddings: base64 float32 vectors (512 values each).
    dynamic_refs: base64 float32 temporal vectors (128 values each), or empty.
    spatial_origin ("scan" or "render") records how the spatial embeddings were
    made; verify() needs it, so it is stored with every template.

    Ownership check: if the client already exists under a DIFFERENT employee the
    save is refused and nothing is written. Re-enrolling replaces only the sets that
    are given: spatial_embeddings replace the `spatial_origin` set, dynamic_refs
    replace the temporal set; an empty list leaves that set as it was, so enrolling
    by tablet does not wipe the scan set. enrolled_at is kept. One transaction.

    Returns (success: bool, message: str).
    """
    key = _norm(client_id or "")
    if not key:
        return False, "Client ID is required"
    if "|" in key:
        return False, "Client ID must not contain '|'"
    emp_id = _employee_id(employee_username or "")
    if emp_id is None:
        return False, "Employee not found"

    try:
        spatial = _decode_embeddings(spatial_embeddings, "Spatial embedding")
        temporal = _decode_embeddings(dynamic_refs, "Dynamic reference")
        _get_db().save_client(
            key, client_id.strip(), emp_id, image_refs or [], spatial, spatial_origin,
            temporal, MODEL_VERSION,
        )
    except PermissionError:
        return False, (
            f"A client named '{client_id}' is already enrolled by another "
            f"employee. Choose a different ID."
        )
    except ValueError as e:
        return False, str(e)
    return True, "Saved"


def delete_client(client_id: str, employee_username: str = None) -> bool:
    """
    Removes the client and all its templates; the verification log is kept.
    With employee_username, another employee's client is refused (returns False).
    Without it nothing is checked (the old call); pass it from the GUI.
    """
    if employee_username is None:
        _get_db().delete_user(_norm(client_id))
        return True
    emp_id = _employee_id(employee_username)
    if emp_id is None:
        return False
    return _get_db().delete_client_owned(_norm(client_id), emp_id)


def count_clients_enrolled_this_month(employee_username: str = None) -> int:
    emp_id = None
    if employee_username is not None:
        emp_id = _employee_id(employee_username)
        if emp_id is None:
            return 0
    return _get_db().count_clients_enrolled_in_month(datetime.now().strftime("%Y-%m"), emp_id)


# ============================================================
# VERIFICATION LOG (scoped per employee)
# ============================================================

def append_verification_log(client_id: str, decision: str, confidence: float,
                            modality_state: str, employee_username: str,
                            routing: str = None, degraded: bool = None, reasons: list = None,
                            model_version: str = None, threshold: float = None):
    """routing / degraded / reasons / threshold are what verify() returned; model_version
    is the tag of the references used. All optional so the old call still works."""
    if decision == "Sign again":  # step-up prompt, not a final verdict
        return
    emp_id = _employee_id(employee_username or "")
    _get_db().log_verification(
        _norm(client_id), modality_state, confidence, decision,
        employee_id=emp_id, display_name=client_id.strip(),
        employee_label=None if emp_id is not None else employee_username,
        routing=routing, degraded=degraded, reasons=reasons,
        model_version=model_version, threshold=threshold,
    )


def get_verification_log(employee_username: str = None) -> list:
    emp_id = None
    if employee_username is not None:
        emp_id = _employee_id(employee_username)
        if emp_id is None:
            return []
    return _get_db().get_log_events(employee_id=emp_id)


def get_last_verification_for_client(client_id: str, employee_username: str = None):
    emp_id = None
    if employee_username is not None:
        emp_id = _employee_id(employee_username)
        if emp_id is None:
            return None
    events = _get_db().get_log_events(
        employee_id=emp_id, user_id=_norm(client_id), newest_first=True, limit=1)
    return events[0] if events else None


def get_verification_stats(employee_username: str = None) -> dict:
    emp_id = None
    if employee_username is not None:
        emp_id = _employee_id(employee_username)
        if emp_id is None:
            return {"total": 0, "genuine": 0, "genuine_rate": 0.0}
    total, genuine = _get_db().count_log_events(emp_id)
    genuine_rate = (genuine / total * 100) if total > 0 else 0.0
    return {"total": total, "genuine": genuine, "genuine_rate": round(genuine_rate, 1)}
