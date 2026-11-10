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
 
# The third decision of step-up verification (src.verification.SIGN_AGAIN).
# Restated here so storage never imports PyTorch. It is not a verdict, so it is
# never logged and never counted (AGENTS.md, step-up rules).
SIGN_AGAIN = "Sign again"
FINAL_DECISIONS = ("Genuine", "Forged")
 
# Cap on spatial references per client, enrolled + added (AGENTS.md: the measured
# arms held 6 to 8).
MAX_SPATIAL_REFERENCES = 8
 
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
        with open(key_path, "r", encoding="utf-8") as f:  # skylos: ignore[SKY-D215,SKY-D325] key file in the fixed data folder (SIGVER_HOME set by the operator)
            return f.read().strip()
    if os.path.exists(db_path):
        raise RuntimeError(
            f"The database {db_path} exists but its key file {key_path} is missing "
            f"(and {PASSPHRASE_ENV_VAR} is not set). Restore the key file; a new key "
            f"is not created because it could not open the stored templates."
        )
    passphrase = secrets.token_hex(32)
    try:
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # skylos: ignore[SKY-D215,SKY-D324] O_EXCL refuses an existing file or symlink
    except FileExistsError:  # another process created it first
        with open(key_path, "r", encoding="utf-8") as f:  # skylos: ignore[SKY-D215,SKY-D325] key file in the fixed data folder (SIGVER_HOME set by the operator)
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
                raise RuntimeError(
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
 
def _client_to_gui(rec: dict) -> dict:
    return {
        "display_name": rec["display_name"],
        "enrolled_by": rec["enrolled_by"],
        "image_references": list(rec["image_references"]),
        "dynamic_references": [_b64(a) for a in rec["temporal"]],
        "spatial_embeddings": [_b64(a) for a in rec["spatial"]],
        "spatial_origin": rec["spatial_origin"],
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
    save is refused and nothing is written. Re-enrolling your own client replaces
    all its templates and keeps enrolled_at. The whole save is one transaction.
 
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
 
 
def delete_client(client_id: str):
    """Removes the client and all its templates. The verification log is kept."""
    _get_db().delete_user(_norm(client_id))
 
 
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
                            modality_state: str, employee_username: str):
    if decision not in FINAL_DECISIONS:
        # "Sign again" is a request, not a verdict: one event per visit, final verdict only.
        return
    emp_id = _employee_id(employee_username or "")
    _get_db().log_verification(
        _norm(client_id), modality_state, confidence, decision,
        employee_id=emp_id, display_name=client_id.strip(),
        employee_label=None if emp_id is not None else employee_username,
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
 
 
# ============================================================
# REFERENCES (enrolled + confirmed signings added later)
# ============================================================
 
def get_reference_embeddings(client_id: str, employee_username: str = None) -> tuple:
    """Returns (base64 embeddings, spatial_origin): enrolled references first,
    then confirmed signings added later. The database keeps added signings in the
    client's own origin set, so a spatial probe is scored only against its origin."""
    record = get_client(client_id, employee_username)
    if record is None:
        return [], "scan"
    return list(record["spatial_embeddings"]), record["spatial_origin"] or "scan"
 
 
def add_confirmed_reference(client_id: str, embedding_b64: str, employee_username: str) -> bool:
    """Append one confirmed signing's spatial embedding to the client's references.
 
    Call it only when decision == "Genuine", degraded is False, and the employee
    confirmed the person's identity (AGENTS.md). Stores the embedding only,
    encrypted, in the client's spatial_origin set. When the cap is reached the
    OLDEST ADDED signing is dropped; the enrolled references are never dropped.
    """
    key = _norm(client_id or "")
    emp_id = _employee_id(employee_username or "")
    if not key or emp_id is None:
        return False
    db = _get_db()
    record = db.get_client_record(key, emp_id)
    if record is None or record["spatial_origin"] is None:
        return False
    raw = _decode_embeddings([embedding_b64], "Confirmed signing")[0]
    db.add_confirmed_reference(key, "spatial", raw, MODEL_VERSION,
                               spatial_origin=record["spatial_origin"],
                               max_per_set=MAX_SPATIAL_REFERENCES)
    return True
