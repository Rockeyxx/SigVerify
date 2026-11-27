"""GUI persistence layer: per-company encrypted SQLite vaults under ~/.sigverify/companies/<slug>/."""

import base64
import binascii
import os
import re
import shutil
import sqlite3
import threading
import unicodedata
from datetime import datetime

from cryptography.exceptions import InvalidTag

from config import settings as _settings
from src.database.db_manager import (
    EVENT_BACKUP,
    EVENT_DELETE,
    FINAL_DECISIONS,
    PASSPHRASE_ENV_VAR,
    DatabaseLockedError,
    DatabaseManager,
    VaultFileError,
    VaultInUseError,
    probe_fingerprint,  # noqa: F401 (re-exported for GUI replay checks)
)

MODEL_VERSION = _settings.MODEL_VERSION
MAX_REFERENCES = _settings.MAX_REFERENCES

DB_FILENAME = "signatures.db"
COMPANIES_DIRNAME = "companies"
COMPANY_NAME_FILE = "company.txt"
HOME_ENV_VAR = "SIGVER_HOME"

# The third decision of step-up verification (src.verification.SIGN_AGAIN).
# Restated here so storage never imports PyTorch. It is not a verdict, so it is
# never logged and never counted (AGENTS.md, step-up rules).
SIGN_AGAIN = "Sign again"

ADMIN_WARNING = (
    "Passwords are not stored anywhere. If the owner and every member forget their "
    "passwords, the enrolled signatures cannot be recovered and every signer must enrol again."
)


class StorageError(RuntimeError):
    """A storage problem the GUI shows to the employee instead of crashing."""


class StorageKeyError(StorageError):
    """The database cannot be opened with what was given (wrong or missing key).
    A new key is never created to get past it."""


class StorageLockedError(StorageError):
    """Nobody is logged in, so the templates cannot be decrypted."""


_company = None                 # the chosen company's name (as in its company.txt)
_db = None
_session = None                 # the logged-in employee: {"id", "username", "full_name", "is_admin"}
_unlocked_by_passphrase = False
_db_lock = threading.Lock()


# ============================================================
# Database location, opening, session
# ============================================================

def _data_dir() -> str:
    override = os.environ.get(HOME_ENV_VAR)
    base = override if override else os.path.join("~", ".sigverify")
    return os.path.abspath(os.path.expanduser(base))


# Names Windows keeps for devices: a folder with one of them as its whole name cannot be made.
_WINDOWS_DEVICE_NAMES = {"con", "prn", "aux", "nul", *(f"{p}{i}" for p in ("com", "lpt") for i in range(1, 10))}


def _slug(name: str) -> str:
    """Letters, marks and digits of any script (Unicode categories L, M, N), case-folded;
    every other run is one "-" (so no "/", "." or ".." survives: the folder stays inside
    companies/). Marks are kept so diacritics and vowel signs stay inside their word."""
    text = unicodedata.normalize("NFKC", name.strip()).casefold()
    kept = "".join(c if unicodedata.category(c)[0] in "LMN" else "-" for c in text)
    return re.sub(r"-+", "-", kept).strip("-")


def _companies_dir() -> str:
    return os.path.join(_data_dir(), COMPANIES_DIRNAME)


def _company_folder(name: str) -> str:
    return os.path.join(_companies_dir(), _slug(name))


def _read_company_name(slug: str):
    """The name in companies/<slug>/company.txt, or None (no such company)."""
    if not slug:
        return None
    try:
        with open(os.path.join(_companies_dir(), slug, COMPANY_NAME_FILE), encoding="utf-8") as f:  # skylos: ignore[SKY-D215,SKY-D325] slug is reduced to letters, digits and '-', inside the company folder; read is bounded
            return f.read(200).strip()
    except OSError:
        return None


def _get_db() -> DatabaseManager:
    global _db, _unlocked_by_passphrase
    if _db is not None:
        return _db
    with _db_lock:
        if _db is None:
            if _company is None:
                raise StorageLockedError("Choose a vault first.")
            data_dir = _company_folder(_company)
            db_path = os.path.join(data_dir, DB_FILENAME)
            try:
                db = DatabaseManager(db_path)        # opens locked; a login unlocks it
            except RuntimeError as e:                # a schema this code cannot read
                if "is older than" in str(e):
                    raise StorageError(
                        "This database was made by an older version of the app and cannot be opened. "
                        f"Move the folder {data_dir} somewhere else and start the app again; "
                        "signers must enrol again.") from None
                raise StorageError(str(e)) from None
            env = os.environ.get(PASSPHRASE_ENV_VAR)
            try:
                if env:
                    db.unlock_with_passphrase(env)
            except InvalidTag:
                raise StorageKeyError(f"{PASSPHRASE_ENV_VAR} does not match the database {db_path}.") from None
            except ValueError as e:
                raise StorageKeyError(str(e)) from None
            _unlocked_by_passphrase = bool(env)
            _db = db
    return _db


def _unlocked_db() -> DatabaseManager:
    db = _get_db()
    if not db.unlocked:
        raise StorageLockedError("Log in first: the signature data is locked.")
    return db


def _reset(company=None):
    """Forget the open database handle, the session and the chosen company, as a fresh start
    does (tests, or after SIGVER_HOME changes); `company` is chosen instead, unchecked."""
    global _db, _session, _company
    with _db_lock:
        _db = None
        _session = None
        _company = company


def _select(company) -> None:
    logout()                      # lock the old data key before dropping its handle
    _reset(company)


def logout() -> None:
    """Forget who is logged in and the data key, so nothing can be decrypted until the next login."""
    global _session
    _session = None
    if _db is not None and not _unlocked_by_passphrase:
        _db.lock()


def _norm(client_id: str) -> str:
    return client_id.strip().lower()


def _employee_id(username):
    emp = _unlocked_db().find_employee(username)
    return None if emp is None else emp["id"]


class NotLoggedInAsError(PermissionError):
    """A write named an employee other than the one logged in."""


def _actor(employee_username: str = None) -> int:
    """
    The id of the employee doing a write: always the LOGGED-IN employee. The username a
    page passes is only a cross-check; a name that is not the session's is refused
    (NotLoggedInAsError), so nobody acts, or is logged, as somebody else (the admin above
    all). Raises StorageLockedError when nobody is logged in.
    """
    if _session is None:
        raise StorageLockedError("Log in first.")
    if employee_username is not None and _norm(employee_username) != _norm(_session["username"]):
        raise NotLoggedInAsError(
            f"You are logged in as {_session['username']}, not {employee_username.strip()}.")
    emp = _get_db().find_employee(_session["username"])
    if emp is None or emp["id"] != _session["id"]:          # removed meanwhile
        raise StorageLockedError("This account no longer exists. Log in again.")
    return emp["id"]


def _public_employee(emp: dict) -> dict:
    """What the GUI sees of an employee: never a key or a password."""
    return {"full_name": emp["full_name"], "username": emp["username"], "is_admin": emp["is_admin"]}


def _b64(arr) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


# ============================================================
# COMPANIES (one folder, database and data key each)
# ============================================================

def list_companies() -> list:
    """The company names, sorted without regard to case."""
    try:
        slugs = os.listdir(_companies_dir())
    except FileNotFoundError:
        return []
    except OSError as e:                     # not a folder, no permission
        raise StorageError(f"The vault folder {_companies_dir()} cannot be read: {e}") from None
    names = [_read_company_name(slug) for slug in slugs]
    return sorted((n for n in names if n), key=str.casefold)


def current_company():
    return _company


def open_company(company_name: str) -> None:
    """Choose the company later calls work on; nobody is logged in afterwards.
    Raises StorageError for an unknown name (the chosen company stays as it was)."""
    name = _read_company_name(_slug(company_name or ""))
    if name is None:
        raise StorageError(f"No vault named '{(company_name or '').strip()}'.")
    _select(name)


def _new_company(company_name: str, fill) -> tuple:
    """Make the company's folder, choose it and run fill() -> (ok, message) inside it, then
    record the name in its database and in company.txt. The company is listed (company.txt)
    only once fill() made its admin, so a crash before that lists no company nobody can log
    into. On any failure the folder is removed and the company chosen before is chosen again
    (logged out)."""
    name = (company_name or "").strip()
    if not 2 <= len(name) <= 40 or not _slug(name):
        return False, "The vault name must have 2 to 40 characters, with at least one letter or digit."
    if _slug(name) in _WINDOWS_DEVICE_NAMES:
        return False, f"'{name}' is a name Windows keeps for a device. Choose another vault name."
    folder = _company_folder(name)
    if os.path.isdir(folder) and not os.path.exists(os.path.join(folder, COMPANY_NAME_FILE)):
        # Left by a crash during registration: unlisted, nobody can log into it.
        shutil.rmtree(folder, ignore_errors=True)
    try:
        os.makedirs(_data_dir(), mode=0o700, exist_ok=True)
        os.makedirs(_companies_dir(), mode=0o700, exist_ok=True)
        os.mkdir(folder, 0o700)
    except FileExistsError:
        return False, "A vault with this name already exists"
    except OSError as e:
        return False, f"The vault folder could not be made: {e}"
    previous, done = _company, False
    try:
        _select(name)
        ok, message = fill()
        if ok:
            _get_db().set_company_name(name)
            with open(os.path.join(folder, COMPANY_NAME_FILE), "w", encoding="utf-8") as f:
                f.write(name)
            done = True
    except (OSError, sqlite3.Error, StorageError) as e:   # disk full, unreadable folder
        message = str(e)
    finally:
        if not done:
            _select(previous)     # drop the handle before removing its file
            shutil.rmtree(folder, ignore_errors=True)
    return done, message


def create_company(company_name: str, full_name: str, username: str, password: str) -> tuple:
    """Register a new company: its folder, database and data key, and its first employee,
    the admin, who is logged in. Returns (success, message)."""
    return _new_company(company_name, lambda: register_employee(full_name, username, password))


def restore_company(company_name: str, backup_path: str, username: str, password: str) -> tuple:
    """A new company made from a backup, with the checks of restore_database (`username` /
    `password` must be an admin of the BACKUP, who is logged in). Returns (success, message)."""
    return _new_company(company_name, lambda: restore_database(backup_path, username, password))


# ============================================================
# EMPLOYEES
# ============================================================

def is_first_run() -> bool:
    """True before any company exists: the start screen is Register a new company."""
    return not list_companies()


def current_employee():
    """The logged-in employee as the GUI sees them, or None."""
    return None if _session is None else _public_employee(_session)


def find_employee(username: str):
    emp = _unlocked_db().find_employee(username)
    return None if emp is None else _public_employee(emp)


def list_employees() -> list:
    return [_public_employee(e) for e in _unlocked_db().list_employees()]


def register_employee(full_name: str, username: str, password: str) -> tuple:
    """The first account of a company is its admin (and is logged in); create_company makes
    it. After that only a logged-in admin adds employees. Returns (success, message)."""
    global _session
    db = _get_db()
    first = not db.has_employees()
    creator = _session["id"] if _session else None
    try:
        emp_id = db.create_employee(full_name, username, password, created_by=creator)
    except PermissionError:
        return False, "Only the owner can add members. Ask the owner to add your account."
    except DatabaseLockedError:
        return False, "Log in as the owner to add members."
    except (ValueError, LookupError, RuntimeError) as e:
        return False, str(e)
    if first:
        _session = {"id": emp_id, "username": username.strip(), "full_name": full_name, "is_admin": True}
        return True, f"Owner account created. {ADMIN_WARNING}"
    return True, "Account created successfully"


def authenticate_employee(username: str, password: str) -> tuple:
    """Logging in unlocks the signature data. Returns (success, employee or None, message)."""
    global _session
    try:
        emp, reason = _get_db().authenticate_employee(username, password)
    except (VaultFileError, VaultInUseError) as e:      # a changed file, or the vault open elsewhere
        return False, None, str(e)
    if emp is None:
        return False, None, "User ID not found" if reason == "not_found" else "Incorrect password"
    _session = emp
    _get_db().purge_expired(emp["id"])   # the vault is unlocked only now
    return True, _public_employee(emp), ""


def update_employee_profile(old_username: str, new_full_name: str, new_username: str,
                            new_password: str = None, current_password: str = None) -> tuple:
    """The logged-in employee changes their OWN profile (old_username must be theirs).
    A new password needs current_password. A forgotten password is reset by the admin
    (reset_employee_password). Returns (success, message, employee or None)."""
    global _session
    try:
        actor_id = _actor(old_username)
        emp = _unlocked_db().update_employee(actor_id, old_username, new_full_name, new_username,
                                             new_password, current_password)
    except NotLoggedInAsError:
        return False, "You can only change your own profile", None
    except LookupError:
        return False, "Member not found", None
    except (PermissionError, ValueError, StorageError) as e:
        return False, str(e), None
    _session = emp
    return True, "Profile updated successfully", _public_employee(emp)


def _admin_action(admin_username: str, action) -> tuple:
    try:
        admin_id = _actor(admin_username)
    except (PermissionError, StorageError) as e:
        return False, str(e)
    try:
        action(_unlocked_db(), admin_id)
    except PermissionError as e:
        return False, str(e)
    except (LookupError, ValueError, StorageError) as e:
        return False, str(e)
    return True, "Done"


def reset_employee_password(admin_username: str, username: str, new_password: str) -> tuple:
    """The admin gives an employee a new password (forgotten password). Only that slot changes."""
    return _admin_action(admin_username, lambda db, a: db.reset_password(a, username, new_password))


def remove_employee(admin_username: str, username: str) -> tuple:
    """The admin removes an employee: they can no longer log in. Their clients and log rows stay."""
    return _admin_action(admin_username, lambda db, a: db.remove_employee(a, username))


# ============================================================
# CLIENTS (one company list; employee_username narrows to "enrolled by")
# ============================================================

def _set_to_gui(found):
    if found is None:
        return None
    return {
        "embeddings": [_b64(a) for a in found["embeddings"]],
        "model_version": found["model_version"],
        "enrolled_at": found["enrolled_at"],
        # Embedded by other weights (any row of the set): ask for re-enrolment.
        "stale": any(v != MODEL_VERSION for v in found["model_versions"]),
        "capture_device": found["capture_device"],
        "expires_at": found["expires_at"],
        # Past TEMPLATE_RETENTION_DAYS: the next login deletes it (ISO/IEC 24745).
        "expired": datetime.fromisoformat(found["expires_at"]) <= datetime.now(),
    }


def _client_to_gui(rec: dict) -> dict:
    """
    Old keys are kept for the existing pages (spatial_embeddings / spatial_origin show the
    "scan" set, or the "render" set when there is no scan set). New keys:
      spatial_sets: {"scan": {...}, "render": {...}}  (a missing origin is absent)
      temporal_set: {...} or None
    each set = {"embeddings": [base64], "model_version", "enrolled_at", "stale", "capture_device",
                "expires_at", "expired"}.
    """
    return {
        "display_name": rec["display_name"],
        "enrolled_by": rec["enrolled_by"],
        "dynamic_references": [_b64(a) for a in rec["temporal"]],
        "spatial_embeddings": [_b64(a) for a in rec["spatial"]],
        "spatial_origin": rec["spatial_origin"],
        "spatial_sets": {o: _set_to_gui(v) for o, v in rec["sets"].items()},
        "temporal_set": _set_to_gui(rec["temporal_set"]),
        "enrolled_at": rec["enrolled_at"],
        "updated_at": rec["updated_at"],
    }


def _narrowing_id(employee_username):
    """(ok, employee id or None). ok is False for an unknown employee name."""
    if employee_username is None:
        return True, None
    emp_id = _employee_id(employee_username)
    return emp_id is not None, emp_id


def client_exists(client_id: str, employee_username: str = None) -> bool:
    ok, emp_id = _narrowing_id(employee_username)
    return ok and _get_db().has_client(_norm(client_id), emp_id)


def get_client(client_id: str, employee_username: str = None):
    """
    The client, decrypted (plain base64 embeddings), or None. With employee_username,
    only if that employee enrolled it; pass None to look in the whole company list.
    """
    ok, emp_id = _narrowing_id(employee_username)
    if not ok:
        return None
    rec = _unlocked_db().get_client_record(_norm(client_id), emp_id)
    return None if rec is None else _client_to_gui(rec)


def list_clients(employee_username: str = None) -> list:
    """Returns list of dicts: {key, display_name, ...record}. None = every client."""
    ok, emp_id = _narrowing_id(employee_username)
    if not ok:
        return []
    return [{**_client_to_gui(r), "key": r["key"]} for r in _unlocked_db().list_client_records(emp_id)]


def _decode_embeddings(values, what: str) -> list:
    out = []
    for i, value in enumerate(values or []):
        try:
            out.append(base64.b64decode(value, validate=True))
        except (binascii.Error, ValueError, TypeError):
            raise ValueError(f"{what} #{i + 1} is not valid base64") from None
    return out


def save_client_enrollment(client_id: str, dynamic_refs, employee_username: str, *,
                           spatial_embeddings: list = None, spatial_origin: str = "scan",
                           capture_device: dict = None) -> tuple:
    """Save or replace a client's base64 embeddings and capture metadata; returns (ok, message)."""
    key = _norm(client_id or "")
    if not key:
        return False, "Signer ID is required"
    if "|" in key:
        return False, "Signer ID must not contain '|'"
    try:
        if _employee_id(employee_username or "") is None:
            return False, "Member not found"
        emp_id = _actor(employee_username)
    except (PermissionError, StorageError) as e:
        return False, str(e)

    try:
        spatial = _decode_embeddings(spatial_embeddings, "Spatial embedding")
        temporal = None if dynamic_refs is None else _decode_embeddings(dynamic_refs, "Dynamic reference")
        _unlocked_db().save_client(
            key, client_id.strip(), emp_id, spatial_embeddings=spatial, spatial_origin=spatial_origin,
            temporal_embeddings=temporal, model_version=MODEL_VERSION, capture_device=capture_device,
        )
    except PermissionError:
        return False, (
            f"'{client_id}' already has references. Only the owner can replace them; "
            f"ask the owner to re-enrol this signer."
        )
    except LookupError:
        return False, "Member not found"
    except (ValueError, StorageError) as e:
        return False, str(e)
    return True, "Saved"


def delete_client(client_id: str, employee_username: str = None) -> tuple:
    """Admin only: removes the client and all its templates; the verification log is kept and
    the deletion is logged. The logged-in employee is the one deleting. Returns
    (success, message); refused when not the admin, not the logged-in employee, or no such client."""
    try:
        emp_id = _actor(employee_username)
        deleted = _get_db().delete_client(_norm(client_id), emp_id)
    except (PermissionError, LookupError, StorageError) as e:
        return False, str(e)
    if not deleted:
        return False, f"No signer '{client_id.strip()}'"
    return True, f"Deleted signer '{client_id.strip()}'"


def count_clients_enrolled_this_month(employee_username: str = None) -> int:
    ok, emp_id = _narrowing_id(employee_username)
    if not ok:
        return 0
    return _get_db().count_clients_enrolled_in_month(datetime.now().strftime("%Y-%m"), emp_id)


# ============================================================
# VERIFICATION LOG
# ============================================================

def append_verification_log(client_id: str, decision: str, confidence: float,
                            modality_state: str, employee_username: str,
                            routing: str = None, degraded: bool = None, reasons: list = None,
                            model_version: str = None, threshold: float = None,
                            probe_fingerprints: list = None):
    """One event per visit, final verdict only. routing / degraded / reasons / threshold are
    what verify() returned; model_version is the tag of the references used;
    probe_fingerprints are probe_fingerprint() of each probe embedding, for the replay check.
    The event is logged as the logged-in employee; another name raises NotLoggedInAsError,
    and no login raises StorageLockedError (nothing can be verified without one)."""
    if decision not in FINAL_DECISIONS:
        # "Sign again" is a request, not a verdict: one event per visit, final verdict only.
        return
    emp_id = _actor(employee_username)
    _get_db().log_verification(
        _norm(client_id), modality_state, confidence, decision,
        employee_id=emp_id, display_name=client_id.strip(),
        employee_label=None,
        routing=routing, degraded=degraded, reasons=reasons,
        model_version=model_version, threshold=threshold, probe_hashes=probe_fingerprints,
    )


def is_replayed_probe(client_id: str, probe_fingerprints: list) -> bool:
    """True if this visit's probe is a replayed file, not a new signing: it equals one of
    the client's stored references (e.g. an enrolment scan given again), was already
    accepted for the client within REPLAY_WINDOW_DAYS, or repeats one signing."""
    return _unlocked_db().is_replayed_probe(_norm(client_id), probe_fingerprints)


def log_replay_refused(client_id: str, employee_username: str, probe_fingerprints: list) -> None:
    """Log a visit refused as a replay (not a verdict, not counted), as the logged-in
    employee; another name raises NotLoggedInAsError, no login StorageLockedError."""
    emp_id = _actor(employee_username)
    _get_db().log_replay_refused(_norm(client_id), client_id.strip(), emp_id, probe_fingerprints)


def log_verify_refused(client_id: str, employee_username: str, reasons: list) -> None:
    """Log a verification refused before it ran (stale set, other tablet), with the reasons
    shown; not a verdict, not counted. Logged as the logged-in employee, like log_replay_refused."""
    emp_id = _actor(employee_username)
    _get_db().log_verify_refused(_norm(client_id), client_id.strip(), emp_id, reasons)


def get_verification_log(employee_username: str = None) -> list:
    ok, emp_id = _narrowing_id(employee_username)
    if not ok:
        return []
    return _get_db().get_log_events(employee_id=emp_id)


def get_audit_log(employee_username: str = None) -> list:
    """Every audit row: verifications, enrolments, re-enrolments, deletions and refusals
    (the "event" key says which)."""
    ok, emp_id = _narrowing_id(employee_username)
    if not ok:
        return []
    return _get_db().get_log_events(employee_id=emp_id, event=None)


def get_last_verification_for_client(client_id: str, employee_username: str = None):
    ok, emp_id = _narrowing_id(employee_username)
    if not ok:
        return None
    events = _get_db().get_log_events(
        employee_id=emp_id, user_id=_norm(client_id), newest_first=True, limit=1)
    return events[0] if events else None


def last_event(event: str) -> dict | None:
    """Return the newest audit row for `event` in the current company, or None."""
    events = _get_db().get_log_events(event=event, newest_first=True, limit=1)
    if not events:
        return None
    e = events[0]
    return {
        "timestamp": e["timestamp"],
        "employee": e["employee"],
        "client": e["client_display_name"],
        "details": e["reasons"],
    }


def get_verification_stats(employee_username: str = None) -> dict:
    ok, emp_id = _narrowing_id(employee_username)
    if not ok:
        return {"total": 0, "genuine": 0, "genuine_rate": 0.0}
    total, genuine = _get_db().count_log_events(emp_id)
    genuine_rate = (genuine / total * 100) if total > 0 else 0.0
    return {"total": total, "genuine": genuine, "genuine_rate": round(genuine_rate, 1)}


# ============================================================
# REFERENCES (enrolled + confirmed signings added later)
# ============================================================

def get_reference_embeddings(client_id: str, employee_username: str = None,
                             spatial_origin: str = None) -> tuple:
    """Returns (base64 embeddings, spatial_origin): enrolled references first, then
    confirmed signings added later. Without spatial_origin, the client's "scan" set (or
    "render" when there is none); a spatial probe is scored only against its own origin."""
    record = get_client(client_id, employee_username)
    if record is None:
        return [], spatial_origin or "scan"
    if spatial_origin is None:
        return list(record["spatial_embeddings"]), record["spatial_origin"] or "scan"
    found = record["spatial_sets"].get(spatial_origin)
    return (list(found["embeddings"]) if found else []), spatial_origin


def _same_device(stored: dict, current: dict) -> bool:
    def norm(d, k):
        return str(d.get(k) or "").strip().lower()
    return all(norm(stored, k) == norm(current, k) for k in ("device_model", "os"))


def _expired_message(what: str, found: dict) -> str:
    return f"The {what} expired on {found['expires_at'][:10]}. Enrol the signer again."


def check_references(client_id: str, spatial_origin: str = None, use_pen: bool = False,
                     capture_device: dict = None) -> list:
    """Return user-facing error messages if stored references are expired, stale, or from another tablet."""
    record = get_client(client_id)
    if record is None:
        return []
    problems = []
    if spatial_origin is not None:
        found = record["spatial_sets"].get(spatial_origin)
        if found and found["expired"]:
            problems.append(_expired_message(f"{spatial_origin} references", found))
        if found and found["stale"]:
            problems.append(
                f"The {spatial_origin} references were made by another model version "
                f"({found['model_version']}, this app runs {MODEL_VERSION}). "
                f"Ask the owner to re-enrol the signer's paper signatures.")
    pen = record["temporal_set"]
    if use_pen and pen:
        if pen["expired"]:
            problems.append(_expired_message("pen references", pen))
        if pen["stale"]:
            problems.append(
                f"The pen references were made by another model version "
                f"({pen['model_version']}, this app runs {MODEL_VERSION}). "
                f"Ask the owner to re-enrol the signer's pen signatures.")
        stored = pen["capture_device"]
        if stored and not capture_device:
            problems.append(
                f"This pen signing did not say which tablet it came from; the pen references were "
                f"signed on {stored.get('device_model')} ({stored.get('os')}). Sign again on the tablet.")
        elif stored and not _same_device(stored, capture_device):
            problems.append(
                f"The pen references were signed on {stored.get('device_model')} "
                f"({stored.get('os')}); this signing is on {capture_device.get('device_model')} "
                f"({capture_device.get('os')}). Sign on the same tablet model and system, "
                f"or ask the owner to re-enrol the pen signatures.")
    return problems


def add_confirmed_reference(client_id: str, embedding_b64: str, employee_username: str,
                            modality: str = "spatial", spatial_origin: str = "scan") -> bool:
    """Append a confirmed Genuine, non-degraded embedding to the matching set up to MAX_REFERENCES."""
    if modality not in ("spatial", "temporal"):
        raise ValueError(f"modality must be 'spatial' or 'temporal', got {modality!r}")
    if modality == "spatial" and spatial_origin not in ("scan", "render"):
        raise ValueError(f"spatial_origin must be 'scan' or 'render', got {spatial_origin!r}")
    key = _norm(client_id or "")
    if not key:
        return False
    try:
        if _employee_id(employee_username or "") is None:
            return False
        _actor(employee_username)
    except (PermissionError, StorageError):
        return False
    db = _unlocked_db()
    record = db.get_client_record(key)
    if record is None:
        return False
    raw = _decode_embeddings([embedding_b64], "Confirmed signing")[0]
    if modality == "temporal":
        if not record["temporal"]:
            return False   # no pen set to add to: enrol the client on the tablet first
        try:
            db.add_confirmed_reference(key, "temporal", raw, MODEL_VERSION, max_per_set=MAX_REFERENCES)
        except (LookupError, ValueError):
            return False   # the set is gone or made by other weights
        return True
    if spatial_origin not in record["sets"]:
        return False   # no set of that origin: never put it in the other one
    try:
        db.add_confirmed_reference(key, "spatial", raw, MODEL_VERSION,
                                   spatial_origin=spatial_origin, max_per_set=MAX_REFERENCES)
    except (LookupError, ValueError):
        return False
    return True


# ============================================================
# BACKUP / RESTORE
# ============================================================

def backup_database(dest_folder: str, employee_username: str) -> tuple:
    """Copy the encrypted vault file into `dest_folder` (a folder the employee picks).
    No password and no unwrapped key goes into the copy; it opens with the same employee
    passwords. Returns (success, backup file path or message)."""
    try:
        if _employee_id(employee_username or "") is None:
            return False, "Member not found"
        actor = _actor(employee_username)
        db = _unlocked_db()
    except (PermissionError, StorageError) as e:
        return False, str(e)
    if not os.path.isdir(dest_folder):
        return False, f"No folder at {dest_folder}"
    name = f"signatures-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db"
    try:
        return True, db.backup_to(actor, os.path.join(dest_folder, name))
    except (OSError, sqlite3.Error) as e:   # PermissionError is an OSError
        return False, str(e)


def restore_database(backup_path: str, username: str, password: str) -> tuple:
    """
    Replace the current company's database with a backup. Allowed for the logged-in admin
    (or on a database with no accounts yet, as restore_company makes). The backup is checked before anything is replaced: it must be a
    sound backup of this app, and `username` / `password` must be an admin of the BACKUP.
    The database layer checks the admin again, keeps a copy of the live database beside it
    and logs the restore. That admin is logged in afterwards. Returns (success, message).
    """
    global _session
    db = _get_db()
    if db.has_employees() and not (_session and _session["is_admin"]):
        return False, "Only the owner can restore a backup."
    try:
        emp = db.restore_from(_session["id"] if _session else None, backup_path, username, password, _company)
    except (ValueError, PermissionError) as e:
        return False, str(e)
    except (OSError, sqlite3.Error) as e:   # disk full, unreadable file, locked database
        return False, f"The backup could not be restored: {e}"
    _session = emp
    if _company:      # a backup made before names were recorded: the vault keeps its name
        db.set_company_name(_company)
    return True, "Backup restored"
