"""
Encrypted template store (SQLite + AES-256-GCM at the application layer).

Boundary (AGENTS.md): this layer holds the keys. It receives plain embeddings
from SignatureService.enroll_*() as float32 bytes (or np.ndarray), encrypts them
for storage, and returns decrypted float32 arrays for SignatureService.verify().

Keys: one random 256-bit data key (DEK) encrypts every template. The DEK is never
stored in the clear. The `key_slots` table holds wrapped copies of it:
  * one slot per employee, wrapped with a key derived (scrypt) from that employee's
    login password. Logging in IS unwrapping the slot: AES-GCM refuses a wrong
    password, so no password hash is stored at all;
  * at most one "passphrase" slot, for tests and CI only (SIGVER_DB_PASSPHRASE).
No password and no unwrapped key is ever written to disk. A copied database is
useless without an employee's password. If every password is lost, the templates
cannot be recovered.

Employees: the first account on an empty database is the admin. Only the admin adds
employees, resets a password, removes an employee, deletes a client, replaces a
client's existing references, or backs up or restores the database. Any employee
enrols new clients and verifies any client: clients belong to the company, not to the
employee who enrolled them (`enrolled_by` is recorded, never used to block). Every
enrolment, re-enrolment, deletion and restore, and every refused attempt, is written to
the plaintext audit log with the employee.
"""
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from datetime import datetime, timedelta

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from config.settings import (
    MAX_REFERENCES,
    MIN_PASSWORD_LENGTH,
    REPLAY_WINDOW_DAYS,
    SPATIAL_EMBEDDING_DIM,
    TEMPORAL_EMBEDDING_DIM,
)

EXPECTED_DIMENSIONS = {
    "spatial": SPATIAL_EMBEDDING_DIM,
    "temporal": TEMPORAL_EMBEDDING_DIM,
}
SPATIAL_ORIGINS = ("render", "scan")   # "render": enroll_from_tablet, "scan": enroll_multimodal / unimodal spatial
SOURCES = ("enroll", "added")          # "added": confirmed signings appended after a Genuine verification
FINAL_DECISIONS = ("Genuine", "Forged")

PASSPHRASE_ENV_VAR = "SIGVER_DB_PASSPHRASE"   # tests and CI only
SCHEMA_VERSION = 5
WRAP_AAD = b"sigverify-dek-v2"

# Audit events. "verify" rows carry a decision; the others record who changed a client or
# an account, or who was refused. Only "verify" rows are counted as verifications.
EVENT_VERIFY = "verify"
EVENT_ENROL = "enrol"
EVENT_REENROL = "re-enrol"
EVENT_DELETE = "delete"
EVENT_REENROL_REFUSED = "re-enrol refused"
EVENT_DELETE_REFUSED = "delete refused"
EVENT_RESTORE = "restore"
EVENT_REPLAY_REFUSED = "replay refused"
EVENT_VERIFY_REFUSED = "verify refused"
EVENT_EMPLOYEE_ADDED = "employee added"
EVENT_PASSWORD_RESET = "password reset"
EVENT_EMPLOYEE_REMOVED = "employee removed"
EVENT_PROFILE_CHANGED = "profile changed"
EVENT_BACKUP = "backup"


class DatabaseLockedError(RuntimeError):
    """The data key is not loaded: an employee has to log in first."""


def _now_local() -> str:
    """Local ISO timestamp without timezone, the format the GUI has always stored."""
    return datetime.now().isoformat(timespec="seconds")


def probe_fingerprint(embedding) -> str:
    """SHA-256 of one probe embedding (float32 bytes or array), for the replay check.
    The hash is stored in the audit log; the embedding itself never is."""
    if isinstance(embedding, (bytes, bytearray)):
        raw = bytes(embedding)
    else:
        raw = np.asarray(embedding, dtype=np.float32).ravel().tobytes()
    return hashlib.sha256(raw).hexdigest()


_EMPLOYEES_SQL = """
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            full_name TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""


_KEY_SLOTS_SQL = """
    CREATE TABLE IF NOT EXISTS key_slots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL CHECK (kind IN ('employee', 'passphrase')),
        employee_id INTEGER UNIQUE REFERENCES employees(id) ON DELETE CASCADE,
        salt BLOB NOT NULL,
        kdf_n INTEGER NOT NULL,
        nonce BLOB NOT NULL,
        wrapped_dek BLOB NOT NULL,
        CHECK ((kind = 'employee') = (employee_id IS NOT NULL))
    )"""
_ONE_SHARED_SLOT_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS one_shared_slot_per_kind ON key_slots(kind) "
    "WHERE kind != 'employee'"
)


class DatabaseManager:
    KDF_N = 2 ** 17  # scrypt cost for every key slot; tests lower it for speed

    def __init__(self, db_path="signatures.db", passphrase=None):
        """
        Opens the store LOCKED: nothing can be decrypted until authenticate_employee()
        succeeds. With `passphrase` (tests and CI only) the passphrase slot unlocks it at
        once, and is created on a new database.
        """
        self.db_path = db_path
        self._dek = None
        self.init_db()
        if passphrase is not None:
            self.unlock_with_passphrase(passphrase)

    # ------------------------------------------------------------------ keys
    @staticmethod
    def _to_secret(value) -> bytes:
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not value:
            raise ValueError("Passphrase must not be empty")
        return bytes(value)

    def _derive(self, secret: bytes, salt: bytes, n: int) -> bytes:
        return Scrypt(salt=salt, length=32, n=n, r=8, p=1).derive(secret)

    def _wrap(self, secret: bytes, dek: bytes) -> tuple:
        """(salt, kdf_n, nonce, wrapped_dek) for a new key slot."""
        salt, nonce, n = os.urandom(16), os.urandom(12), int(self.KDF_N)
        wrapped = AESGCM(self._derive(secret, salt, n)).encrypt(nonce, dek, WRAP_AAD)
        return salt, n, nonce, wrapped

    def _unwrap(self, secret: bytes, salt, n, nonce, wrapped, aad=WRAP_AAD) -> bytes:
        """Raises InvalidTag on a wrong secret."""
        return AESGCM(self._derive(secret, salt, int(n))).decrypt(nonce, wrapped, aad)

    @property
    def unlocked(self) -> bool:
        return self._dek is not None

    def lock(self) -> None:
        """Forget the data key (logout). The next access needs a login again."""
        self._dek = None

    def _require_key(self) -> bytes:
        if self._dek is None:
            raise DatabaseLockedError("The template store is locked: log in first")
        return self._dek

    def unlock_with_passphrase(self, passphrase) -> None:
        """Tests and CI (SIGVER_DB_PASSPHRASE)."""
        passphrase = self._to_secret(passphrase)
        with closing(self.get_connection()) as conn:
            row = conn.execute(
                "SELECT salt, kdf_n, nonce, wrapped_dek FROM key_slots WHERE kind = 'passphrase'"
            ).fetchone()
            any_slot = conn.execute("SELECT 1 FROM key_slots LIMIT 1").fetchone()
        if row:
            self._dek = self._unwrap(passphrase, *row)  # wrong passphrase: InvalidTag, never a new key
            return
        if any_slot:
            raise ValueError("This database is unlocked by employee passwords only; log in instead")
        dek = AESGCM.generate_key(256)
        slot = self._wrap(passphrase, dek)
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM key_slots LIMIT 1").fetchone():
                raise RuntimeError("Another process created the key at the same time; open the database again")
            conn.execute(
                "INSERT INTO key_slots (kind, employee_id, salt, kdf_n, nonce, wrapped_dek) "
                "VALUES ('passphrase', NULL, ?, ?, ?, ?)", slot)
        self._dek = dek

    # -------------------------------------------------------------- database
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _transaction(self):
        """One explicit write transaction (BEGIN IMMEDIATE): commit on success, roll back on any error."""
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    @contextmanager
    def _migration(self):
        """A schema change as ONE transaction, foreign keys off while tables are rebuilt.
        DDL is transactional in SQLite, so a crash half-way leaves the old schema intact."""
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.execute("PRAGMA foreign_keys = OFF")  # cannot change inside a transaction
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise RuntimeError("Schema migration broke a foreign key; nothing was changed")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    def init_db(self):
        """ISO/IEC 9075-1: relational schema. Refuses a database made before v5 (nothing is written)."""
        with closing(self.get_connection()) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            has_tables = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='enrollments'"
            ).fetchone()
        if has_tables:
            if version < SCHEMA_VERSION:
                raise RuntimeError(f"schema v{version} is older than v{SCHEMA_VERSION} and is not supported")
            if version != SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.db_path} was created with schema version {version}, expected "
                    f"{SCHEMA_VERSION}. Delete it and re-enroll."
                )
            return
        with self._migration() as conn:
            conn.execute(_EMPLOYEES_SQL)
            conn.execute(_KEY_SLOTS_SQL)
            conn.execute(_ONE_SHARED_SLOT_SQL)
            # `users` holds the clients (user_id = lower-cased client ID).
            # enrolled_by points at employees.id so a username rename detaches nothing.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    display_name TEXT,
                    enrolled_by INTEGER REFERENCES employees(id),
                    spatial_origin TEXT CHECK (spatial_origin IS NULL OR spatial_origin IN ('render', 'scan')),
                    enrolled_at TEXT,
                    updated_at TEXT
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS enrollments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    modality TEXT NOT NULL CHECK (modality IN ('spatial', 'temporal')),
                    spatial_origin TEXT CHECK (spatial_origin IN ('render', 'scan')),
                    encrypted_embedding BLOB NOT NULL,
                    nonce BLOB NOT NULL,
                    model_version TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'enroll' CHECK (source IN ('enroll', 'added')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    enrolled_at TEXT,
                    capture_device TEXT,
                    CHECK ((modality = 'spatial' AND spatial_origin IS NOT NULL)
                        OR (modality = 'temporal' AND spatial_origin IS NULL)),
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )""")
            # No foreign key on user_id: the log outlives a deleted client.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    client_display_name TEXT,
                    employee_id INTEGER REFERENCES employees(id),
                    employee_label TEXT,
                    modality_state TEXT,
                    similarity_score REAL,
                    decision TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    routing TEXT,
                    degraded INTEGER,
                    reasons TEXT,
                    model_version TEXT,
                    threshold REAL,
                    event TEXT NOT NULL DEFAULT 'verify',
                    probe_hashes TEXT
                )""")
            conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")  # skylos: ignore[SKY-D211] PRAGMA takes no bound parameter; module int constant

    # ---------------------------------------------------------------- crypto
    @staticmethod
    def _aad(user_id: str, modality: str, spatial_origin) -> bytes:
        return f"{user_id}|{modality}|{spatial_origin or ''}".encode("utf-8")

    def encrypt_embedding(self, user_id, modality, spatial_origin, raw_bytes: bytes):
        nonce = os.urandom(12)
        ct = AESGCM(self._require_key()).encrypt(nonce, raw_bytes, self._aad(user_id, modality, spatial_origin))
        return ct, nonce

    def decrypt_embedding(self, user_id, modality, spatial_origin, ct: bytes, nonce: bytes) -> np.ndarray:
        raw = AESGCM(self._require_key()).decrypt(nonce, ct, self._aad(user_id, modality, spatial_origin))
        return np.frombuffer(raw, dtype=np.float32).copy()

    # ------------------------------------------------------------ validation
    @staticmethod
    def _check_user(user_id):
        if not isinstance(user_id, str) or not user_id or "|" in user_id:
            raise ValueError("user_id must be a non-empty string without '|'")

    @staticmethod
    def _check_modality(modality, spatial_origin):
        if modality not in EXPECTED_DIMENSIONS:
            raise ValueError(f"Unsupported modality: {modality}")
        if modality == "spatial":
            if spatial_origin not in SPATIAL_ORIGINS:
                raise ValueError(f"spatial templates need spatial_origin in {SPATIAL_ORIGINS}, got {spatial_origin!r}")
        elif spatial_origin is not None:
            raise ValueError("spatial_origin applies to spatial templates only")

    @staticmethod
    def _check_password(password) -> bytes:
        if not password or len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"The password must have at least {MIN_PASSWORD_LENGTH} characters")
        return password.encode("utf-8")

    @staticmethod
    def _to_raw_bytes(modality: str, embedding) -> bytes:
        dim = EXPECTED_DIMENSIONS[modality]
        if isinstance(embedding, (bytes, bytearray)):
            if len(embedding) != dim * 4:
                raise ValueError(
                    f"Invalid byte size for {modality}: expected {dim * 4} bytes, got {len(embedding)}"
                )
            raw = bytes(embedding)
            arr = np.frombuffer(raw, dtype=np.float32)
        elif isinstance(embedding, np.ndarray):
            if embedding.size != dim:
                raise ValueError(f"Invalid embedding dimension for {modality}: expected {dim}, got {embedding.size}")
            arr = embedding.astype(np.float32).ravel()
            raw = arr.tobytes()
        else:
            raise TypeError(f"Expected embedding as bytes or np.ndarray, got {type(embedding)}")
        if not np.isfinite(arr).all():
            raise ValueError("Embedding contains NaN or Inf")
        return raw

    # ------------------------------------------------------------- employees
    @staticmethod
    def _employee_dict(row) -> dict:
        return {"id": row[0], "username": row[1], "full_name": row[2], "is_admin": bool(row[3])}

    @staticmethod
    def _is_admin(conn, employee_id) -> bool:
        """Raises LookupError for an unknown or removed employee."""
        row = conn.execute("SELECT is_admin, active FROM employees WHERE id = ?", (employee_id,)).fetchone()
        if row is None or not row[1]:
            raise LookupError("Employee not found")
        return bool(row[0])

    def has_employees(self) -> bool:
        with closing(self.get_connection()) as conn:
            return conn.execute("SELECT 1 FROM employees LIMIT 1").fetchone() is not None

    def create_employee(self, full_name: str, username: str, password: str, created_by: int = None) -> int:
        """
        Register an employee and give them a key slot. The first account on an empty
        database is the admin (and creates the data key on a new install); after that only
        the admin (`created_by`, logged in) adds employees. Raises PermissionError for any
        other caller, ValueError for an empty or taken username or a short password.
        """
        username = (username or "").strip()
        if not username:
            raise ValueError("User ID is required")
        secret = self._check_password(password)
        with closing(self.get_connection()) as conn:
            first = conn.execute("SELECT 1 FROM employees LIMIT 1").fetchone() is None
            any_slot = conn.execute("SELECT 1 FROM key_slots LIMIT 1").fetchone() is not None
            if not first and (created_by is None or not self._is_admin(conn, created_by)):
                raise PermissionError("Only the admin can add employees")
        if self._dek is not None:
            dek = self._dek
        elif first and not any_slot:
            dek = AESGCM.generate_key(256)   # new install: the admin's slot is the first copy
        else:
            raise DatabaseLockedError("The template store is locked: log in first")
        slot = self._wrap(secret, dek)       # slow (scrypt): outside the write lock
        with self._transaction() as conn:
            still_first = conn.execute("SELECT 1 FROM employees LIMIT 1").fetchone() is None
            if still_first != first:
                raise RuntimeError("Another account was created at the same time; try again")
            if not first and not self._is_admin(conn, created_by):   # the creator may have lost admin meanwhile
                raise PermissionError("Only the admin can add employees")
            try:
                cur = conn.execute(
                    "INSERT INTO employees (username, full_name, is_admin) VALUES (?, ?, ?)",
                    (username, full_name, int(first)))
            except sqlite3.IntegrityError:
                raise ValueError("This User ID already exists") from None
            conn.execute(
                "INSERT INTO key_slots (kind, employee_id, salt, kdf_n, nonce, wrapped_dek) "
                "VALUES ('employee', ?, ?, ?, ?, ?)", (cur.lastrowid,) + slot)
            emp_id = cur.lastrowid
            # The first account (the admin) is logged as added by itself.
            self._log_event(conn, EVENT_EMPLOYEE_ADDED, None, None, emp_id if first else created_by,
                            [f"{username} (admin)" if first else username])
        self._dek = dek
        return emp_id

    def find_employee(self, username: str):
        """{'id', 'username', 'full_name', 'is_admin'} of an active employee, or None. Case-insensitive."""
        with closing(self.get_connection()) as conn:
            row = conn.execute(
                "SELECT id, username, full_name, is_admin FROM employees "
                "WHERE username = ? COLLATE NOCASE AND active = 1",
                ((username or "").strip(),),
            ).fetchone()
        return None if row is None else self._employee_dict(row)

    def list_employees(self) -> list:
        with closing(self.get_connection()) as conn:
            rows = conn.execute(
                "SELECT id, username, full_name, is_admin FROM employees WHERE active = 1 ORDER BY id"
            ).fetchall()
        return [self._employee_dict(r) for r in rows]

    def authenticate_employee(self, username: str, password: str):
        """
        Returns (employee dict or None, reason), reason in {"", "not_found", "bad_password"}.
        Success loads the data key: the password unwraps the employee's key slot.
        """
        with closing(self.get_connection()) as conn:
            row = conn.execute(
                "SELECT id, username, full_name, is_admin FROM employees "
                "WHERE username = ? COLLATE NOCASE AND active = 1",
                ((username or "").strip(),),
            ).fetchone()
            if row is None:
                return None, "not_found"
            slot = conn.execute(
                "SELECT salt, kdf_n, nonce, wrapped_dek FROM key_slots WHERE employee_id = ?", (row[0],)
            ).fetchone()
        if slot is None:
            return None, "bad_password"
        try:
            self._dek = self._unwrap((password or "").encode("utf-8"), *slot)
        except InvalidTag:
            return None, "bad_password"
        return self._employee_dict(row), ""

    def update_employee(self, actor_id: int, old_username: str, new_full_name: str, new_username: str,
                        new_password: str = None, current_password: str = None) -> dict:
        """
        An employee changes their OWN name / username / password, in one transaction.
        `actor_id` is the logged-in employee and must be the one changed: nobody, the admin
        included, edits another account here (the admin uses reset_password for a forgotten
        one). A new password needs `current_password`, checked by unwrapping the employee's
        key slot, and re-wraps that slot. The store must be unlocked (a logged-in session).
        Clients and log rows point at employees.id, so a rename keeps all of them.
        Raises LookupError (unknown employee), PermissionError (another account, or a wrong
        current password) or ValueError (empty / taken username, short password).
        """
        new_username = (new_username or "").strip()
        if not new_username:
            raise ValueError("User ID is required")
        dek = self._require_key()
        with closing(self.get_connection()) as conn:
            row = conn.execute(
                "SELECT id, is_admin, username, full_name FROM employees "
                "WHERE username = ? COLLATE NOCASE AND active = 1",
                ((old_username or "").strip(),),
            ).fetchone()
            if row is None:
                raise LookupError("Employee not found")
            if row[0] != actor_id:
                raise PermissionError("You can only change your own account")
            current = conn.execute(
                "SELECT salt, kdf_n, nonce, wrapped_dek FROM key_slots WHERE employee_id = ?",
                (row[0],)).fetchone()
        slot = None
        if new_password:
            secret = self._check_password(new_password)
            if not current_password or current is None:
                raise PermissionError("Enter your current password to set a new one")
            try:
                self._unwrap(current_password.encode("utf-8"), *current)
            except InvalidTag:
                raise PermissionError("The current password is incorrect") from None
            slot = self._wrap(secret, dek)
        with self._transaction() as conn:
            emp_id = row[0]
            if conn.execute("SELECT active FROM employees WHERE id = ?", (emp_id,)).fetchone() != (1,):
                raise LookupError("Employee not found")
            if slot is not None and conn.execute(
                    "SELECT salt, kdf_n, nonce, wrapped_dek FROM key_slots WHERE employee_id = ?",
                    (emp_id,)).fetchone() != tuple(current):
                raise ValueError("Your password was changed meanwhile. Try again.")
            clash = conn.execute(
                "SELECT 1 FROM employees WHERE username = ? COLLATE NOCASE AND id != ?",
                (new_username, emp_id),
            ).fetchone()
            if clash:
                raise ValueError("This User ID is already taken")
            conn.execute(
                "UPDATE employees SET full_name = ?, username = ? WHERE id = ?",
                (new_full_name, new_username, emp_id))
            if slot is not None:
                self._replace_slot(conn, emp_id, slot)
            changes = [text for changed, text in (
                (new_full_name != row[3], "name changed"),
                (new_username != row[2], f"username {row[2]} -> {new_username}"),
                (slot is not None, "password changed")) if changed]   # never the password itself
            if changes:
                self._log_event(conn, EVENT_PROFILE_CHANGED, None, None, emp_id, changes)
        return {"id": emp_id, "username": new_username, "full_name": new_full_name, "is_admin": bool(row[1])}

    @staticmethod
    def _replace_slot(conn, emp_id: int, slot: tuple) -> None:
        conn.execute("DELETE FROM key_slots WHERE employee_id = ?", (emp_id,))
        conn.execute(
            "INSERT INTO key_slots (kind, employee_id, salt, kdf_n, nonce, wrapped_dek) "
            "VALUES ('employee', ?, ?, ?, ?, ?)", (emp_id,) + slot)

    def reset_password(self, admin_id: int, username: str, new_password: str) -> None:
        """The admin gives an employee a new password (a new key slot). Only that slot changes."""
        slot = self._wrap(self._check_password(new_password), self._require_key())
        with self._transaction() as conn:
            if not self._is_admin(conn, admin_id):
                raise PermissionError("Only the admin can reset a password")
            row = conn.execute(
                "SELECT id, username FROM employees WHERE username = ? COLLATE NOCASE AND active = 1",
                ((username or "").strip(),)).fetchone()
            if row is None:
                raise LookupError("Employee not found")
            if row[0] == admin_id:
                # Otherwise anyone at an unlocked admin session could take the account over.
                raise ValueError("Change your own password in your profile, with your current password")
            self._replace_slot(conn, row[0], slot)
            self._log_event(conn, EVENT_PASSWORD_RESET, None, None, admin_id, [row[1]])

    def remove_employee(self, admin_id: int, username: str) -> None:
        """The admin removes an employee: their key slot is deleted and they cannot log in.
        The row stays (inactive) so clients and log rows keep the name."""
        with self._transaction() as conn:
            if not self._is_admin(conn, admin_id):
                raise PermissionError("Only the admin can remove an employee")
            row = conn.execute(
                "SELECT id, username FROM employees WHERE username = ? COLLATE NOCASE AND active = 1",
                ((username or "").strip(),)).fetchone()
            if row is None:
                raise LookupError("Employee not found")
            if row[0] == admin_id:
                raise ValueError("The admin cannot remove their own account")
            conn.execute("DELETE FROM key_slots WHERE employee_id = ?", (row[0],))
            conn.execute("UPDATE employees SET active = 0 WHERE id = ?", (row[0],))
            self._log_event(conn, EVENT_EMPLOYEE_REMOVED, None, None, admin_id, [row[1]])

    # --------------------------------------------------------------- clients
    @staticmethod
    def _set_exists(conn, user_id, modality, spatial_origin) -> bool:
        return conn.execute(
            "SELECT 1 FROM enrollments WHERE user_id = ? AND modality = ? AND spatial_origin IS ? LIMIT 1",
            (user_id, modality, spatial_origin)).fetchone() is not None

    def save_client(self, user_id: str, display_name: str, employee_id: int, *,
                    spatial_embeddings, spatial_origin, temporal_embeddings,
                    model_version: str, capture_device: dict = None) -> None:
        """
        Enroll or re-enroll a client in ONE transaction, logged with the employee.

          spatial_embeddings: replace the `spatial_origin` set ("scan" or "render");
                              None or [] leaves every spatial set untouched.
          temporal_embeddings: None keeps the stored pen set as it is (source and
                              model_version untouched); [] removes it; a list replaces it.
          capture_device:     {"device_model", "os", ...} of the tablet the pen set was
                              signed on, stored with the new pen set.

        Any employee enrols a new client or adds a set the client does not have yet.
        Replacing or removing a set the client already holds, or changing the display
        name, is a re-enrolment: admin only (PermissionError, and the refused attempt is
        logged; a non-admin's differently-cased name is simply not applied). Every change
        is logged. A row with no owner (made by add_references) is treated as new: its
        templates are dropped first, so nobody inherits sets they never enrolled.
        """
        self._check_user(user_id)
        if not model_version:
            raise ValueError("model_version is required")
        spatial = list(spatial_embeddings or [])
        if spatial and spatial_origin not in SPATIAL_ORIGINS:
            raise ValueError(f"spatial_origin must be one of {SPATIAL_ORIGINS}, got {spatial_origin!r}")
        keep_temporal = temporal_embeddings is None
        temporal = [] if keep_temporal else list(temporal_embeddings)
        device = None if capture_device is None else json.dumps(capture_device, sort_keys=True)
        now = _now_local()

        rows = []  # validate and encrypt everything before touching the DB
        for modality, origin, embs, dev in (("spatial", spatial_origin, spatial, None),
                                            ("temporal", None, temporal, device)):
            for emb in embs:
                raw = self._to_raw_bytes(modality, emb)
                ct, nonce = self.encrypt_embedding(user_id, modality, origin, raw)
                rows.append((user_id, modality, origin, ct, nonce, model_version, "enroll", now, dev))

        changes = []
        try:
            with self._transaction() as conn:
                is_admin = self._is_admin(conn, employee_id)
                existing = conn.execute(
                    "SELECT enrolled_by, display_name FROM users WHERE user_id = ?", (user_id,)).fetchone()
                if existing is not None and existing[0] is None:   # ownerless: start clean
                    conn.execute("DELETE FROM enrollments WHERE user_id = ?", (user_id,))
                    conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                    existing = None
                if existing is None and not rows:
                    raise ValueError("Nothing to save: give paper or pen signatures")
                replaced = []
                if existing is not None:
                    if spatial and self._set_exists(conn, user_id, "spatial", spatial_origin):
                        replaced.append(f"{spatial_origin} set")
                    if not keep_temporal and self._set_exists(conn, user_id, "temporal", None):
                        replaced.append("pen set")
                    if display_name != existing[1]:
                        if is_admin:
                            replaced.append("display name")
                        else:
                            display_name = existing[1]   # only the admin renames a client
                if replaced and not is_admin:
                    changes = replaced
                    raise PermissionError("Only the admin can replace a client's existing references")
                if spatial:
                    conn.execute(
                        "DELETE FROM enrollments WHERE user_id = ? AND modality = 'spatial' AND spatial_origin = ?",
                        (user_id, spatial_origin))
                if not keep_temporal:
                    conn.execute("DELETE FROM enrollments WHERE user_id = ? AND modality = 'temporal'", (user_id,))
                if existing is None:
                    conn.execute(
                        """INSERT INTO users (user_id, display_name, enrolled_by, enrolled_at, updated_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (user_id, display_name, employee_id, now, now))
                else:  # enrolled_by and enrolled_at stay: who enrolled first is recorded
                    conn.execute(
                        "UPDATE users SET display_name = ?, updated_at = ? WHERE user_id = ?",
                        (display_name, now, user_id))
                conn.executemany(
                    """INSERT INTO enrollments
                       (user_id, modality, spatial_origin, encrypted_embedding, nonce,
                        model_version, source, enrolled_at, capture_device)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows)
                saved = [f"{spatial_origin} set"] if spatial else []
                if temporal:
                    saved.append("pen set")
                elif "pen set" in replaced:
                    saved.append("pen set removed")
                saved += [c for c in replaced if c == "display name"]
                if saved:
                    self._log_event(conn, EVENT_REENROL if replaced else EVENT_ENROL,
                                    user_id, display_name, employee_id, saved)
        except PermissionError:
            with self._transaction() as conn:
                self._log_event(conn, EVENT_REENROL_REFUSED, user_id, display_name, employee_id, changes)
            raise

    _CLIENT_BY_ID = """
        SELECT u.user_id, u.display_name, u.enrolled_by, e.username, u.enrolled_at, u.updated_at
        FROM users u JOIN employees e ON e.id = u.enrolled_by
        WHERE u.user_id = ?"""
    _CLIENTS = """
        SELECT u.user_id, u.display_name, u.enrolled_by, e.username, u.enrolled_at, u.updated_at
        FROM users u JOIN employees e ON e.id = u.enrolled_by
        WHERE ? IS NULL OR u.enrolled_by = ?
        ORDER BY u.rowid ASC"""

    def _fetch_set(self, conn, user_id, modality, spatial_origin):
        """One reference set, or None if empty:
        {'embeddings', 'model_version', 'model_versions', 'enrolled_at', 'capture_device'}.
        model_versions lists every tag in the set, so one stale row makes the set stale."""
        rows = conn.execute(
            """SELECT encrypted_embedding, nonce, model_version, COALESCE(enrolled_at, created_at),
                      capture_device, source
               FROM enrollments WHERE user_id = ? AND modality = ? AND spatial_origin IS ?
               ORDER BY id ASC""",
            (user_id, modality, spatial_origin)).fetchall()
        if not rows:
            return None
        enrolled = [r for r in rows if r[5] == "enroll"] or rows
        device = next((r[4] for r in reversed(enrolled) if r[4]), None)
        return {
            "embeddings": [self.decrypt_embedding(user_id, modality, spatial_origin, r[0], r[1]) for r in rows],
            "model_version": enrolled[0][2],
            "model_versions": sorted({r[2] for r in rows}),
            "enrolled_at": enrolled[0][3],
            "capture_device": None if device is None else json.loads(device),
        }

    def _assemble_client(self, conn, row) -> dict:
        user_id, display_name, _, owner_name, enrolled_at, updated_at = row
        sets = {}
        for o in SPATIAL_ORIGINS:
            found = self._fetch_set(conn, user_id, "spatial", o)
            if found:
                sets[o] = found
        temporal = self._fetch_set(conn, user_id, "temporal", None)
        primary = "scan" if "scan" in sets else ("render" if "render" in sets else None)
        return {
            "key": user_id,
            "display_name": display_name,
            "enrolled_by": owner_name,
            "sets": sets,                  # {"scan": {...}, "render": {...}}, a missing origin is absent
            "temporal_set": temporal,      # {...} or None
            # flat view kept for the pages written before several sets existed:
            "spatial": sets[primary]["embeddings"] if primary else [],
            "temporal": temporal["embeddings"] if temporal else [],
            "spatial_origin": primary,
            "enrolled_at": enrolled_at,
            "updated_at": updated_at,
        }

    def get_client_record(self, user_id: str, employee_id: int = None):
        """Decrypted client record, or None. employee_id narrows to clients that employee enrolled."""
        self._require_key()
        with closing(self.get_connection()) as conn:
            row = conn.execute(self._CLIENT_BY_ID, (user_id,)).fetchone()
            if row is None or (employee_id is not None and row[2] != employee_id):
                return None
            return self._assemble_client(conn, row)

    def list_client_records(self, employee_id: int = None) -> list:
        """Every client (the company list), or only those `employee_id` enrolled."""
        self._require_key()
        with closing(self.get_connection()) as conn:
            rows = conn.execute(self._CLIENTS, (employee_id, employee_id)).fetchall()
            return [self._assemble_client(conn, r) for r in rows]

    def has_client(self, user_id: str, employee_id: int = None) -> bool:
        with closing(self.get_connection()) as conn:
            if employee_id is None:
                row = conn.execute(
                    "SELECT 1 FROM users WHERE user_id = ? AND enrolled_by IS NOT NULL", (user_id,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM users WHERE user_id = ? AND enrolled_by = ?", (user_id, employee_id)).fetchone()
        return row is not None

    def count_clients_enrolled_in_month(self, year_month: str, employee_id: int = None) -> int:
        """year_month is 'YYYY-MM' (local time, like enrolled_at)."""
        with closing(self.get_connection()) as conn:
            if employee_id is None:
                row = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE enrolled_by IS NOT NULL AND substr(enrolled_at, 1, 7) = ?",
                    (year_month,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE enrolled_by = ? AND substr(enrolled_at, 1, 7) = ?",
                    (employee_id, year_month)).fetchone()
        return row[0]

    def delete_client(self, user_id: str, employee_id: int) -> bool:
        """
        Admin only: delete the client and, by cascade, all templates. The audit log is kept,
        and the deletion (or the refused attempt, PermissionError) is logged with the employee.
        Returns False if there is no such client.
        """
        try:
            with self._transaction() as conn:
                if not self._is_admin(conn, employee_id):
                    raise PermissionError("Only the admin can delete a client")
                row = conn.execute("SELECT display_name FROM users WHERE user_id = ?", (user_id,)).fetchone()
                if row is None:
                    return False
                conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                self._log_event(conn, EVENT_DELETE, user_id, row[0], employee_id)
                return True
        except PermissionError:
            with self._transaction() as conn:
                self._log_event(conn, EVENT_DELETE_REFUSED, user_id, None, employee_id)
            raise

    # ----------------------------------------------------------------- write
    def _add_references(self, user_id: str, modality: str, embeddings, model_version: str,
                        spatial_origin: str = None, source: str = "enroll") -> int:
        """
        Store several templates of one kind in ONE transaction (all or nothing).
        `embeddings` is the list from enroll_*() (bytes) or np.ndarray vectors.
        Private: the public write paths are save_client() and add_confirmed_reference().
        A client created here has no owner; the first save_client() starts it clean.
        """
        self._check_user(user_id)
        self._check_modality(modality, spatial_origin)
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if not model_version:
            raise ValueError("model_version is required")
        embeddings = list(embeddings)
        if not embeddings:
            raise ValueError("No embeddings given")

        rows = []
        for emb in embeddings:  # validate and encrypt everything before touching the DB
            raw = self._to_raw_bytes(modality, emb)
            ct, nonce = self.encrypt_embedding(user_id, modality, spatial_origin, raw)
            rows.append((user_id, modality, spatial_origin, ct, nonce, model_version, source))

        with self._transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
            conn.executemany(
                """INSERT INTO enrollments
                   (user_id, modality, spatial_origin, encrypted_embedding, nonce, model_version, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)

    def add_confirmed_reference(self, user_id: str, modality: str, embedding, model_version: str,
                                spatial_origin: str = None, max_per_set: int = MAX_REFERENCES) -> None:
        """
        Append a confirmed signing to an existing client's references.

        CALLER'S RESPONSIBILITY (the database cannot check this): call only after a
        verification with decision == "Genuine", degraded == False AND the employee's
        confirmation. Never add a rejected, degraded or unconfirmed signing.

        Keeps at most `max_per_set` templates per (user, modality, origin) set by
        dropping the oldest *added* templates first; enrolled templates are never dropped.
        Insert and trim are one transaction: both or neither.
        Raises LookupError when the client has no such set, and ValueError when any row of
        the set was made by other weights than `model_version` (a stale set takes no rows).
        """
        self._check_user(user_id)
        self._check_modality(modality, spatial_origin)
        if not model_version:
            raise ValueError("model_version is required")
        raw = self._to_raw_bytes(modality, embedding)
        ct, nonce = self.encrypt_embedding(user_id, modality, spatial_origin, raw)
        with self._transaction() as conn:
            versions = {r[0] for r in conn.execute(
                "SELECT model_version FROM enrollments WHERE user_id = ? AND modality = ? AND spatial_origin IS ?",
                (user_id, modality, spatial_origin))}
            if not versions:
                raise LookupError(f"Client {user_id!r} has no {modality} set to add to")
            if versions != {model_version}:
                raise ValueError("The references were made by another model version; re-enrol them first")
            conn.execute(
                """INSERT INTO enrollments
                   (user_id, modality, spatial_origin, encrypted_embedding, nonce,
                    model_version, source, enrolled_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'added', ?)""",
                (user_id, modality, spatial_origin, ct, nonce, model_version, _now_local()))
            total = conn.execute(
                "SELECT COUNT(*) FROM enrollments WHERE user_id = ? AND modality = ? AND spatial_origin IS ?",
                (user_id, modality, spatial_origin)).fetchone()[0]
            excess = total - max_per_set
            if excess > 0:
                conn.execute(
                    """DELETE FROM enrollments WHERE id IN (
                           SELECT id FROM enrollments
                           WHERE user_id = ? AND modality = ? AND spatial_origin IS ? AND source = 'added'
                           ORDER BY id ASC LIMIT ?)""",
                    (user_id, modality, spatial_origin, excess))

    # ------------------------------------------------------------------ read
    def get_references(self, user_id: str, modality: str, spatial_origin: str = None,
                       model_version: str = None) -> list:
        """
        Decrypted float32 vectors, oldest first.
        For spatial templates pass the origin you will give to verify() as
        reference_spatial_origin; render and scan sets must never be mixed.
        """
        self._check_modality(modality, spatial_origin)
        with closing(self.get_connection()) as conn:
            rows = conn.execute(
                """SELECT encrypted_embedding, nonce, model_version FROM enrollments
                   WHERE user_id = ? AND modality = ? AND spatial_origin IS ? ORDER BY id ASC""",
                (user_id, modality, spatial_origin)).fetchall()
        return [self.decrypt_embedding(user_id, modality, spatial_origin, ct, nonce)
                for ct, nonce, version in rows if model_version is None or version == model_version]

    # ----------------------------------------------------------------- audit
    @staticmethod
    def _log_event(conn, event, user_id, display_name, employee_id, changes=None) -> None:
        conn.execute(
            """INSERT INTO audit_log (event, user_id, client_display_name, employee_id, reasons, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event, user_id, display_name, employee_id,
             json.dumps(list(changes)) if changes else None, _now_local()))

    def log_verification(self, user_id: str, modality_state: str, score: float, decision: str,
                         employee_id: int = None, display_name: str = None,
                         employee_label: str = None, timestamp: str = None,
                         routing: str = None, degraded: bool = None, reasons=None,
                         model_version: str = None, threshold: float = None,
                         probe_hashes=None) -> None:
        """
        IEEE 730: record EVERY final verification event (no de-duplication).
        "Sign again" is not a verdict (step-up) and must not be logged.
        employee_label is only a fallback name for an event whose employee has no account.
        routing / degraded / reasons / model_version / threshold come from verify() and
        explain the decision later; they stay plaintext and queryable (reasons is a JSON list).
        probe_hashes: probe_fingerprint() of each signing, for the replay check.
        """
        if decision not in FINAL_DECISIONS:
            raise ValueError(f"{decision!r} is not a final verdict; log the final decision of the visit")
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO audit_log (event, user_id, client_display_name, employee_id, employee_label,
                                          modality_state, similarity_score, decision, timestamp,
                                          routing, degraded, reasons, model_version, threshold,
                                          probe_hashes)
                   VALUES ('verify', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, display_name, employee_id, employee_label, modality_state, score,
                 decision,
                 datetime.fromisoformat(timestamp).isoformat(timespec="seconds") if timestamp else _now_local(),
                 routing,
                 None if degraded is None else int(bool(degraded)),
                 None if reasons is None else json.dumps(list(reasons)), model_version, threshold,
                 json.dumps(list(probe_hashes)) if probe_hashes else None),
            )

    def log_replay_refused(self, user_id: str, display_name: str, employee_id: int, probe_hashes) -> None:
        """A visit refused as a replay, with its probe hashes. Not a verdict: it is not
        counted and not listed with the verifications (get_log_events' default)."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO audit_log (event, user_id, client_display_name, employee_id, timestamp,
                                          probe_hashes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (EVENT_REPLAY_REFUSED, user_id, display_name, employee_id, _now_local(),
                 json.dumps(list(probe_hashes)) if probe_hashes else None))

    def log_verify_refused(self, user_id: str, display_name: str, employee_id: int, reasons) -> None:
        """A verification refused before verify() ran (a stale reference set, a pen probe
        from another tablet), with the reasons shown. Not a verdict: it is not counted and
        not listed with the verifications (get_log_events' default)."""
        with self._transaction() as conn:
            self._log_event(conn, EVENT_VERIFY_REFUSED, user_id, display_name, employee_id, reasons)

    def is_replayed_probe(self, user_id: str, probe_hashes, window_days: float = REPLAY_WINDOW_DAYS,
                          now: datetime = None) -> bool:
        """
        True if one of `probe_hashes` is a replayed file rather than a new signing:
          * it equals one of the client's stored references (enrolled or added), at any age:
            the enrolment scans may still be on disk, and resubmitting one gives
            exactly the stored embedding;
          * it was already ACCEPTED (Genuine) for this client within `window_days`;
          * the visit repeats one signing.
        Two real signings never give identical embeddings; a replayed file does.
        Needs the store unlocked (the references are decrypted to be hashed).
        """
        wanted = [h for h in (probe_hashes or []) if h]
        if not wanted:
            return False
        if len(set(wanted)) < len(wanted):
            return True
        self._require_key()
        cutoff = ((now or datetime.now()) - timedelta(days=window_days)).isoformat(timespec="seconds")
        with closing(self.get_connection()) as conn:
            rows = conn.execute(
                """SELECT probe_hashes FROM audit_log
                   WHERE event = 'verify' AND user_id = ? AND decision = 'Genuine'
                   AND probe_hashes IS NOT NULL AND timestamp >= ?""",
                (user_id, cutoff)).fetchall()
            refs = conn.execute(
                "SELECT modality, spatial_origin, encrypted_embedding, nonce FROM enrollments WHERE user_id = ?",
                (user_id,)).fetchall()
        wanted = set(wanted)
        if any(probe_fingerprint(self.decrypt_embedding(user_id, m, o, ct, n)) in wanted
               for m, o, ct, n in refs):
            return True
        return any(wanted & set(json.loads(r[0])) for r in rows)

    def get_log_events(self, employee_id: int = None, user_id: str = None,
                       newest_first: bool = False, limit: int = None, event: str = EVENT_VERIFY) -> list:
        """Audit rows, oldest first. event=None returns every kind (verify, enrol, re-enrol, delete,
        refusals, account changes, backup, restore)."""
        sql = """SELECT a.user_id, a.client_display_name, a.decision, a.similarity_score,
                        a.modality_state, COALESCE(e.username, a.employee_label, ''), a.timestamp,
                        a.routing, a.degraded, a.reasons, a.model_version, a.threshold, a.event
                 FROM audit_log a LEFT JOIN employees e ON e.id = a.employee_id
                 WHERE (? IS NULL OR a.employee_id = ?) AND (? IS NULL OR a.user_id = ?)
                   AND (? IS NULL OR a.event = ?)"""
        params = [employee_id, employee_id, user_id, user_id, event, event]
        sql += " ORDER BY a.id DESC" if newest_first else " ORDER BY a.id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with closing(self.get_connection()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {"client_key": r[0], "client_display_name": r[1], "decision": r[2], "confidence": r[3],
             "modality_state": r[4], "employee": r[5], "timestamp": r[6],
             "routing": r[7], "degraded": None if r[8] is None else bool(r[8]),
             "reasons": None if r[9] is None else json.loads(r[9]),
             "model_version": r[10], "threshold": r[11], "event": r[12]}
            for r in rows
        ]

    def count_log_events(self, employee_id: int = None):
        """(total, genuine) over the final-verdict log (verification events only)."""
        with closing(self.get_connection()) as conn:
            total, genuine = conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(decision = 'Genuine'), 0) FROM audit_log
                   WHERE event = 'verify' AND (? IS NULL OR employee_id = ?)""",
                (employee_id, employee_id)).fetchone()
        return total, genuine

    # ------------------------------------------------------- backup / restore
    def backup_to(self, actor_id: int, dest_path: str) -> str:
        """
        Copy the whole (encrypted) database to `dest_path`, consistently, through SQLite's
        backup API. The copy holds only what the database holds: wrapped key slots, never a
        password or an unwrapped key, so it opens with the same employee passwords.
        Only an active admin (`actor_id`) may do it (PermissionError). The file is created
        owner-only and never over an existing file (FileExistsError).
        """
        with closing(self.get_connection()) as conn:
            if not self._is_admin(conn, actor_id):
                raise PermissionError("Only the admin can back up the database")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        os.close(os.open(dest_path, flags, 0o600))  # skylos: ignore[SKY-D215,SKY-D324] the admin picks the backup folder; O_EXCL never follows a link or overwrites a file
        try:
            with closing(self.get_connection()) as src, closing(sqlite3.connect(dest_path)) as dst:
                src.backup(dst)
        except BaseException:
            os.unlink(dest_path)  # skylos: ignore[SKY-D215] removes only the file this call just created
            raise
        with self._transaction() as conn:
            self._log_event(conn, EVENT_BACKUP, None, None, actor_id, [os.path.basename(dest_path)])
        return dest_path

    def restore_from(self, actor_id, backup_path: str, username: str, password: str) -> dict:
        """
        Replace this database with a backup. If this database has employees, only its active
        admin (`actor_id`) may do it (PermissionError). Everything is checked on a scratch
        copy first: it holds no trigger, no view and no table this app does not create, it
        is a sound SQLite database of the schema this code reads (an older one is refused),
        and `username` / `password` is an ADMIN of the backup whose key slot opens it. Then
        the live database is copied to `<name>.before-restore-<YYYYmmdd-HHMMSS>` beside it
        (owner-only), a "restore" row is written into the scratch copy, and only then is the
        live database replaced. A failure before the replacement leaves the live data as it
        was (the owner-only copy adds one "backup" audit row to the live file). Afterwards the store is unlocked with the backup's key. Returns that admin's
        employee dict.
        Raises ValueError (not a usable backup), PermissionError (not an admin, wrong password).
        """
        live_admin = None
        if self.has_employees():
            with closing(self.get_connection()) as conn:
                try:
                    is_admin = self._is_admin(conn, actor_id)
                except LookupError:
                    is_admin = False
                if not is_admin:
                    raise PermissionError("Only the admin can restore a backup")
                live_admin = conn.execute("SELECT username FROM employees WHERE id = ?", (actor_id,)).fetchone()[0]
        if not os.path.isfile(backup_path):
            raise ValueError(f"No backup file at {backup_path}")
        scratch_dir = tempfile.mkdtemp(prefix="sigverify-restore-")
        scratch = os.path.join(scratch_dir, "check.db")
        try:
            shutil.copyfile(backup_path, scratch)
            fresh = type(self)(os.path.join(scratch_dir, "fresh.db"))   # the tables a new database gets
            with closing(fresh.get_connection()) as conn:
                tables = {n for (n,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            try:
                with closing(sqlite3.connect(scratch)) as conn:
                    version = conn.execute("PRAGMA user_version").fetchone()[0]
                    found = conn.execute("SELECT type, name FROM sqlite_master").fetchall()
                    if found and version < SCHEMA_VERSION:   # same message as init_db, not "foreign table"
                        raise ValueError(f"schema v{version} is older than v{SCHEMA_VERSION} and is not supported")
                    for kind, name in found:
                        if kind in ("trigger", "view"):
                            raise ValueError(f"The backup contains a {kind}, which this app never creates; "
                                             "refusing it.")
                        if kind == "table" and name not in tables:
                            raise ValueError(f"The backup contains a table ({name}) this app never creates; "
                                             "refusing it.")
                    ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
                if ok != "ok":
                    raise ValueError("The backup file is damaged")
                candidate = type(self)(scratch)
            except sqlite3.DatabaseError:
                raise ValueError("This file is not a backup of this app") from None
            except RuntimeError as e:
                raise ValueError(str(e)) from None
            employee, _ = candidate.authenticate_employee(username, password)
            if employee is None:
                raise PermissionError("The user ID or password does not open this backup")
            if not employee["is_admin"]:
                raise PermissionError("Only an admin of the backup can restore it")
            if live_admin is not None:
                base = f"{self.db_path}.before-restore-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                dest, n = base, 1
                while True:   # never over an existing copy: -2, -3, ...
                    try:
                        self.backup_to(actor_id, dest)
                        break
                    except FileExistsError:
                        n += 1
                        dest = f"{base}-{n}"
            with candidate._transaction() as conn:
                self._log_event(conn, EVENT_RESTORE, None, None, employee["id"], [
                    f"restored by {live_admin or 'new install'} from {os.path.basename(backup_path)}"])
            with closing(sqlite3.connect(scratch)) as src, closing(sqlite3.connect(self.db_path)) as dst:
                src.backup(dst)
            self._dek = candidate._dek
            return employee
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
