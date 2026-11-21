"""
Encrypted template store: one SQLite database per vault, encrypted as a whole file.

Boundary (AGENTS.md): this layer holds the keys. It receives plain embeddings
from SignatureService.enroll_*() as float32 bytes (or np.ndarray), encrypts them
for storage, and returns decrypted float32 arrays for SignatureService.verify().

The file at rest (one file, so one os.replace saves the header and the body together):
  "SIGVAULT" | format version (2 bytes) | header length (4 bytes) | header (JSON) |
  nonce (12 bytes) | AES-256-GCM(data key, nonce, the serialized database, AAD = all before the nonce)
The header is plaintext and holds only what login needs: each active member's User ID
and key slot (scrypt salt and cost, nonce, wrapped data key), and the passphrase slot if
any. No name, no owner flag, no signer and no audit row. Every other byte is encrypted,
and the header is authenticated as the AAD: changing any byte makes the file refuse to
open, and nothing loads. The header is rebuilt from the encrypted `key_slots` table at
every save, so the two always agree.

In use, the database exists only in memory: login decrypts the body into one in-memory
SQLite connection (Connection.deserialize). Every committed write saves the whole file:
serialize, encrypt with a fresh random nonce, write an owner-only temp file beside the
vault, fsync, os.replace. A crash leaves the previous file or the new one, never half of
one; at worst the last write is lost. No plaintext copy, journal or temp file reaches the
disk. One app instance per vault: an OS lock on `<vault>.lock` is held from login to logout.

Keys: one random 256-bit data key (DEK) encrypts the file and, again, every template row
(with associated data that binds the row to its signer). The DEK is never stored in the
clear. It is wrapped once per member, with a key derived (scrypt) from that member's
password. Logging in IS unwrapping the slot: AES-GCM refuses a wrong password, so no
password hash is stored at all. At most one "passphrase" slot exists, for tests and CI
only (SIGVER_DB_PASSPHRASE). If every password is lost, nothing can be recovered.

Employees: the first account on an empty database is the admin. Only the admin adds
employees, resets a password, removes an employee, deletes a client, replaces a
client's existing references, or backs up or restores the database. Any employee
enrols new clients and verifies any client: clients belong to the company, not to the
employee who enrolled them (`enrolled_by` is recorded, never used to block). Every
enrolment, re-enrolment, deletion and restore, and every refused attempt, is written to
the audit log with the employee; the owner reads it after login.
"""
import base64
import hashlib
import json
import os
import sqlite3
import string
import struct
import threading
from collections import Counter
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta

if os.name == "nt":
    import msvcrt
else:
    import fcntl

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from config.settings import (
    MAX_REFERENCES,
    MAX_VAULT_BYTES,
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
VAULT_MAGIC = b"SIGVAULT"
VAULT_FORMAT_VERSION = 1
MAX_KDF_N = 2 ** 20   # a header asking for more scrypt work than this is refused (memory)

_VAULT_PREFIX = struct.Struct(">8sHI")   # magic, format version, header length
_NONCE_BYTES = 12
_TAG_BYTES = 16
_SQLITE_MAGIC = b"SQLite format 3\x00"
_EMPTY_HEADER = {"slots": {}, "passphrase": None}
_NOCASE = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)   # SQLite's COLLATE NOCASE
_FILE_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
_DAMAGED = "The vault file is damaged or was changed; it cannot be opened."
_NOT_A_VAULT = "This file is not a vault of this app."
_OLD_PLAINTEXT = ("This is a plaintext database made by an older version of the app; "
                  "it is older than the encrypted vault format and is not supported.")

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


# The audit columns that say what happened: two rows equal in all of them are the same
# event. restore_from carries a live row into the restored database unless the backup holds it.
_AUDIT_IDENTITY = ("a.timestamp, a.event, a.user_id, a.client_display_name, a.decision, a.similarity_score, "
                   "a.reasons, a.probe_hashes, a.modality_state, a.routing, a.threshold, a.model_version")
_BACKUP_AUDIT_SQL = "SELECT " + _AUDIT_IDENTITY + " FROM audit_log a"
_LIVE_AUDIT_SQL = ("SELECT " + _AUDIT_IDENTITY + ", a.employee_id, COALESCE(a.employee_label, e.username, ''), "
                   "a.degraded FROM audit_log a LEFT JOIN employees e ON e.id = a.employee_id ORDER BY a.id ASC")


class DatabaseLockedError(RuntimeError):
    """The data key is not loaded: an employee has to log in first."""


class VaultFileError(RuntimeError):
    """The vault file was not written by this app, or a byte of it was changed."""


class VaultInUseError(RuntimeError):
    """Another app instance has this vault open."""


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
# The company this database belongs to (one row). A database made before it has no row,
# and its company is unknown.
_COMPANY_INFO_SQL = "CREATE TABLE IF NOT EXISTS company_info (name TEXT NOT NULL)"
_ONE_SHARED_SLOT_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS one_shared_slot_per_kind ON key_slots(kind) "
    "WHERE kind != 'employee'"
)


def _create_schema(conn) -> None:
    """ISO/IEC 9075-1: the relational schema of a new vault, in one transaction."""
    conn.execute("BEGIN")
    conn.execute(_EMPLOYEES_SQL)
    conn.execute(_KEY_SLOTS_SQL)
    conn.execute(_ONE_SHARED_SLOT_SQL)
    conn.execute(_COMPANY_INFO_SQL)
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
    conn.execute("COMMIT")


def _memory_db(body: bytes = None):
    """An in-memory connection holding `body` (a serialized database), or a new vault's
    empty tables. Refuses a schema this code does not read."""
    conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
    try:
        conn.execute("PRAGMA temp_store = MEMORY")   # sorts and temp indexes never spill plaintext to the OS temp folder
        if body is None:
            _create_schema(conn)
        else:
            conn.deserialize(body)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='enrollments'").fetchone():
                raise VaultFileError(_NOT_A_VAULT)
            if version < SCHEMA_VERSION:
                raise RuntimeError(f"schema v{version} is older than v{SCHEMA_VERSION} and is not supported")
            if version != SCHEMA_VERSION:
                raise RuntimeError(f"The vault was created with schema version {version}, expected "
                                   f"{SCHEMA_VERSION}. Move it away and enrol again.")
        conn.execute("PRAGMA foreign_keys = ON")
    except BaseException:
        conn.close()
        raise
    return conn


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _slot_tuple(slot) -> tuple:
    """(salt, kdf_n, nonce, wrapped_dek) of one header slot. Raises ValueError, KeyError or
    TypeError for a malformed one, and refuses a scrypt cost that is not a power of two up to MAX_KDF_N."""
    n = slot["n"]
    if type(n) is not int or not 2 <= n <= MAX_KDF_N or n & (n - 1):
        raise ValueError("bad scrypt cost")
    nonce = base64.b64decode(slot["nonce"], validate=True)
    if len(nonce) != _NONCE_BYTES:
        raise ValueError("bad slot nonce")
    return (base64.b64decode(slot["salt"], validate=True), n, nonce, base64.b64decode(slot["wrapped"], validate=True))


def _header_bytes(conn) -> bytes:
    """The plaintext header, rebuilt from the encrypted key_slots table: active members only."""
    def slot(salt, n, nonce, wrapped):
        return {"salt": _b64(salt), "n": n, "nonce": _b64(nonce), "wrapped": _b64(wrapped)}
    members = {username: slot(*rest) for username, *rest in conn.execute(
        "SELECT e.username, k.salt, k.kdf_n, k.nonce, k.wrapped_dek FROM key_slots k "
        "JOIN employees e ON e.id = k.employee_id WHERE e.active = 1 ORDER BY k.id")}
    shared = conn.execute(
        "SELECT salt, kdf_n, nonce, wrapped_dek FROM key_slots WHERE kind = 'passphrase'").fetchone()
    return json.dumps({"kdf": "scrypt", "r": 8, "p": 1, "slots": members,
                       "passphrase": None if shared is None else slot(*shared)}).encode("utf-8")


def _encrypt_vault(conn, dek: bytes) -> bytes:
    """The whole vault file for the database in `conn`, with a fresh random nonce."""
    head = _header_bytes(conn)
    aad = _VAULT_PREFIX.pack(VAULT_MAGIC, VAULT_FORMAT_VERSION, len(head)) + head
    nonce = os.urandom(_NONCE_BYTES)
    return aad + nonce + AESGCM(dek).encrypt(nonce, conn.serialize(), aad)


def _parse_vault(data: bytes) -> tuple:
    """(header, authenticated bytes, nonce, ciphertext) of a vault file. Raises RuntimeError for
    an old plaintext database and VaultFileError for anything else this app did not write."""
    if data.startswith(_SQLITE_MAGIC):
        raise RuntimeError(_OLD_PLAINTEXT)
    if len(data) < _VAULT_PREFIX.size:
        raise VaultFileError(_NOT_A_VAULT)
    magic, version, length = _VAULT_PREFIX.unpack_from(data)
    if magic != VAULT_MAGIC:
        raise VaultFileError(_NOT_A_VAULT)
    if version != VAULT_FORMAT_VERSION:
        raise VaultFileError(f"Vault format {version} is not supported by this version of the app.")
    end = _VAULT_PREFIX.size + length
    if len(data) < end + _NONCE_BYTES + _TAG_BYTES:
        raise VaultFileError(_DAMAGED)
    try:
        head = json.loads(data[_VAULT_PREFIX.size:end].decode("utf-8"))
        for slot in [*head["slots"].values(), *([head["passphrase"]] if head["passphrase"] else [])]:
            _slot_tuple(slot)
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
        raise VaultFileError(_DAMAGED) from None
    return head, data[:end], data[end:end + _NONCE_BYTES], data[end + _NONCE_BYTES:]


def _decrypt_vault(data: bytes, dek: bytes) -> bytes:
    """The serialized database inside a vault file. Any changed byte: VaultFileError."""
    _, aad, nonce, ciphertext = _parse_vault(data)
    try:
        return AESGCM(dek).decrypt(nonce, ciphertext, aad)
    except InvalidTag:
        raise VaultFileError(_DAMAGED) from None


def _write_new(path: str, data: bytes) -> None:
    """Create `path` owner-only and write `data`, never over an existing file (FileExistsError)."""
    fd = os.open(path, _FILE_FLAGS, 0o600)  # skylos: ignore[SKY-D215,SKY-D324] the vault folder or the folder the owner picked; O_EXCL never follows a link or overwrites a file
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        os.unlink(path)  # skylos: ignore[SKY-D215] removes only the file this call just created
        raise


def _atomic_write(path: str, data: bytes) -> None:
    """Replace `path` with `data`: an owner-only temp file beside it, fsync, then os.replace,
    so a crash leaves the old file or the new one, never half of one."""
    folder = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(folder, f".{os.path.basename(path)}.{os.urandom(8).hex()}.tmp")
    try:
        _write_new(tmp, data)
        os.replace(tmp, path)  # skylos: ignore[SKY-D215] temp name made here, inside the vault folder
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(tmp)  # skylos: ignore[SKY-D215] temp name made here, inside the vault folder
        raise
    if os.name != "nt":   # the rename itself must reach the disk too
        dir_fd = os.open(folder, os.O_RDONLY)  # skylos: ignore[SKY-D215,SKY-D325] the vault's own folder, opened only to fsync it
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _lock_file(path: str):
    """An exclusive OS lock on `path`, held until the returned file is closed. The OS drops it
    when the process ends, so a crash leaves no stale lock. Raises VaultInUseError."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)  # skylos: ignore[SKY-D215] <vault>.lock beside the vault file
    f = os.fdopen(fd, "r+b")
    try:
        if os.name == "nt":
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise VaultInUseError("This vault is open in another window of the app. Close it there first.") from None
    return f


class DatabaseManager:
    KDF_N = 2 ** 17  # scrypt cost for every key slot; tests lower it for speed
    EVENT_BACKUP = EVENT_BACKUP
    EVENT_DELETE = EVENT_DELETE

    def __init__(self, db_path="signatures.db", passphrase=None, *, _backup_bytes=None):
        """
        Opens the store LOCKED: only the header is read, and nothing can be decrypted until
        authenticate_employee() succeeds. With `passphrase` (tests and CI only) the
        passphrase slot unlocks it at once, and is created on a new database.
        `_backup_bytes` (restore_from only): a backup held in memory, checked and edited
        there, never written back to `db_path` and never locked.
        """
        self.db_path = db_path
        self._dek = None
        self._conn = None        # the decrypted database, in memory, while unlocked
        self._instance = None    # the <vault>.lock file while unlocked
        self._backup = _backup_bytes
        self._lock = threading.RLock()
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
        """Logout: drop the decrypted database and the data key, and release the vault for
        another instance. Every write is already saved."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._dek = None
            if self._instance is not None:
                self._instance.close()
                self._instance = None

    def _require_key(self) -> bytes:
        if self._dek is None:
            raise DatabaseLockedError("The template store is locked: log in first")
        return self._dek

    def unlock_with_passphrase(self, passphrase) -> None:
        """Tests and CI (SIGVER_DB_PASSPHRASE)."""
        passphrase = self._to_secret(passphrase)
        data, head = self._load()
        if head["passphrase"]:
            dek = self._unwrap(passphrase, *_slot_tuple(head["passphrase"]))  # wrong passphrase: InvalidTag, never a new key
            self._open(dek, data)
            return
        if head["slots"]:
            raise ValueError("This database is unlocked by member passwords only; log in instead")
        dek = AESGCM.generate_key(256)
        slot = self._wrap(passphrase, dek)
        self._open(dek, None)
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT INTO key_slots (kind, employee_id, salt, kdf_n, nonce, wrapped_dek) "
                    "VALUES ('passphrase', NULL, ?, ?, ?, ?)", slot)
        except BaseException:
            self.lock()
            raise

    # -------------------------------------------------------------- database
    def _file_bytes(self):
        """The vault file's bytes, or None (no vault yet)."""
        if self._backup is not None:
            return self._backup
        try:
            with open(self.db_path, "rb") as f:
                data = f.read(MAX_VAULT_BYTES + 1)
        except FileNotFoundError:
            return None
        if len(data) > MAX_VAULT_BYTES:
            raise VaultFileError(_NOT_A_VAULT)
        return data

    def _load(self) -> tuple:
        """(file bytes or None, header), read once so the header and the body are the same file's."""
        data = self._file_bytes()
        return data, (_EMPTY_HEADER if data is None else _parse_vault(data)[0])

    def init_db(self):
        """Checks that the vault file, if any, is one this code reads (header only; nothing is
        written). A new vault's tables are made in memory when its first key slot is."""
        self._load()

    def _open(self, dek: bytes, data) -> None:
        """Decrypt `data` with `dek` into memory as the open store (None: a new vault, empty
        tables) and take the vault's instance lock. On any failure nothing stays loaded."""
        with self._lock:
            try:
                if self._backup is None:
                    if self._instance is None:
                        self._instance = _lock_file(self.db_path + ".lock")
                    current = self._file_bytes()   # another instance may have saved while we ran scrypt
                    if data is None and current is not None:
                        raise RuntimeError("Another account was created at the same time; try again")
                    data = current
                conn = _memory_db(None if data is None else _decrypt_vault(data, dek))
            except BaseException:
                if self._conn is None:
                    self.lock()
                raise
            if self._conn is not None:
                self._conn.close()
            self._conn, self._dek = conn, dek
            if self._backup is None:
                self._remove_stale_temp_files()

    def _remove_stale_temp_files(self) -> None:
        """Temp files a crash left before its os.replace: ciphertext, safe to drop (we hold the lock)."""
        folder, prefix = os.path.dirname(os.path.abspath(self.db_path)), f".{os.path.basename(self.db_path)}."
        for name in os.listdir(folder):
            if name.startswith(prefix) and name.endswith(".tmp"):
                with suppress(OSError):
                    os.unlink(os.path.join(folder, name))  # skylos: ignore[SKY-D215] only our own temp names, inside the vault folder

    def _require_conn(self):
        if self._conn is None:
            raise DatabaseLockedError("The template store is locked: log in first")
        return self._conn

    @contextmanager
    def _read(self):
        """The decrypted database, held against other threads (the GUI writes from a worker)."""
        with self._lock:
            yield self._require_conn()

    @contextmanager
    def _transaction(self):
        """One write transaction (BEGIN IMMEDIATE): roll back on any error; after COMMIT the
        whole vault file is saved."""
        with self._lock:
            conn = self._require_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            self._persist()

    def _persist(self) -> None:
        """Save the whole vault file. If saving fails, memory goes back to the file on disk,
        so the app never shows a write that was not kept."""
        if self._backup is not None:      # a backup under check is never written back
            return
        try:
            _atomic_write(self.db_path, _encrypt_vault(self._conn, self._dek))
        except BaseException:
            try:
                data = self._file_bytes()
                if data is None:
                    self.lock()               # a new vault whose first save failed: nothing was kept
                else:
                    self._conn.deserialize(_decrypt_vault(data, self._dek))
            except Exception:
                self.lock()
            raise

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
            raise LookupError("Member not found")
        return bool(row[0])

    def company_name(self):
        """The company name recorded in this database, or None (not recorded: unknown)."""
        with self._read() as conn:
            try:
                row = conn.execute("SELECT name FROM company_info").fetchone()
            except sqlite3.OperationalError:          # made before the table existed
                return None
        return None if row is None else row[0]

    def set_company_name(self, name: str) -> None:
        with self._transaction() as conn:
            conn.execute(_COMPANY_INFO_SQL)
            conn.execute("DELETE FROM company_info")
            conn.execute("INSERT INTO company_info (name) VALUES (?)", (name,))

    def has_employees(self) -> bool:
        """Answered from the header before login (an owner is always active, so it has a slot)."""
        with self._lock:
            if self._conn is not None:
                return self._conn.execute("SELECT 1 FROM employees LIMIT 1").fetchone() is not None
        return bool(self._load()[1]["slots"])

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
        new_vault = False
        if self.unlocked:
            with self._read() as conn:
                first = conn.execute("SELECT 1 FROM employees LIMIT 1").fetchone() is None
                if not first and (created_by is None or not self._is_admin(conn, created_by)):
                    raise PermissionError("Only the owner can add members")
            dek = self._dek
        else:
            head = self._load()[1]
            if head["slots"] and created_by is None:
                raise PermissionError("Only the owner can add members")
            if head["slots"] or head["passphrase"]:
                raise DatabaseLockedError("The template store is locked: log in first")
            first, new_vault, dek = True, True, AESGCM.generate_key(256)   # the admin's slot is the first copy
        slot = self._wrap(secret, dek)       # slow (scrypt): outside the write lock
        if new_vault:
            self._open(dek, None)
        try:
            with self._transaction() as conn:
                still_first = conn.execute("SELECT 1 FROM employees LIMIT 1").fetchone() is None
                if still_first != first:
                    raise RuntimeError("Another account was created at the same time; try again")
                if not first and not self._is_admin(conn, created_by):   # the creator may have lost admin meanwhile
                    raise PermissionError("Only the owner can add members")
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
                                [f"{username} (owner)" if first else username])
        except BaseException:
            if new_vault:
                self.lock()
            raise
        return emp_id

    def find_employee(self, username: str):
        """{'id', 'username', 'full_name', 'is_admin'} of an active employee, or None. Case-insensitive."""
        with self._read() as conn:
            row = conn.execute(
                "SELECT id, username, full_name, is_admin FROM employees "
                "WHERE username = ? COLLATE NOCASE AND active = 1",
                ((username or "").strip(),),
            ).fetchone()
        return None if row is None else self._employee_dict(row)

    def list_employees(self) -> list:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT id, username, full_name, is_admin FROM employees WHERE active = 1 ORDER BY id"
            ).fetchall()
        return [self._employee_dict(r) for r in rows]

    def authenticate_employee(self, username: str, password: str):
        """
        Returns (employee dict or None, reason), reason in {"", "not_found", "bad_password"}.
        The User ID and key slot come from the plaintext header; the password unwraps the
        slot, and that key decrypts the body into memory. Raises VaultFileError when the
        file was changed and VaultInUseError when another instance has the vault open.
        """
        name = (username or "").strip()
        data, head = self._load()
        found = [s for u, s in head["slots"].items() if u.translate(_NOCASE) == name.translate(_NOCASE)]
        if not found:
            return None, "not_found"
        try:
            dek = self._unwrap((password or "").encode("utf-8"), *_slot_tuple(found[0]))
        except InvalidTag:
            return None, "bad_password"
        with self._lock:
            if self._conn is None or self._dek != dek:
                self._open(dek, data)
            row = self._conn.execute(
                "SELECT id, username, full_name, is_admin FROM employees "
                "WHERE username = ? COLLATE NOCASE AND active = 1", (name,)).fetchone()
            if row is None:       # cannot happen: the header is rebuilt from these rows, and authenticated
                self.lock()
                return None, "not_found"
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
        with self._read() as conn:
            row = conn.execute(
                "SELECT id, is_admin, username, full_name FROM employees "
                "WHERE username = ? COLLATE NOCASE AND active = 1",
                ((old_username or "").strip(),),
            ).fetchone()
            if row is None:
                raise LookupError("Member not found")
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
                raise LookupError("Member not found")
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
                raise PermissionError("Only the owner can reset a password")
            row = conn.execute(
                "SELECT id, username FROM employees WHERE username = ? COLLATE NOCASE AND active = 1",
                ((username or "").strip(),)).fetchone()
            if row is None:
                raise LookupError("Member not found")
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
                raise PermissionError("Only the owner can remove a member")
            row = conn.execute(
                "SELECT id, username FROM employees WHERE username = ? COLLATE NOCASE AND active = 1",
                ((username or "").strip(),)).fetchone()
            if row is None:
                raise LookupError("Member not found")
            if row[0] == admin_id:
                raise ValueError("The owner cannot remove their own account")
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
                    raise PermissionError("Only the owner can replace a signer's existing references")
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
        with self._read() as conn:
            row = conn.execute(self._CLIENT_BY_ID, (user_id,)).fetchone()
            if row is None or (employee_id is not None and row[2] != employee_id):
                return None
            return self._assemble_client(conn, row)

    def list_client_records(self, employee_id: int = None) -> list:
        """Every client (the company list), or only those `employee_id` enrolled."""
        self._require_key()
        with self._read() as conn:
            rows = conn.execute(self._CLIENTS, (employee_id, employee_id)).fetchall()
            return [self._assemble_client(conn, r) for r in rows]

    def has_client(self, user_id: str, employee_id: int = None) -> bool:
        with self._read() as conn:
            if employee_id is None:
                row = conn.execute(
                    "SELECT 1 FROM users WHERE user_id = ? AND enrolled_by IS NOT NULL", (user_id,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM users WHERE user_id = ? AND enrolled_by = ?", (user_id, employee_id)).fetchone()
        return row is not None

    def count_clients_enrolled_in_month(self, year_month: str, employee_id: int = None) -> int:
        """year_month is 'YYYY-MM' (local time, like enrolled_at)."""
        with self._read() as conn:
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
                    raise PermissionError("Only the owner can delete a signer")
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
                raise LookupError(f"Signer {user_id!r} has no {modality} set to add to")
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
        with self._read() as conn:
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
        with self._read() as conn:
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
        with self._read() as conn:
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
        with self._read() as conn:
            total, genuine = conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(decision = 'Genuine'), 0) FROM audit_log
                   WHERE event = 'verify' AND (? IS NULL OR employee_id = ?)""",
                (employee_id, employee_id)).fetchone()
        return total, genuine

    # ------------------------------------------------------- backup / restore
    def backup_to(self, actor_id: int, dest_path: str) -> str:
        """
        Copy the vault file, as saved after the last write, to `dest_path`: header and
        encrypted body, never a password or an unwrapped key, so it opens with the same
        employee passwords. Only an active admin (`actor_id`) may do it (PermissionError).
        The file is created owner-only and never over an existing file (FileExistsError).
        """
        with self._read() as conn:
            if not self._is_admin(conn, actor_id):
                raise PermissionError("Only the owner can back up the database")
            data = self._file_bytes()      # saved after every write, so it is current
        _write_new(dest_path, data)
        with self._transaction() as conn:
            self._log_event(conn, EVENT_BACKUP, None, None, actor_id, [os.path.basename(dest_path)])
        return dest_path

    def restore_from(self, actor_id, backup_path: str, username: str, password: str, company: str = None) -> dict:
        """
        Replace this vault with a backup. If this vault has employees, only its logged-in
        active admin (`actor_id`) may do it (PermissionError). The backup is read once and
        checked in memory only: `username` / `password` is an employee of the backup whose
        key slot opens it, its body decrypts and authenticates, it holds no trigger, no view
        and no table this app does not create, it is a sound database of the schema this
        code reads (an older one is refused), and that employee is an ADMIN of the backup.
        With `company`, a backup that records another company name is refused ("This backup
        belongs to vault ..."); one with no name recorded is allowed.
        Then the live file is copied byte for byte to `<name>.before-restore-<YYYYmmdd-HHMMSS>`
        beside it (owner-only, encrypted like any backup). Every live audit row the backup
        does not hold (compared by content, not by time, so a row from the backup's last
        second or after the clock stepped back is kept) is added to the backup's body in live
        order, with employee_id NULL and the employee's username as employee_label, so a
        restore never erases the log. Then a "restore" row (who, which file, how many rows
        were carried, the safety copy's name) is added, and only then is the live file
        replaced, atomically. Any failure before that leaves the live file byte-identical.
        Afterwards the store is open with the backup's key. Returns that admin's employee dict.
        Raises ValueError (not a usable backup, another company's), PermissionError (not an
        admin, wrong password).
        """
        live_admin = None
        if self.has_employees():
            try:
                with self._read() as conn:
                    if self._is_admin(conn, actor_id):
                        live_admin = conn.execute(
                            "SELECT username FROM employees WHERE id = ?", (actor_id,)).fetchone()[0]
            except (LookupError, DatabaseLockedError):
                pass
            if live_admin is None:
                raise PermissionError("Only the owner can restore a backup")
        if not os.path.isfile(backup_path):
            raise ValueError(f"No backup file at {backup_path}")
        if os.path.getsize(backup_path) > MAX_VAULT_BYTES:
            raise ValueError("This file is too large to be a backup of this app")
        with open(backup_path, "rb") as f:  # skylos: ignore[SKY-D215,SKY-D325] the owner picks the backup file; size checked just above, read bounded
            data = f.read(MAX_VAULT_BYTES + 1)
        try:
            candidate = type(self)(backup_path, _backup_bytes=data)
        except RuntimeError as e:                     # an old plaintext database, or not a vault at all
            raise ValueError(str(e)) from None
        try:                                      # the backup's key and plaintext never outlive this call
            return self._restore_checked(candidate, backup_path, username, password, company, live_admin)
        finally:
            candidate.lock()

    def _restore_checked(self, candidate, backup_path, username, password, company, live_admin) -> dict:
        """restore_from after the permission check: open, check, keep a copy, carry, replace."""
        try:
            employee, _ = candidate.authenticate_employee(username, password)
            if employee is None:
                raise PermissionError("The user ID or password does not open this backup")
            fresh = _memory_db()                     # the tables a new vault gets
            try:
                tables = {n for (n,) in fresh.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            finally:
                fresh.close()
            with candidate._read() as conn:
                for kind, name in conn.execute("SELECT type, name FROM sqlite_master").fetchall():
                    if kind in ("trigger", "view"):
                        raise ValueError(f"The backup contains a {kind}, which this app never creates; "
                                         "refusing it.")
                    if kind == "table" and name not in tables:
                        raise ValueError(f"The backup contains a table ({name}) this app never creates; "
                                         "refusing it.")
                if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("The backup file is damaged")
        except sqlite3.DatabaseError:
            raise ValueError("This file is not a backup of this app") from None
        except RuntimeError as e:                     # an old or changed file, a schema this code cannot read
            raise ValueError(str(e)) from None
        recorded = candidate.company_name()
        if company is not None and recorded is not None and recorded != company:
            raise ValueError(f"This backup belongs to vault '{recorded}'.")
        if not employee["is_admin"]:
            raise PermissionError("Only an owner of the backup can restore it")
        safety, carried = None, []
        if live_admin is not None:
            with self._lock:
                live = self._file_bytes()
            base = f"{self.db_path}.before-restore-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            safety, n = base, 1
            while True:   # never over an existing copy: -2, -3, ...
                try:
                    _write_new(safety, live)
                    break
                except FileExistsError:
                    n += 1
                    safety = f"{base}-{n}"
            # A row is carried unless the backup holds an identical one (by content, not by
            # time: clocks repeat a second and step back). Counted, so a repeated row is too.
            with candidate._read() as conn:
                in_backup = Counter(conn.execute(_BACKUP_AUDIT_SQL))  # skylos: ignore[SKY-D211] module constant query, no user input
            with self._read() as conn:
                for row in conn.execute(_LIVE_AUDIT_SQL):  # skylos: ignore[SKY-D211] module constant query, no user input
                    key = row[:12]
                    if in_backup[key]:
                        in_backup[key] -= 1
                    else:
                        carried.append(row)
            if carried:
                with candidate._transaction() as conn:
                    conn.executemany(
                        """INSERT INTO audit_log (timestamp, event, user_id, client_display_name, decision,
                                                 similarity_score, reasons, probe_hashes, modality_state,
                                                 routing, threshold, model_version, employee_id,
                                                 employee_label, degraded)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
                        [r[:12] + r[13:] for r in carried])
        source = os.path.basename(backup_path)
        details = [f"restored by {live_admin} from {source}" if live_admin is not None
                   else f"restored by {employee['username']} into a new database from {source}"]
        if carried:
            details.append(f"{len(carried)} audit rows not in the backup carried over")
        if safety is not None:
            details.append(f"live vault kept as {os.path.basename(safety)}")
        with candidate._transaction() as conn:
            self._log_event(conn, EVENT_RESTORE, None, None, employee["id"], details)
        with self._lock:
            had_conn = self._conn is not None
            try:
                if self._instance is None:
                    self._instance = _lock_file(self.db_path + ".lock")
                _atomic_write(self.db_path, _encrypt_vault(candidate._conn, candidate._dek))
            except BaseException:
                if not had_conn:
                    self.lock()
                raise
            if had_conn:
                self._conn.close()
            self._conn, self._dek = candidate._conn, candidate._dek
            candidate._conn = None
        return employee
