"""
Encrypted template store (SQLite + AES-256-GCM at the application layer).

Boundary (AGENTS.md): this layer holds the keys. It receives plain embeddings
from SignatureService.enroll_*() as float32 bytes (or np.ndarray), encrypts them
for storage, and returns decrypted float32 arrays for SignatureService.verify().

Key management: one random 256-bit data key (DEK) is wrapped by a key derived
from the operator passphrase (scrypt + random salt stored in the `meta` table).
The passphrase itself is never stored in the database. A wrong passphrase raises
InvalidTag and never creates a new key.

Employees: passwords are never stored. Each employee has a random 16-byte salt and
a scrypt hash (n=2**14, r=8, p=1); login compares with hmac.compare_digest.

Ownership: a client (table `users`) belongs to an employee by employees.id, not by
username, so renaming an employee cannot detach clients or log rows from them.
"""
import hashlib
import hmac
import json
import os
import sqlite3
import warnings
from contextlib import closing, contextmanager
from datetime import datetime, timezone

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

try:
    from config.settings import SPATIAL_EMBEDDING_DIM, TEMPORAL_EMBEDDING_DIM
except ImportError:  # only when the file is run outside the repo
    warnings.warn("config.settings not importable; using default embedding widths 512/128")
    SPATIAL_EMBEDDING_DIM, TEMPORAL_EMBEDDING_DIM = 512, 128

EXPECTED_DIMENSIONS = {
    "spatial": SPATIAL_EMBEDDING_DIM,
    "temporal": TEMPORAL_EMBEDDING_DIM,
}
SPATIAL_ORIGINS = ("render", "scan")   # "render": enroll_from_tablet, "scan": enroll_multimodal / unimodal spatial
SOURCES = ("enroll", "added")          # "added": confirmed signings appended after a Genuine verification

MAX_REFERENCES_PER_SET = 8             # AGENTS.md: measured arms held 6 to 8; not measured beyond
PASSPHRASE_ENV_VAR = "SIGVER_DB_PASSPHRASE"
SCHEMA_VERSION = 4
OLDEST_MIGRATABLE_SCHEMA = 3
WRAP_AAD = b"sigverify-dek-v1"
EXPIRES_FORMAT = "%Y-%m-%d %H:%M:%S"   # UTC, same format as SQLite CURRENT_TIMESTAMP

# Employee password hashing (standard library only).
PASSWORD_SALT_BYTES = 16
PASSWORD_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}


def _now_local() -> str:
    """Local ISO timestamp without timezone, the format the GUI has always stored."""
    return datetime.now().isoformat(timespec="seconds")


class DatabaseManager:
    KDF_N = 2 ** 17  # scrypt cost for the key-wrapping key; tests lower it for speed

    def __init__(self, db_path="signatures.db", passphrase=None):
        self.db_path = db_path
        passphrase = self._resolve_passphrase(passphrase)
        self.init_db()
        self._dek = self._open_data_key(passphrase)

    # ------------------------------------------------------------------ keys
    @staticmethod
    def _resolve_passphrase(passphrase) -> bytes:
        if passphrase is None:
            passphrase = os.environ.get(PASSPHRASE_ENV_VAR)
        if passphrase is None:
            raise ValueError(
                f"A passphrase is required: pass passphrase=... or set {PASSPHRASE_ENV_VAR}"
            )
        if isinstance(passphrase, str):
            passphrase = passphrase.encode("utf-8")
        if not passphrase:
            raise ValueError("Passphrase must not be empty")
        return bytes(passphrase)

    def _wrapping_key(self, passphrase: bytes, salt: bytes) -> bytes:
        return Scrypt(salt=salt, length=32, n=self.KDF_N, r=8, p=1).derive(passphrase)

    def _open_data_key(self, passphrase: bytes) -> bytes:
        with closing(self.get_connection()) as conn:
            row = conn.execute("SELECT salt, wrapped_dek, nonce FROM meta WHERE id = 1").fetchone()
            if row is None:  # first run only
                salt, nonce, dek = os.urandom(16), os.urandom(12), AESGCM.generate_key(256)
                wrapped = AESGCM(self._wrapping_key(passphrase, salt)).encrypt(nonce, dek, WRAP_AAD)
                with conn:
                    conn.execute(
                        "INSERT INTO meta (id, salt, wrapped_dek, nonce) VALUES (1, ?, ?, ?)",
                        (salt, wrapped, nonce),
                    )
                return dek
        salt, wrapped, nonce = row
        # Wrong passphrase -> InvalidTag. Never falls through to generating a new key.
        return AESGCM(self._wrapping_key(passphrase, salt)).decrypt(nonce, wrapped, WRAP_AAD)

    def change_passphrase(self, new_passphrase) -> None:
        """Re-wrap the data key under a new passphrase. Stored templates are untouched."""
        new_pp = self._resolve_passphrase(new_passphrase)
        salt, nonce = os.urandom(16), os.urandom(12)
        wrapped = AESGCM(self._wrapping_key(new_pp, salt)).encrypt(nonce, self._dek, WRAP_AAD)
        with closing(self.get_connection()) as conn, conn:
            conn.execute(
                "UPDATE meta SET salt = ?, wrapped_dek = ?, nonce = ? WHERE id = 1",
                (salt, wrapped, nonce),
            )

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

    def init_db(self):
        """ISO/IEC 9075-1: relational schema. Refuses a database from an older schema."""
        with closing(self.get_connection()) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            has_tables = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='enrollments'"
            ).fetchone()
            if has_tables and version == OLDEST_MIGRATABLE_SCHEMA:
                self._migrate_3_to_4(conn)
                version = SCHEMA_VERSION
            if has_tables and version != SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.db_path} was created with an older schema (version {version}, "
                    f"expected {SCHEMA_VERSION}). Delete it and re-enroll."
                )
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS meta (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        salt BLOB NOT NULL,
                        wrapped_dek BLOB NOT NULL,
                        nonce BLOB NOT NULL
                    )""")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS employees (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                        full_name TEXT NOT NULL,
                        salt BLOB NOT NULL,
                        password_hash BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )""")
                # `users` holds the clients (user_id = lower-cased client ID).
                # enrolled_by points at employees.id so a username rename detaches nothing.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        display_name TEXT,
                        enrolled_by INTEGER REFERENCES employees(id),
                        image_refs_enc BLOB,
                        image_refs_nonce BLOB,
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
                        expires_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        enrolled_at TEXT,
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
                        threshold REAL
                    )""")
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _migrate_3_to_4(conn):
        """v3 -> v4 keeps all data: adds per-set enrolment time and the fuller audit columns."""
        with conn:
            conn.execute("ALTER TABLE enrollments ADD COLUMN enrolled_at TEXT")
            for col, typ in (("routing", "TEXT"), ("degraded", "INTEGER"), ("reasons", "TEXT"),
                             ("model_version", "TEXT"), ("threshold", "REAL")):
                conn.execute(f"ALTER TABLE audit_log ADD COLUMN {col} {typ}")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ---------------------------------------------------------------- crypto
    @staticmethod
    def _aad(user_id: str, modality: str, spatial_origin) -> bytes:
        return f"{user_id}|{modality}|{spatial_origin or ''}".encode("utf-8")

    def encrypt_embedding(self, user_id, modality, spatial_origin, raw_bytes: bytes):
        nonce = os.urandom(12)
        ct = AESGCM(self._dek).encrypt(nonce, raw_bytes, self._aad(user_id, modality, spatial_origin))
        return ct, nonce

    def decrypt_embedding(self, user_id, modality, spatial_origin, ct: bytes, nonce: bytes) -> np.ndarray:
        raw = AESGCM(self._dek).decrypt(nonce, ct, self._aad(user_id, modality, spatial_origin))
        return np.frombuffer(raw, dtype=np.float32).copy()

    @staticmethod
    def _refs_aad(user_id: str) -> bytes:
        return f"{user_id}|image_refs".encode("utf-8")

    def _encrypt_blob(self, aad: bytes, raw: bytes):
        nonce = os.urandom(12)
        return AESGCM(self._dek).encrypt(nonce, raw, aad), nonce

    def _decrypt_blob(self, aad: bytes, ct: bytes, nonce: bytes) -> bytes:
        return AESGCM(self._dek).decrypt(nonce, ct, aad)

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

    @staticmethod
    def _normalize_expiry(expires_at):
        if expires_at is None:
            return None
        if isinstance(expires_at, datetime):
            if expires_at.tzinfo is not None:
                expires_at = expires_at.astimezone(timezone.utc)
            return expires_at.strftime(EXPIRES_FORMAT)
        datetime.strptime(expires_at, EXPIRES_FORMAT)  # raises ValueError on a bad format
        return expires_at

    # ------------------------------------------------------------- employees
    @staticmethod
    def _hash_password(password: str, salt: bytes) -> bytes:
        return hashlib.scrypt(password.encode("utf-8"), salt=salt, **PASSWORD_SCRYPT)

    def create_employee(self, full_name: str, username: str, password: str) -> int:
        """Register an employee. The password is never stored, only salt + scrypt hash."""
        username = (username or "").strip()
        if not username:
            raise ValueError("User ID is required")
        if not password:
            raise ValueError("Password is required")
        salt = os.urandom(PASSWORD_SALT_BYTES)
        pw_hash = self._hash_password(password, salt)
        try:
            with closing(self.get_connection()) as conn, conn:
                cur = conn.execute(
                    "INSERT INTO employees (username, full_name, salt, password_hash) VALUES (?, ?, ?, ?)",
                    (username, full_name, salt, pw_hash),
                )
                return cur.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError("This User ID already exists") from None

    def find_employee(self, username: str):
        """{'id', 'username', 'full_name'} (no salt, no hash) or None. Case-insensitive."""
        with closing(self.get_connection()) as conn:
            row = conn.execute(
                "SELECT id, username, full_name FROM employees WHERE username = ? COLLATE NOCASE",
                ((username or "").strip(),),
            ).fetchone()
        return None if row is None else {"id": row[0], "username": row[1], "full_name": row[2]}

    def authenticate_employee(self, username: str, password: str):
        """Returns (employee dict or None, reason) with reason in {"", "not_found", "bad_password"}."""
        with closing(self.get_connection()) as conn:
            row = conn.execute(
                "SELECT id, username, full_name, salt, password_hash FROM employees "
                "WHERE username = ? COLLATE NOCASE",
                ((username or "").strip(),),
            ).fetchone()
        if row is None:
            return None, "not_found"
        candidate = self._hash_password(password or "", row[3])
        if not hmac.compare_digest(candidate, row[4]):
            return None, "bad_password"
        return {"id": row[0], "username": row[1], "full_name": row[2]}, ""

    def update_employee(self, old_username: str, new_full_name: str, new_username: str,
                        new_password: str = None) -> dict:
        """
        Change name / username / password in one transaction.
        Clients and log rows point at employees.id, so a rename keeps all of them.
        Raises LookupError (unknown employee) or ValueError (empty / taken username).
        """
        new_username = (new_username or "").strip()
        if not new_username:
            raise ValueError("User ID is required")
        salt = pw_hash = None
        if new_password:
            salt = os.urandom(PASSWORD_SALT_BYTES)
            pw_hash = self._hash_password(new_password, salt)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT id FROM employees WHERE username = ? COLLATE NOCASE",
                ((old_username or "").strip(),),
            ).fetchone()
            if row is None:
                raise LookupError("Employee not found")
            emp_id = row[0]
            clash = conn.execute(
                "SELECT 1 FROM employees WHERE username = ? COLLATE NOCASE AND id != ?",
                (new_username, emp_id),
            ).fetchone()
            if clash:
                raise ValueError("This User ID is already taken")
            if pw_hash is not None:
                conn.execute(
                    "UPDATE employees SET full_name = ?, username = ?, salt = ?, password_hash = ? WHERE id = ?",
                    (new_full_name, new_username, salt, pw_hash, emp_id),
                )
            else:
                conn.execute(
                    "UPDATE employees SET full_name = ?, username = ? WHERE id = ?",
                    (new_full_name, new_username, emp_id),
                )
        return {"id": emp_id, "username": new_username, "full_name": new_full_name}

    # --------------------------------------------------------------- clients
    def save_client(self, user_id: str, display_name: str, employee_id: int, image_refs,
                    spatial_embeddings, spatial_origin: str, temporal_embeddings,
                    model_version: str) -> None:
        """
        Enroll or re-enroll a client in ONE transaction. Only the sets that are given are
        replaced: spatial_embeddings replace the `spatial_origin` set ("scan" or "render"),
        temporal_embeddings replace the temporal set; an empty list leaves that set untouched.
        image_refs are replaced when given or when a scan set is replaced.
        A row with no owner (made by add_references) is claimed by the first employee who saves.
        Raises PermissionError if another employee owns user_id, ValueError on bad input.
        """
        self._check_user(user_id)
        if spatial_origin not in SPATIAL_ORIGINS:
            raise ValueError(f"spatial_origin must be one of {SPATIAL_ORIGINS}, got {spatial_origin!r}")
        if not model_version:
            raise ValueError("model_version is required")
        spatial_embeddings = list(spatial_embeddings or [])
        temporal_embeddings = list(temporal_embeddings or [])
        now = _now_local()

        rows = []  # validate and encrypt everything before touching the DB
        for modality, origin, embs in (
            ("spatial", spatial_origin, spatial_embeddings),
            ("temporal", None, temporal_embeddings),
        ):
            for emb in embs:
                raw = self._to_raw_bytes(modality, emb)
                ct, nonce = self.encrypt_embedding(user_id, modality, origin, raw)
                rows.append((user_id, modality, origin, ct, nonce, model_version, "enroll", None, now))
        replace_refs = bool(image_refs) or (spatial_origin == "scan" and bool(spatial_embeddings))
        refs_json = json.dumps([str(p) for p in (image_refs or [])]).encode("utf-8")
        refs_ct, refs_nonce = self._encrypt_blob(self._refs_aad(user_id), refs_json)

        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT enrolled_by FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if existing is not None and existing[0] is not None and existing[0] != employee_id:
                raise PermissionError(f"Client {user_id!r} belongs to another employee")
            if spatial_embeddings:
                conn.execute(
                    "DELETE FROM enrollments WHERE user_id = ? AND modality = 'spatial' AND spatial_origin = ?",
                    (user_id, spatial_origin))
            if temporal_embeddings:
                conn.execute(
                    "DELETE FROM enrollments WHERE user_id = ? AND modality = 'temporal'", (user_id,))
            if existing is None:
                conn.execute(
                    """INSERT INTO users (user_id, display_name, enrolled_by, image_refs_enc,
                                          image_refs_nonce, spatial_origin, enrolled_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, display_name, employee_id, refs_ct, refs_nonce, spatial_origin, now, now),
                )
            else:  # enrolled_at is kept (set on first claim of an ownerless row)
                conn.execute(
                    """UPDATE users SET display_name = ?, enrolled_by = ?, spatial_origin = ?,
                              enrolled_at = COALESCE(enrolled_at, ?), updated_at = ?
                       WHERE user_id = ?""",
                    (display_name, employee_id, spatial_origin, now, now, user_id),
                )
                if replace_refs:
                    conn.execute(
                        "UPDATE users SET image_refs_enc = ?, image_refs_nonce = ? WHERE user_id = ?",
                        (refs_ct, refs_nonce, user_id))
            conn.executemany(
                """INSERT INTO enrollments
                   (user_id, modality, spatial_origin, encrypted_embedding, nonce,
                    model_version, source, expires_at, enrolled_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    _CLIENT_SELECT = """
        SELECT u.user_id, u.display_name, u.enrolled_by, e.username, u.image_refs_enc,
               u.image_refs_nonce, u.spatial_origin, u.enrolled_at, u.updated_at
        FROM users u JOIN employees e ON e.id = u.enrolled_by"""

    def _fetch_set(self, conn, user_id, modality, spatial_origin):
        """One reference set: {'embeddings', 'model_version', 'enrolled_at'} or None if empty."""
        rows = conn.execute(
            """SELECT encrypted_embedding, nonce, model_version, COALESCE(enrolled_at, created_at)
               FROM enrollments WHERE user_id = ? AND modality = ? AND spatial_origin IS ?
               AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP) ORDER BY id ASC""",
            (user_id, modality, spatial_origin)).fetchall()
        if not rows:
            return None
        return {
            "embeddings": [self.decrypt_embedding(user_id, modality, spatial_origin, r[0], r[1]) for r in rows],
            "model_version": rows[-1][2],
            "enrolled_at": rows[0][3],
        }

    def _assemble_client(self, conn, row) -> dict:
        user_id, display_name, _, owner_name, refs_ct, refs_nonce, origin, enrolled_at, updated_at = row
        refs = []
        if refs_ct is not None:
            refs = json.loads(self._decrypt_blob(self._refs_aad(user_id), refs_ct, refs_nonce).decode("utf-8"))
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
            "image_references": refs,
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
        """Decrypted client record, or None. With employee_id, only that employee's clients."""
        with closing(self.get_connection()) as conn:
            row = conn.execute(self._CLIENT_SELECT + " WHERE u.user_id = ?", (user_id,)).fetchone()
            if row is None or (employee_id is not None and row[2] != employee_id):
                return None
            return self._assemble_client(conn, row)

    def list_client_records(self, employee_id: int = None) -> list:
        sql, params = self._CLIENT_SELECT, ()
        if employee_id is not None:
            sql, params = sql + " WHERE u.enrolled_by = ?", (employee_id,)
        with closing(self.get_connection()) as conn:
            rows = conn.execute(sql + " ORDER BY u.rowid ASC", params).fetchall()
            return [self._assemble_client(conn, r) for r in rows]

    def has_client(self, user_id: str, employee_id: int = None) -> bool:
        sql, params = "SELECT 1 FROM users WHERE user_id = ? AND enrolled_by IS NOT NULL", [user_id]
        if employee_id is not None:
            sql += " AND enrolled_by = ?"
            params.append(employee_id)
        with closing(self.get_connection()) as conn:
            return conn.execute(sql, params).fetchone() is not None

    def count_clients_enrolled_in_month(self, year_month: str, employee_id: int = None) -> int:
        """year_month is 'YYYY-MM' (local time, like enrolled_at)."""
        sql = "SELECT COUNT(*) FROM users WHERE enrolled_by IS NOT NULL AND substr(enrolled_at, 1, 7) = ?"
        params = [year_month]
        if employee_id is not None:
            sql += " AND enrolled_by = ?"
            params.append(employee_id)
        with closing(self.get_connection()) as conn:
            return conn.execute(sql, params).fetchone()[0]

    # ----------------------------------------------------------------- write
    def add_references(self, user_id: str, modality: str, embeddings, model_version: str,
                       spatial_origin: str = None, expires_at=None, source: str = "enroll") -> int:
        """
        Store several templates of one kind in ONE transaction (all or nothing).
        `embeddings` is the list from enroll_*() (bytes) or np.ndarray vectors.
        """
        self._check_user(user_id)
        self._check_modality(modality, spatial_origin)
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if not model_version:
            raise ValueError("model_version is required")
        expires = self._normalize_expiry(expires_at)
        embeddings = list(embeddings)
        if not embeddings:
            raise ValueError("No embeddings given")

        rows = []
        for emb in embeddings:  # validate and encrypt everything before touching the DB
            raw = self._to_raw_bytes(modality, emb)
            ct, nonce = self.encrypt_embedding(user_id, modality, spatial_origin, raw)
            rows.append((user_id, modality, spatial_origin, ct, nonce, model_version, source, expires))

        with closing(self.get_connection()) as conn, conn:
            conn.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
            conn.executemany(
                """INSERT INTO enrollments
                   (user_id, modality, spatial_origin, encrypted_embedding, nonce,
                    model_version, source, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)

    def add_reference(self, user_id: str, modality: str, embedding, model_version: str,
                      spatial_origin: str = None, expires_at=None) -> None:
        """Store one template (bytes from enroll_*() or float32 ndarray)."""
        self.add_references(user_id, modality, [embedding], model_version,
                            spatial_origin=spatial_origin, expires_at=expires_at)

    def add_confirmed_reference(self, user_id: str, modality: str, embedding, model_version: str,
                                spatial_origin: str = None, max_per_set: int = MAX_REFERENCES_PER_SET) -> None:
        """
        Append a confirmed signing to an existing user's references.

        CALLER'S RESPONSIBILITY (the database cannot check this): call only after a
        verification with decision == "Genuine", degraded == False AND the employee's
        confirmation. Never add a rejected, degraded or unconfirmed signing.

        Keeps at most `max_per_set` templates per (user, modality, origin) set by
        dropping the oldest *added* templates first; enrolled templates are never dropped.
        """
        self._check_user(user_id)
        self._check_modality(modality, spatial_origin)
        if not model_version:
            raise ValueError("model_version is required")
        raw = self._to_raw_bytes(modality, embedding)
        ct, nonce = self.encrypt_embedding(user_id, modality, spatial_origin, raw)
        now = _now_local()
        with self._transaction() as conn:  # insert and trim together: both or neither
            if conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone() is None:
                raise ValueError(f"Unknown user {user_id!r}: enroll before adding confirmed signings")
            conn.execute(
                """INSERT INTO enrollments
                   (user_id, modality, spatial_origin, encrypted_embedding, nonce,
                    model_version, source, expires_at, enrolled_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'added', NULL, ?)""",
                (user_id, modality, spatial_origin, ct, nonce, model_version, now))
            where, params = "user_id = ? AND modality = ? AND spatial_origin IS ?", (user_id, modality, spatial_origin)
            total = conn.execute(f"SELECT COUNT(*) FROM enrollments WHERE {where}", params).fetchone()[0]
            excess = total - max_per_set
            if excess > 0:
                conn.execute(
                    f"""DELETE FROM enrollments WHERE id IN (
                            SELECT id FROM enrollments
                            WHERE {where} AND source = 'added'
                            ORDER BY id ASC LIMIT ?)""",
                    params + (excess,),
                )

    # ------------------------------------------------------------------ read
    def _fetch_refs(self, conn, user_id, modality, spatial_origin, model_version=None) -> list:
        query = """SELECT encrypted_embedding, nonce FROM enrollments
                   WHERE user_id = ? AND modality = ? AND spatial_origin IS ?
                   AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"""
        params = [user_id, modality, spatial_origin]
        if model_version is not None:
            query += " AND model_version = ?"
            params.append(model_version)
        query += " ORDER BY id ASC"
        rows = conn.execute(query, params).fetchall()
        return [self.decrypt_embedding(user_id, modality, spatial_origin, ct, nonce) for ct, nonce in rows]

    def get_references(self, user_id: str, modality: str, spatial_origin: str = None,
                       model_version: str = None) -> list:
        """
        Decrypted float32 vectors, oldest first, expired ones excluded.
        For spatial templates pass the origin you will give to verify() as
        reference_spatial_origin; render and scan sets must never be mixed.
        """
        self._check_modality(modality, spatial_origin)
        with closing(self.get_connection()) as conn:
            return self._fetch_refs(conn, user_id, modality, spatial_origin, model_version)

    def list_spatial_origins(self, user_id: str) -> list:
        """Which spatial reference sets (render / scan) the user currently holds."""
        with closing(self.get_connection()) as conn:
            rows = conn.execute(
                """SELECT DISTINCT spatial_origin FROM enrollments
                   WHERE user_id = ? AND modality = 'spatial'
                   AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                   ORDER BY spatial_origin""",
                (user_id,),
            ).fetchall()
        return [r[0] for r in rows]

    # ---------------------------------------------------------------- delete
    def revoke_reference(self, user_id: str, modality: str = None, spatial_origin: str = None) -> int:
        """Delete a user's templates (all, one modality, or one spatial origin). Returns rows deleted."""
        sql, params = "DELETE FROM enrollments WHERE user_id = ?", [user_id]
        if modality is not None:
            sql += " AND modality = ?"
            params.append(modality)
        if spatial_origin is not None:
            sql += " AND spatial_origin = ?"
            params.append(spatial_origin)
        with closing(self.get_connection()) as conn, conn:
            return conn.execute(sql, params).rowcount

    def delete_client_owned(self, user_id: str, employee_id: int) -> bool:
        """Delete a client only if employee_id owns it. Returns True if deleted."""
        with self._transaction() as conn:
            row = conn.execute("SELECT enrolled_by FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if row is None or row[0] != employee_id:
                return False
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            return True

    def delete_user(self, user_id: str) -> None:
        """Delete the user and, by cascade, all templates. The audit log is kept."""
        with closing(self.get_connection()) as conn, conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

    def purge_expired(self) -> int:
        """Physically delete expired templates (ISO/IEC 24745 retention). Returns rows deleted."""
        with closing(self.get_connection()) as conn, conn:
            return conn.execute(
                "DELETE FROM enrollments WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP"
            ).rowcount

    # ----------------------------------------------------------------- audit
    def log_verification(self, user_id: str, modality_state: str, score: float, decision: str,
                         employee_id: int = None, display_name: str = None,
                         employee_label: str = None, timestamp: str = None,
                         routing: str = None, degraded: bool = None, reasons=None,
                         model_version: str = None, threshold: float = None) -> None:
        """
        IEEE 730: record EVERY final verification event (no de-duplication).
        "Sign again" is not a verdict (step-up) and must not be logged.
        employee_label is only a fallback name for an event whose employee has no account.
        routing / degraded / reasons / model_version / threshold come from verify() and
        explain the decision later; they stay plaintext and queryable (reasons is a JSON list).
        """
        if decision == "Sign again":
            raise ValueError('"Sign again" is not a final verdict; log the final decision of the visit')
        with closing(self.get_connection()) as conn, conn:
            conn.execute(
                """INSERT INTO audit_log (user_id, client_display_name, employee_id, employee_label,
                                          modality_state, similarity_score, decision, timestamp,
                                          routing, degraded, reasons, model_version, threshold)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, display_name, employee_id, employee_label, modality_state, score,
                 decision, timestamp or _now_local(), routing,
                 None if degraded is None else int(bool(degraded)),
                 None if reasons is None else json.dumps(list(reasons)), model_version, threshold),
            )

    def get_log_events(self, employee_id: int = None, user_id: str = None,
                       newest_first: bool = False, limit: int = None) -> list:
        sql = """SELECT a.user_id, a.client_display_name, a.decision, a.similarity_score,
                        a.modality_state, COALESCE(e.username, a.employee_label, ''), a.timestamp,
                        a.routing, a.degraded, a.reasons, a.model_version, a.threshold
                 FROM audit_log a LEFT JOIN employees e ON e.id = a.employee_id"""
        where, params = [], []
        if employee_id is not None:
            where.append("a.employee_id = ?")
            params.append(employee_id)
        if user_id is not None:
            where.append("a.user_id = ?")
            params.append(user_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY a.id " + ("DESC" if newest_first else "ASC")
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
             "model_version": r[10], "threshold": r[11]}
            for r in rows
        ]

    def count_log_events(self, employee_id: int = None):
        """(total, genuine) over the final-verdict log."""
        sql = "SELECT COUNT(*), COALESCE(SUM(decision = 'Genuine'), 0) FROM audit_log"
        params = []
        if employee_id is not None:
            sql += " WHERE employee_id = ?"
            params.append(employee_id)
        with closing(self.get_connection()) as conn:
            total, genuine = conn.execute(sql, params).fetchone()
        return total, genuine
