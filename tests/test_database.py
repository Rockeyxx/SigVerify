"""Issue #33 at the database level: key slots per employee, the admin role, company-wide
clients with an audit trail, the pen set kept on re-save, stale sets, the replay check,
backup / restore, and the migration of v3 / v4 databases. Temporary databases only."""
import os
import shutil
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from config.settings import MAX_REFERENCES, REPLAY_WINDOW_DAYS
from src.database.db_manager import (
    DatabaseLockedError,
    DatabaseManager,
    probe_fingerprint,
)

V1 = "v-old"
PW_ADMIN = "admin-password-1"
PW_BOB = "bob-password-123"
PW_CAROL = "carol-password-1"


@pytest.fixture(autouse=True)
def fast_kdf(monkeypatch):
    monkeypatch.setattr(DatabaseManager, "KDF_N", 2 ** 10)


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "signatures.db")


@pytest.fixture
def db(path):
    """The production path: no passphrase, the first account creates the key."""
    return DatabaseManager(path)


@pytest.fixture
def staff(db):
    admin = db.create_employee("Admin", "admin", PW_ADMIN)
    bob = db.create_employee("Bob", "bob", PW_BOB, created_by=admin)
    return admin, bob


def vec(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def scan(seed=0, n=5):
    return [vec(512, seed + i) for i in range(n)]


def pen(seed=100, n=5):
    return [vec(128, seed + i) for i in range(n)]


def slot_rows(path):
    with sqlite3.connect(path) as conn:
        return conn.execute(
            "SELECT employee_id, salt, wrapped_dek FROM key_slots WHERE kind = 'employee' ORDER BY employee_id"
        ).fetchall()


def file_bytes(p):
    with open(p, "rb") as f:
        return f.read()


def events(db, **kw):
    return [(e["event"], e["client_key"], e["employee"]) for e in db.get_log_events(event=None, **kw)]


# ------------------------------------------------------------ key slots (item 6)
def test_a_new_database_opens_locked_and_the_first_account_is_the_admin(db, path):
    assert not db.unlocked
    with pytest.raises(DatabaseLockedError):
        db.list_client_records()
    admin = db.create_employee("Admin", "admin", PW_ADMIN)
    assert db.unlocked and db.find_employee("admin")["is_admin"] is True
    bob = db.create_employee("Bob", "bob", PW_BOB, created_by=admin)
    assert db.find_employee("bob")["is_admin"] is False
    with sqlite3.connect(path) as conn:
        kinds = conn.execute("SELECT kind, employee_id FROM key_slots ORDER BY id").fetchall()
    assert kinds == [("employee", admin), ("employee", bob)]      # no passphrase slot, no key file


def test_a_copied_database_cannot_be_opened_without_a_password(db, staff, path, tmp_path):
    db.save_client("c1", "C1", staff[0], spatial_embeddings=scan(), spatial_origin="scan", temporal_embeddings=pen(), model_version=V1)
    copy = str(tmp_path / "stolen.db")
    shutil.copyfile(path, copy)
    thief = DatabaseManager(copy)
    assert not thief.unlocked
    with pytest.raises(DatabaseLockedError):
        thief.get_client_record("c1")
    assert thief.authenticate_employee("admin", "guess-guess-guess") == (None, "bad_password")
    assert db._dek not in file_bytes(copy)
    assert PW_ADMIN.encode() not in file_bytes(copy) and PW_BOB.encode() not in file_bytes(copy)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".key")]


def test_login_unwraps_the_key_and_logout_forgets_it(db, staff, path):
    db.save_client("c1", "C1", staff[1], spatial_embeddings=scan(), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    fresh = DatabaseManager(path)
    emp, reason = fresh.authenticate_employee("BOB", PW_BOB)
    assert reason == "" and emp["username"] == "bob" and emp["is_admin"] is False
    assert len(fresh.get_client_record("c1")["spatial"]) == 5
    fresh.lock()
    with pytest.raises(DatabaseLockedError):
        fresh.get_client_record("c1")
    assert fresh.authenticate_employee("bob", PW_ADMIN) == (None, "bad_password")


def test_only_the_admin_adds_employees(db, staff):
    admin, bob = staff
    with pytest.raises(PermissionError):
        db.create_employee("Carol", "carol", PW_CAROL, created_by=bob)
    with pytest.raises(PermissionError):
        db.create_employee("Carol", "carol", PW_CAROL)
    assert db.find_employee("carol") is None


def test_admin_add_reset_remove_each_change_only_that_slot(db, staff, path):
    admin, bob = staff
    before = slot_rows(path)
    carol = db.create_employee("Carol", "carol", PW_CAROL, created_by=admin)
    after_add = slot_rows(path)
    assert after_add[:2] == before and after_add[2][0] == carol

    db.reset_password(admin, "bob", "bob-new-password")          # Bob forgot his password
    after_reset = slot_rows(path)
    assert after_reset[0] == before[0] and after_reset[2] == after_add[2]
    assert after_reset[1] != before[1]
    assert db.authenticate_employee("bob", PW_BOB)[0] is None
    assert db.authenticate_employee("bob", "bob-new-password")[0] is not None

    db.remove_employee(admin, "carol")
    assert slot_rows(path) == after_reset[:2]
    assert db.authenticate_employee("carol", PW_CAROL) == (None, "not_found")
    assert db.find_employee("carol") is None


def test_a_non_admin_cannot_reset_or_remove_and_the_admin_cannot_remove_themself(db, staff):
    admin, bob = staff
    with pytest.raises(PermissionError):
        db.reset_password(bob, "admin", "taken-over-pass")
    with pytest.raises(PermissionError):
        db.remove_employee(bob, "admin")
    with pytest.raises(ValueError):
        db.remove_employee(admin, "admin")
    assert db.authenticate_employee("admin", PW_ADMIN)[0] is not None


def test_a_reset_password_opens_the_same_templates(db, staff, path):
    admin, _ = staff
    db.save_client("c1", "C1", admin, spatial_embeddings=scan(7), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    db.reset_password(admin, "bob", "bob-new-password")
    later = DatabaseManager(path)
    assert later.authenticate_employee("bob", "bob-new-password")[0] is not None
    assert np.array_equal(later.get_client_record("c1")["spatial"][0], scan(7)[0])


# --------------------------------------------- company-wide clients, admin role
def test_anyone_enrols_a_new_client_or_adds_a_missing_set(db, staff):
    admin, bob = staff
    db.save_client("c1", "C1", admin, spatial_embeddings=scan(), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    db.save_client("c1", "C1", bob, spatial_embeddings=[], spatial_origin="scan", temporal_embeddings=pen(), model_version=V1)         # Bob adds the pen set
    rec = db.get_client_record("c1")
    assert rec["enrolled_by"] == "admin" and len(rec["temporal"]) == 5
    assert events(db) == [("enrol", "c1", "admin"), ("enrol", "c1", "bob")]


def test_a_non_admin_reenrol_or_delete_is_refused_logged_and_changes_nothing(db, staff):
    admin, bob = staff
    db.save_client("c1", "C1", admin, spatial_embeddings=scan(0), spatial_origin="scan", temporal_embeddings=pen(100), model_version=V1)
    with pytest.raises(PermissionError):
        db.save_client("c1", "C1", bob, spatial_embeddings=scan(50), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    with pytest.raises(PermissionError):
        db.save_client("c1", "C1", bob, spatial_embeddings=[], spatial_origin="scan", temporal_embeddings=[], model_version=V1)         # removing the pen set
    with pytest.raises(PermissionError):
        db.delete_client("c1", bob)
    rec = db.get_client_record("c1")
    assert np.array_equal(rec["spatial"][0], scan(0)[0]) and len(rec["temporal"]) == 5
    assert events(db) == [("enrol", "c1", "admin"), ("re-enrol refused", "c1", "bob"),
                          ("re-enrol refused", "c1", "bob"), ("delete refused", "c1", "bob")]


def test_the_admin_reenrols_and_verifies_a_client_bob_enrolled_and_both_are_logged(db, staff):
    admin, bob = staff
    db.save_client("c1", "C1", bob, spatial_embeddings=scan(0), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    db.log_verification("c1", "scan", 81.0, "Genuine", employee_id=admin, display_name="C1")
    db.save_client("c1", "C1", admin, spatial_embeddings=scan(50), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    rec = db.get_client_record("c1")
    assert np.array_equal(rec["spatial"][0], scan(50)[0]) and rec["enrolled_by"] == "bob"
    assert events(db) == [("enrol", "c1", "bob"), ("verify", "c1", "admin"), ("re-enrol", "c1", "admin")]
    assert db.get_log_events(event=None)[2]["reasons"] == ["scan set"]


def test_a_deleted_clients_audit_rows_stay(db, staff):
    admin, bob = staff
    db.save_client("gone", "Gone", bob, spatial_embeddings=scan(), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    db.log_verification("gone", "scan", 70.0, "Forged", employee_id=bob, display_name="Gone")
    assert db.delete_client("gone", admin) is True
    assert db.get_client_record("gone") is None
    assert events(db, user_id="gone") == [("enrol", "gone", "bob"), ("verify", "gone", "bob"),
                                          ("delete", "gone", "admin")]
    assert db.delete_client("gone", admin) is False


def test_verification_counts_and_lists_ignore_enrolment_events(db, staff):
    admin, bob = staff
    db.save_client("c1", "C1", bob, spatial_embeddings=scan(), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    db.log_verification("c1", "scan", 81.0, "Genuine", employee_id=bob)
    assert db.count_log_events() == (1, 1)
    assert [e["event"] for e in db.get_log_events()] == ["verify"]


def test_an_ownerless_row_starts_clean_when_first_saved(db, staff):
    _, bob = staff
    db._add_references("orphan", "spatial", scan(900), "older", spatial_origin="render")
    db.save_client("orphan", "Orphan", bob, spatial_embeddings=scan(0), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    rec = db.get_client_record("orphan")
    assert rec["enrolled_by"] == "bob" and set(rec["sets"]) == {"scan"}   # nothing inherited
    assert events(db) == [("enrol", "orphan", "bob")]


def test_a_new_client_with_nothing_to_save_is_refused(db, staff):
    with pytest.raises(ValueError, match="Nothing to save"):
        db.save_client("c1", "C1", staff[0], spatial_embeddings=[], spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    assert not db.has_client("c1")


def test_a_confirmed_signing_needs_an_existing_set_of_the_current_weights(db, staff, path):
    admin, _ = staff
    with pytest.raises(LookupError):
        db.add_confirmed_reference("ghost", "spatial", vec(512, 1), V1, spatial_origin="scan")
    db.save_client("c1", "C1", admin, spatial_embeddings=scan(0), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    with pytest.raises(LookupError):
        db.add_confirmed_reference("c1", "temporal", vec(128, 1), V1)
    with pytest.raises(ValueError):
        db.add_confirmed_reference("c1", "spatial", vec(512, 1), "v-new", spatial_origin="scan")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM enrollments").fetchone() == (5,)


def test_a_given_log_timestamp_is_stored_in_seconds(db, staff):
    db.log_verification("c1", "scan", 90.0, "Genuine", employee_id=staff[0],
                        timestamp="2026-11-20 10:11:12.345678")
    assert db.get_log_events()[-1]["timestamp"] == "2026-11-20T10:11:12"
    with pytest.raises(ValueError):
        db.log_verification("c1", "scan", 90.0, "Genuine", employee_id=staff[0], timestamp="yesterday")


# ---------------------------------------------------- pen set kept on re-save
def test_resaving_with_no_new_pen_set_keeps_its_rows_exactly(db, staff, path):
    admin, _ = staff
    db.save_client("c1", "C1", admin, spatial_embeddings=scan(0), spatial_origin="scan", temporal_embeddings=pen(100), model_version=V1)
    db.add_confirmed_reference("c1", "temporal", vec(128, 999), V1)
    with sqlite3.connect(path) as conn:
        before = conn.execute("SELECT id, source, model_version FROM enrollments "
                              "WHERE modality = 'temporal' ORDER BY id").fetchall()
    db.save_client("c1", "C1", admin, spatial_embeddings=scan(50), spatial_origin="scan", temporal_embeddings=None, model_version="v-new")
    with sqlite3.connect(path) as conn:
        after = conn.execute("SELECT id, source, model_version FROM enrollments "
                             "WHERE modality = 'temporal' ORDER BY id").fetchall()
    assert after == before and [r[1] for r in after] == ["enroll"] * 5 + ["added"]
    db.add_confirmed_reference("c1", "temporal", vec(128, 1000), V1)
    assert len(db.get_references("c1", "temporal")) == 7 <= MAX_REFERENCES   # the next one is kept


def test_an_empty_pen_list_removes_the_pen_set(db, staff):
    admin, _ = staff
    db.save_client("c1", "C1", admin, spatial_embeddings=scan(0), spatial_origin="scan", temporal_embeddings=pen(100), model_version=V1)
    db.save_client("c1", "C1", admin, spatial_embeddings=[], spatial_origin="scan", temporal_embeddings=[], model_version=V1)
    rec = db.get_client_record("c1")
    assert rec["temporal_set"] is None and len(rec["spatial"]) == 5
    assert db.get_log_events(event=None)[-1]["reasons"] == ["pen set removed"]


def test_capture_device_is_stored_with_the_pen_set(db, staff):
    admin, _ = staff
    device = {"device_model": "XPPen Deco 640", "os": "Windows", "qt_platform": "windows"}
    db.save_client("c1", "C1", admin, spatial_embeddings=[], spatial_origin="scan", temporal_embeddings=pen(), model_version=V1, capture_device=device)
    db.add_confirmed_reference("c1", "temporal", vec(128, 5), V1)
    assert db.get_client_record("c1")["temporal_set"]["capture_device"] == device


def test_a_set_lists_every_model_version_it_holds(db, staff, path):
    admin, _ = staff
    db.save_client("c1", "C1", admin, spatial_embeddings=scan(0), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    db.add_confirmed_reference("c1", "spatial", vec(512, 77), V1, spatial_origin="scan")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE enrollments SET model_version = 'v-new' WHERE id = (SELECT MAX(id) FROM enrollments)")
    found = db.get_client_record("c1")["sets"]["scan"]
    assert found["model_version"] == V1 and found["model_versions"] == ["v-new", V1]


# --------------------------------------------------------- replay (item 8)
def test_the_same_probe_twice_is_refused_the_second_time(db, staff):
    admin, _ = staff
    probe = probe_fingerprint(vec(512, 1))
    assert not db.is_replayed_probe("c1", [probe])
    db.log_verification("c1", "scan", 90.0, "Genuine", employee_id=admin, probe_hashes=[probe])
    assert db.is_replayed_probe("c1", [probe])


def test_two_different_genuine_probes_are_both_accepted(db, staff):
    admin, _ = staff
    first, second = probe_fingerprint(vec(512, 1)), probe_fingerprint(vec(512, 2))
    assert not db.is_replayed_probe("c1", [first])
    db.log_verification("c1", "scan", 90.0, "Genuine", employee_id=admin, probe_hashes=[first])
    assert not db.is_replayed_probe("c1", [second])


def test_replay_counts_only_accepted_probes_of_the_same_client_inside_the_window(db, staff):
    admin, _ = staff
    probe = probe_fingerprint(vec(512, 3))
    db.log_verification("c1", "scan", 40.0, "Forged", employee_id=admin, probe_hashes=[probe])
    assert not db.is_replayed_probe("c1", [probe])                      # it was refused, not accepted
    db.log_verification("c2", "scan", 90.0, "Genuine", employee_id=admin, probe_hashes=[probe])
    assert not db.is_replayed_probe("c1", [probe])                      # another client
    old = (datetime.now() - timedelta(days=REPLAY_WINDOW_DAYS + 1)).isoformat(timespec="seconds")
    db.log_verification("c1", "scan", 90.0, "Genuine", employee_id=admin, timestamp=old,
                        probe_hashes=[probe])
    assert not db.is_replayed_probe("c1", [probe])                      # outside the window
    assert db.is_replayed_probe("c1", [probe], window_days=REPLAY_WINDOW_DAYS + 2)


def test_one_signing_given_twice_in_a_visit_is_a_replay(db):
    probe = probe_fingerprint(vec(512, 4))
    assert db.is_replayed_probe("c1", [probe, probe])
    assert not db.is_replayed_probe("c1", [])


def test_the_log_stores_the_hash_never_the_embedding(db, staff, path):
    embedding = vec(512, 5)
    db.log_verification("c1", "scan", 90.0, "Genuine", employee_id=staff[0],
                        probe_hashes=[probe_fingerprint(embedding)])
    assert embedding.tobytes() not in file_bytes(path)
    assert probe_fingerprint(embedding) == probe_fingerprint(embedding.tobytes())


# -------------------------------------------------- backup / restore (item 9)
def test_backup_holds_no_key_or_password_and_opens_with_the_same_passwords(db, staff, path, tmp_path):
    db.save_client("c1", "C1", staff[0], spatial_embeddings=scan(3), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    backup = db.backup_to(staff[0], str(tmp_path / "backup.db"))
    data = file_bytes(backup)
    assert db._dek not in data and PW_ADMIN.encode() not in data and PW_BOB.encode() not in data
    opened = DatabaseManager(backup)
    assert opened.authenticate_employee("bob", PW_BOB)[0] is not None
    assert np.array_equal(opened.get_client_record("c1")["spatial"][0], scan(3)[0])
    with pytest.raises(FileExistsError):
        db.backup_to(staff[0], backup)


def test_restore_brings_the_backup_back(db, staff, path, tmp_path):
    admin, _ = staff
    db.save_client("c1", "C1", admin, spatial_embeddings=scan(3), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    backup = db.backup_to(admin, str(tmp_path / "backup.db"))
    db.delete_client("c1", admin)
    emp = db.restore_from(backup, "admin", PW_ADMIN)
    assert emp["is_admin"] and np.array_equal(db.get_client_record("c1")["spatial"][0], scan(3)[0])


def test_restore_checks_everything_before_replacing_anything(db, staff, path, tmp_path):
    admin, _ = staff
    db.save_client("c1", "C1", admin, spatial_embeddings=scan(3), spatial_origin="scan", temporal_embeddings=None, model_version=V1)
    backup = db.backup_to(admin, str(tmp_path / "backup.db"))
    db.save_client("c2", "C2", admin, spatial_embeddings=scan(9), spatial_origin="scan", temporal_embeddings=None, model_version=V1)   # only in the live database
    with pytest.raises(PermissionError):
        db.restore_from(backup, "admin", "wrong-password-1")
    with pytest.raises(PermissionError, match="admin"):
        db.restore_from(backup, "bob", PW_BOB)                         # not an admin of the backup
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"this is not a database at all" * 100)
    with pytest.raises(ValueError):
        db.restore_from(str(junk), "admin", PW_ADMIN)
    with pytest.raises(ValueError):
        db.restore_from(str(tmp_path / "missing.db"), "admin", PW_ADMIN)
    assert db.has_client("c2")                                         # nothing was replaced
    assert DatabaseManager(path).authenticate_employee("admin", PW_ADMIN)[0] is not None


# ------------------------------------------------------ old databases (v3 / v4)
LEGACY_KEY = "legacy-key-file-contents"
OLD_A = "old-a"                  # a v4 password too short for a key slot (v4 had no minimum)
OLD_B = "old-bob-password"       # a v4 password long enough to become the key slot


def _make_v4(path, legacy_passphrase=LEGACY_KEY, with_meta=True):
    """A database exactly as the v4 code left it: key wrapped in `meta` by the key-file
    passphrase, employees with salted scrypt password hashes, one encrypted template."""
    import hashlib
    dek = AESGCM.generate_key(256)
    salt, nonce = os.urandom(16), os.urandom(12)
    wrap_key = Scrypt(salt=salt, length=32, n=DatabaseManager.KDF_N, r=8, p=1).derive(legacy_passphrase.encode())
    wrapped = AESGCM(wrap_key).encrypt(nonce, dek, b"sigverify-dek-v1")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE meta (id INTEGER PRIMARY KEY CHECK (id = 1), salt BLOB NOT NULL,
                           wrapped_dek BLOB NOT NULL, nonce BLOB NOT NULL);
        CREATE TABLE employees (id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE, full_name TEXT NOT NULL,
            salt BLOB NOT NULL, password_hash BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE users (user_id TEXT PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            display_name TEXT, enrolled_by INTEGER REFERENCES employees(id),
            spatial_origin TEXT CHECK (spatial_origin IS NULL OR spatial_origin IN ('render', 'scan')),
            enrolled_at TEXT, updated_at TEXT);
        CREATE TABLE enrollments (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
            modality TEXT NOT NULL CHECK (modality IN ('spatial', 'temporal')),
            spatial_origin TEXT CHECK (spatial_origin IN ('render', 'scan')),
            encrypted_embedding BLOB NOT NULL, nonce BLOB NOT NULL, model_version TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'enroll' CHECK (source IN ('enroll', 'added')),
            expires_at TIMESTAMP, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, enrolled_at TEXT,
            CHECK ((modality = 'spatial' AND spatial_origin IS NOT NULL)
                OR (modality = 'temporal' AND spatial_origin IS NULL)),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE);
        CREATE TABLE audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
            client_display_name TEXT, employee_id INTEGER REFERENCES employees(id),
            employee_label TEXT, modality_state TEXT, similarity_score REAL, decision TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, routing TEXT, degraded INTEGER,
            reasons TEXT, model_version TEXT, threshold REAL);
        PRAGMA user_version = 4;
    """)
    if with_meta:
        conn.execute("INSERT INTO meta VALUES (1, ?, ?, ?)", (salt, wrapped, nonce))
    else:
        conn.execute("DROP TABLE meta")
    for name, pw in (("alice", OLD_A), ("bob", OLD_B)):
        s = os.urandom(16)
        h = hashlib.scrypt(pw.encode(), salt=s, n=2 ** 14, r=8, p=1, dklen=32)
        conn.execute("INSERT INTO employees (username, full_name, salt, password_hash) VALUES (?, ?, ?, ?)",
                     (name, name.title(), s, h))
    conn.execute("INSERT INTO users (user_id, display_name, enrolled_by, enrolled_at) VALUES ('c1', 'C1', 1, '2025-01-01T00:00:00')")
    template = vec(512, 42)
    n = os.urandom(12)
    ct = AESGCM(dek).encrypt(n, template.tobytes(), b"c1|spatial|scan")
    conn.execute("INSERT INTO enrollments (user_id, modality, spatial_origin, encrypted_embedding, nonce, "
                 "model_version, expires_at) VALUES ('c1', 'spatial', 'scan', ?, ?, 'v0.9', NULL)", (ct, n))
    conn.execute("INSERT INTO audit_log (user_id, decision, similarity_score, employee_id) VALUES ('c1', 'Genuine', 88.0, 2)")
    conn.commit()
    conn.close()
    return template


def _v4_to_v3(path):
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE enrollments DROP COLUMN enrolled_at")
        for col in ("routing", "degraded", "reasons", "model_version", "threshold"):
            conn.execute(f"ALTER TABLE audit_log DROP COLUMN {col}")
        conn.execute("PRAGMA user_version = 3")


@pytest.mark.parametrize("from_version", [4, 3])
def test_an_old_database_is_refused_and_left_unchanged(path, from_version):
    _make_v4(path)
    if from_version == 3:
        _v4_to_v3(path)
    with open(path, "rb") as f:
        before = f.read()
    with pytest.raises(RuntimeError, match=f"schema v{from_version} is older than v5 and is not supported"):
        DatabaseManager(path)
    with open(path, "rb") as f:
        assert f.read() == before


def test_an_unknown_schema_version_is_refused(path):
    _make_v4(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 2")
    with pytest.raises(RuntimeError, match="schema v2 is older than v5"):
        DatabaseManager(path)
