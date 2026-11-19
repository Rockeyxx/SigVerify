"""Encryption, key slots, passwords and atomic writes of the template store (temporary DB only)."""
import hashlib
import sqlite3

import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from src.database import db_manager
from src.database.db_manager import DatabaseLockedError, DatabaseManager, probe_fingerprint

MODEL = "test-model-v1"
PW_A = "alice-password-1"
PW_B = "bob-password-22"


@pytest.fixture(autouse=True)
def fast_kdf(monkeypatch):
    monkeypatch.setattr(DatabaseManager, "KDF_N", 2 ** 10)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def db(db_path):
    """The test/CI path: SIGVER_DB_PASSPHRASE's passphrase slot unlocks the store."""
    return DatabaseManager(db_path, passphrase="pass-one")


def vec(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def file_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def two_employees(db):
    a = db.create_employee("Alice", "alice", PW_A)
    return a, db.create_employee("Bob", "bob", PW_B, created_by=a)


def slots(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT kind, employee_id, salt, wrapped_dek FROM key_slots ORDER BY id").fetchall()


# ---------------------------------------------------------------- templates
def test_embedding_saved_and_loaded_is_identical(db):
    original = vec(512, 1)
    db.add_reference("c1", "spatial", original, MODEL, spatial_origin="scan")
    (loaded,) = db.get_references("c1", "spatial", "scan")
    assert np.array_equal(loaded, original)


def test_stored_bytes_are_not_the_plain_embedding(db, db_path):
    original = vec(512, 2)
    db.add_reference("c1", "spatial", original, MODEL, spatial_origin="scan")
    with sqlite3.connect(db_path) as conn:
        (stored,) = conn.execute("SELECT encrypted_embedding FROM enrollments").fetchone()
    plain = original.tobytes()
    assert stored != plain
    assert plain not in stored
    assert plain not in file_bytes(db_path)


def test_wrong_passphrase_raises_and_keeps_the_original_key(db, db_path):
    before = slots(db_path)
    with pytest.raises(InvalidTag):
        DatabaseManager(db_path, passphrase="not-the-passphrase")
    assert slots(db_path) == before


def test_decrypting_with_the_wrong_data_key_raises(db):
    db.add_reference("c1", "spatial", vec(512, 3), MODEL, spatial_origin="scan")
    db._dek = bytes(32)
    with pytest.raises(InvalidTag):
        db.get_references("c1", "spatial", "scan")


def test_record_moved_to_another_user_raises(db, db_path):
    db.add_reference("alice-client", "spatial", vec(512, 4), MODEL, spatial_origin="scan")
    db.add_reference("bob-client", "temporal", vec(128, 5), MODEL)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE enrollments SET user_id = 'bob-client' WHERE user_id = 'alice-client'")
    with pytest.raises(InvalidTag):
        db.get_references("bob-client", "spatial", "scan")


def test_record_with_changed_origin_raises(db, db_path):
    db.add_reference("c1", "spatial", vec(512, 6), MODEL, spatial_origin="scan")
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE enrollments SET spatial_origin = 'render'")
    with pytest.raises(InvalidTag):
        db.get_references("c1", "spatial", "render")


def test_tampered_ciphertext_raises(db, db_path):
    db.add_reference("c1", "temporal", vec(128, 7), MODEL)
    with sqlite3.connect(db_path) as conn:
        (ct,) = conn.execute("SELECT encrypted_embedding FROM enrollments").fetchone()
        flipped = bytes([ct[0] ^ 1]) + ct[1:]
        conn.execute("UPDATE enrollments SET encrypted_embedding = ?", (flipped,))
    with pytest.raises(InvalidTag):
        db.get_references("c1", "temporal")


def test_data_key_is_not_stored_in_the_database(db, db_path):
    two_employees(db)
    assert db._dek not in file_bytes(db_path)


# ----------------------------------------------------- origin + model version
@pytest.mark.parametrize("origin", ["scan", "render"])
def test_spatial_origin_saved_is_the_one_returned(db, db_path, origin):
    emp, _ = two_employees(db)
    db.save_client("c1", "C1", emp, [], [vec(512, 8)], origin, [], MODEL)
    assert db.get_client_record("c1")["spatial_origin"] == origin
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT spatial_origin FROM enrollments").fetchall() == [(origin,)]


def test_model_version_is_stored_per_template(db, db_path):
    emp, _ = two_employees(db)
    db.save_client("c1", "C1", emp, [], [vec(512, 9)], "scan", [vec(128, 10)], "v-spatial-temporal-7")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT modality, model_version FROM enrollments ORDER BY id").fetchall()
    assert rows == [("spatial", "v-spatial-temporal-7"), ("temporal", "v-spatial-temporal-7")]
    assert db.get_references("c1", "temporal", model_version="other") == []


def test_spatial_and_temporal_templates_are_both_held(db):
    emp, _ = two_employees(db)
    db.save_client("tab", "Tab", emp, [], [vec(512, i) for i in range(5)], "render",
                   [vec(128, 100 + i) for i in range(5)], MODEL)
    rec = db.get_client_record("tab")
    assert len(rec["spatial"]) == 5 and rec["spatial"][0].shape == (512,)
    assert len(rec["temporal"]) == 5 and rec["temporal"][0].shape == (128,)
    assert rec["spatial_origin"] == "render"


# ----------------------------------------------------------------- passwords
def test_wrong_password_refused_right_one_accepted(db):
    db.create_employee("Alice", "alice", "correct horse battery")
    emp, reason = db.authenticate_employee("alice", "wrong horse battery")
    assert emp is None and reason == "bad_password"
    emp, reason = db.authenticate_employee("ALICE", "correct horse battery")
    assert emp["username"] == "alice" and reason == ""
    assert db.authenticate_employee("nobody", "x") == (None, "not_found")


def test_password_text_is_nowhere_in_the_database_file(db, db_path):
    secret = "S3cret-Passw0rd-marker"
    db.create_employee("Alice", "alice", secret)
    data = file_bytes(db_path)
    assert secret.encode() not in data
    assert hashlib.sha256(secret.encode()).digest() not in data


def test_no_password_hash_is_stored_only_a_wrapped_key(db, db_path):
    """The key slot is the login check: there is no cheaper hash to attack instead."""
    a, b = two_employees(db)
    with sqlite3.connect(db_path) as conn:
        hashes = conn.execute("SELECT legacy_salt, legacy_hash FROM employees").fetchall()
    assert hashes == [(None, None), (None, None)]
    rows = [r for r in slots(db_path) if r[0] == "employee"]
    assert [r[1] for r in rows] == [a, b]
    (_, _, salt_a, wrapped_a), (_, _, salt_b, wrapped_b) = rows
    assert len(salt_a) == 16 and salt_a != salt_b and wrapped_a != wrapped_b


def test_each_slot_records_its_scrypt_cost_and_the_default_is_2_17(db, db_path, monkeypatch):
    db.create_employee("A", "a", PW_A)                       # made at the test cost, 2**10
    monkeypatch.setattr(DatabaseManager, "KDF_N", 2 ** 11)   # raising the cost later...
    assert db.authenticate_employee("a", PW_A)[0] is not None   # ...still opens the old slot
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT kdf_n FROM key_slots WHERE kind = 'employee'").fetchone() == (2 ** 10,)
    with open(db_manager.__file__, encoding="utf-8") as f:
        assert "KDF_N = 2 ** 17" in f.read()                 # the shipped cost, lowered only in tests


def test_new_password_rewraps_the_slot(db, db_path):
    a = db.create_employee("A", "a", "old-password-1")
    before = slots(db_path)
    db.update_employee(a, "a", "A", "a", "new-password-2", current_password="old-password-1")
    after = slots(db_path)
    assert [r for r in after if r[0] == "employee"] != [r for r in before if r[0] == "employee"]
    assert db.authenticate_employee("a", "old-password-1")[0] is None
    assert db.authenticate_employee("a", "new-password-2")[0] is not None


def test_short_passwords_are_refused(db):
    with pytest.raises(ValueError, match="at least"):
        db.create_employee("A", "a", "short")
    a = db.create_employee("A", "a", PW_A)
    with pytest.raises(ValueError, match="at least"):
        db.update_employee(a, "a", "A", "a", "short", current_password=PW_A)
    with pytest.raises(ValueError, match="at least"):
        db.reset_password(a, "a", "short")
    assert db.authenticate_employee("a", PW_A)[0] is not None


# ------------------------------------------- own account only (account takeover)
def test_an_employee_cannot_change_another_account(db):
    a, b = two_employees(db)
    for target in ("alice", "bob"):
        actor = b if target == "alice" else a
        with pytest.raises(PermissionError, match="own account"):
            db.update_employee(actor, target, "X", target, "taken-over-pass", current_password="anything-at-all")
        with pytest.raises(PermissionError, match="own account"):
            db.update_employee(actor, target, "X", "renamed")                   # no rename either
    assert db.authenticate_employee("alice", PW_A)[0] is not None
    assert db.find_employee("bob")["full_name"] == "Bob"


def test_a_new_password_needs_the_current_one(db, db_path):
    a = db.create_employee("A", "a", PW_A)
    before = slots(db_path)
    with pytest.raises(PermissionError, match="current password"):
        db.update_employee(a, "a", "A", "a", "new-password-2")
    with pytest.raises(PermissionError, match="incorrect"):
        db.update_employee(a, "a", "A", "a", "new-password-2", current_password="wrong-password-9")
    assert slots(db_path) == before
    assert db.authenticate_employee("a", PW_A)[0] is not None


def test_a_locked_store_changes_no_profile(db):
    a = db.create_employee("A", "a", PW_A)
    db.lock()
    with pytest.raises(DatabaseLockedError):
        db.update_employee(a, "a", "Hacked", "mallory")
    assert db.find_employee("a")["full_name"] == "A" and db.find_employee("mallory") is None


def test_the_admin_cannot_reset_their_own_password(db):
    a, _ = two_employees(db)
    with pytest.raises(ValueError, match="own password"):
        db.reset_password(a, "alice", "taken-over-pass")
    assert db.authenticate_employee("alice", PW_A)[0] is not None


# ------------------------------------------------------ company-wide clients
def test_any_employee_reads_any_client_and_the_enroller_is_recorded(db):
    a, b = two_employees(db)
    original = vec(512, 11)
    db.save_client("shared-id", "Shared-ID", a, ["/p/1.png"], [original], "scan", [], MODEL)
    assert db.get_client_record("shared-id")["enrolled_by"] == "alice"
    assert db.get_client_record("shared-id", b) is None          # "enrolled by Bob": no
    assert not db.has_client("shared-id", b) and db.has_client("shared-id")
    assert [r["key"] for r in db.list_client_records()] == ["shared-id"]
    assert np.array_equal(db.get_client_record("shared-id")["spatial"][0], original)


def test_rename_keeps_clients_and_log_with_the_employee(db):
    a, b = two_employees(db)
    db.save_client("c1", "C1", a, [], [vec(512, 13)], "scan", [], MODEL)
    db.log_verification("c1", "scan", 80.0, "Genuine", employee_id=a, display_name="C1")
    db.update_employee(a, "alice", "Alice N", "alice.new", None)
    new_id = db.find_employee("alice.new")["id"]
    assert new_id == a
    assert [r["key"] for r in db.list_client_records(new_id)] == ["c1"]
    assert db.get_client_record("c1", new_id)["enrolled_by"] == "alice.new"
    events = db.get_log_events(employee_id=new_id)
    assert len(events) == 1 and events[0]["employee"] == "alice.new"
    assert db.find_employee("alice") is None
    assert db.list_client_records(b) == []


def test_rename_to_a_taken_username_is_refused(db):
    a, _ = two_employees(db)
    with pytest.raises(ValueError):
        db.update_employee(a, "alice", "Alice", "BOB", None)
    assert db.find_employee("alice") is not None


# -------------------------------- client images and name (re-enrolment, logged)
def audit(db):
    return [(e["event"], e["employee"], e["reasons"]) for e in db.get_log_events(event=None)]


def test_a_non_admin_cannot_change_a_clients_images_or_name(db):
    a, b = two_employees(db)
    db.save_client("c1", "Client", a, ["/real.png"], [vec(512, 1)], "scan", [], MODEL)
    with pytest.raises(PermissionError):
        db.save_client("c1", "CLIENT", b, ["/evil.png"], None, "scan", None, MODEL)
    rec = db.get_client_record("c1")
    assert rec["image_references"] == ["/real.png"] and rec["display_name"] == "Client"
    assert audit(db)[-1] == ("re-enrol refused", "bob", ["image references"])
    # a set the client lacks is still open to everyone, but the name stays the admin's
    db.save_client("c1", "CLIENT", b, ["/real.png"], None, "scan", [vec(128, 2)], MODEL)
    assert db.get_client_record("c1")["display_name"] == "Client"
    assert audit(db)[-1] == ("enrol", "bob", ["pen set"])


def test_a_non_admin_adds_a_scan_set_with_its_images_to_a_pen_only_client(db):
    a, b = two_employees(db)
    db.save_client("c1", "C1", a, [], None, "scan", [vec(128, 1)], MODEL)
    db.save_client("c1", "C1", b, ["/s.png"], [vec(512, 2)], "scan", None, MODEL)
    assert db.get_client_record("c1")["image_references"] == ["/s.png"]
    assert audit(db)[-1] == ("enrol", "bob", ["scan set"])


def test_the_admins_image_and_name_changes_are_logged(db):
    a, _ = two_employees(db)
    db.save_client("c1", "Client", a, ["/a.png"], [vec(512, 1)], "scan", [], MODEL)
    db.save_client("c1", "CLIENT", a, ["/b.png"], None, "scan", None, MODEL)
    rec = db.get_client_record("c1")
    assert rec["image_references"] == ["/b.png"] and rec["display_name"] == "CLIENT"
    assert audit(db)[-1] == ("re-enrol", "alice", ["image references", "display name"])
    before = len(audit(db))
    db.save_client("c1", "CLIENT", a, ["/b.png"], None, "scan", None, MODEL)   # no change
    assert len(audit(db)) == before


# ------------------------------------------------- replay of a stored reference
def test_an_enrolled_or_added_reference_given_as_a_probe_is_a_replay(db):
    a, _ = two_employees(db)
    scan, pen, added = vec(512, 1), vec(128, 2), vec(512, 3)
    db.save_client("c1", "C1", a, ["/s.png"], [scan], "scan", [pen], MODEL)
    db.add_confirmed_reference("c1", "spatial", added, MODEL, spatial_origin="scan")
    for emb in (scan, pen, added):
        assert db.is_replayed_probe("c1", [probe_fingerprint(emb)])
    assert not db.is_replayed_probe("c1", [probe_fingerprint(vec(512, 4))])
    assert not db.is_replayed_probe("c2", [probe_fingerprint(scan)])       # another client's
    db.lock()
    with pytest.raises(DatabaseLockedError):
        db.is_replayed_probe("c1", [probe_fingerprint(scan)])


# --------------------------------------------------------------- atomic save
def test_failed_save_leaves_the_old_enrollment_untouched(db):
    a, _ = two_employees(db)
    first = [vec(512, i) for i in range(5)]
    db.save_client("c1", "C1", a, ["/old.png"], first, "scan", [], MODEL)
    with pytest.raises(ValueError):
        db.save_client("c1", "C1", a, ["/new.png"], [vec(512, 50), vec(10, 51)], "scan", [], MODEL)
    rec = db.get_client_record("c1", a)
    assert rec["image_references"] == ["/old.png"]
    assert all(np.array_equal(x, y) for x, y in zip(rec["spatial"], first)) and len(rec["spatial"]) == 5


def test_error_inside_the_transaction_rolls_everything_back(db, db_path):
    a, _ = two_employees(db)
    first = [vec(512, i) for i in range(5)]
    db.save_client("c1", "C1", a, ["/old.png"], first, "scan", [], MODEL)
    with sqlite3.connect(db_path) as conn:  # a failure that only happens at INSERT time
        conn.execute("""CREATE TRIGGER boom BEFORE INSERT ON enrollments WHEN NEW.model_version = 'boom'
                        BEGIN SELECT RAISE(ABORT, 'disk on fire'); END""")
    with pytest.raises(sqlite3.DatabaseError):
        db.save_client("c1", "C1", a, ["/new.png"], [vec(512, 60)], "scan", [], "boom")
    rec = db.get_client_record("c1", a)
    assert rec["image_references"] == ["/old.png"] and len(rec["spatial"]) == 5
    assert np.array_equal(rec["spatial"][2], first[2])
    assert [e["event"] for e in db.get_log_events(event=None)] == ["enrol"]   # the log rolled back too


def test_reenrolment_replaces_old_templates_and_keeps_enrolled_at(db, db_path):
    a, _ = two_employees(db)
    db.save_client("c1", "C1", a, [], [vec(512, i) for i in range(5)], "scan", [], MODEL)
    with sqlite3.connect(db_path) as conn:  # backdate, so "kept" cannot be confused with "same second"
        conn.execute("UPDATE users SET enrolled_at = '2020-01-02T03:04:05', updated_at = '2020-01-02T03:04:05'")
    new = [vec(512, 200 + i) for i in range(5)]
    db.save_client("c1", "c1 renamed", a, [], new, "scan", [], MODEL)
    rec = db.get_client_record("c1")
    assert len(rec["spatial"]) == 5
    assert np.array_equal(rec["spatial"][0], new[0])
    assert rec["enrolled_at"] == "2020-01-02T03:04:05"
    assert rec["updated_at"] != "2020-01-02T03:04:05"
    assert rec["display_name"] == "c1 renamed"


def test_image_references_are_bound_to_their_client(db, db_path):
    a, _ = two_employees(db)
    db.save_client("a", "A", a, ["/a.png"], [vec(512, 1)], "scan", [], MODEL)
    db.save_client("b", "B", a, ["/b.png"], [vec(512, 2)], "scan", [], MODEL)
    with sqlite3.connect(db_path) as conn:   # move A's encrypted image list onto B
        ct, nonce = conn.execute("SELECT image_refs_enc, image_refs_nonce FROM users WHERE user_id = 'a'").fetchone()
        conn.execute("UPDATE users SET image_refs_enc = ?, image_refs_nonce = ? WHERE user_id = 'b'", (ct, nonce))
    with pytest.raises(InvalidTag):
        db.get_client_record("b")


def test_a_passphrase_never_creates_a_second_key_on_a_password_database(tmp_path):
    path = str(tmp_path / "s.db")
    DatabaseManager(path).create_employee("Admin", "admin", PW_A)
    before = slots(path)
    with pytest.raises(ValueError, match="employee passwords"):
        DatabaseManager(path, passphrase="ci-secret")
    assert slots(path) == before
