"""Encryption, key slots, passwords and atomic writes of the template store (temporary DB only)."""
import hashlib
import sqlite3

import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from src.database.db_manager import DatabaseLockedError, DatabaseManager, probe_fingerprint
from tests.vault_file import edit_live, on_disk

SHIPPED_KDF_N = DatabaseManager.KDF_N   # read at import, before the autouse fixture lowers it
MODEL = "test-model-v1"
PW_A = "alice-password-1-long"
PW_B = "bob-password-22-long"


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


def slots(db_path, dek):
    with on_disk(db_path, dek) as conn:
        return conn.execute("SELECT kind, employee_id, salt, wrapped_dek FROM key_slots ORDER BY id").fetchall()


# ---------------------------------------------------------------- templates
def test_embedding_saved_and_loaded_is_identical(db):
    original = vec(512, 1)
    db._add_references("c1", "spatial", [original], MODEL, spatial_origin="scan")
    (loaded,) = db.get_references("c1", "spatial", "scan")
    assert np.array_equal(loaded, original)


def test_stored_bytes_are_not_the_plain_embedding(db, db_path):
    original = vec(512, 2)
    db._add_references("c1", "spatial", [original], MODEL, spatial_origin="scan")
    with on_disk(db_path, db._dek) as conn:
        (stored,) = conn.execute("SELECT encrypted_embedding FROM enrollments").fetchone()
    plain = original.tobytes()
    assert stored != plain
    assert plain not in stored
    assert plain not in file_bytes(db_path)


def test_wrong_passphrase_raises_and_keeps_the_original_key(db, db_path):
    before = slots(db_path, db._dek)
    with pytest.raises(InvalidTag):
        DatabaseManager(db_path, passphrase="not-the-passphrase")
    assert slots(db_path, db._dek) == before


def test_decrypting_with_the_wrong_data_key_raises(db):
    db._add_references("c1", "spatial", [vec(512, 3)], MODEL, spatial_origin="scan")
    db._dek = bytes(32)
    with pytest.raises(InvalidTag):
        db.get_references("c1", "spatial", "scan")


def test_record_moved_to_another_user_raises(db, db_path):
    db._add_references("alice-client", "spatial", [vec(512, 4)], MODEL, spatial_origin="scan")
    db._add_references("bob-client", "temporal", [vec(128, 5)], MODEL)
    with edit_live(db) as conn:
        conn.execute("UPDATE enrollments SET user_id = 'bob-client' WHERE user_id = 'alice-client'")
    with pytest.raises(InvalidTag):
        db.get_references("bob-client", "spatial", "scan")


def test_record_with_changed_origin_raises(db, db_path):
    db._add_references("c1", "spatial", [vec(512, 6)], MODEL, spatial_origin="scan")
    with edit_live(db) as conn:
        conn.execute("UPDATE enrollments SET spatial_origin = 'render'")
    with pytest.raises(InvalidTag):
        db.get_references("c1", "spatial", "render")


def test_tampered_ciphertext_raises(db, db_path):
    db._add_references("c1", "temporal", [vec(128, 7)], MODEL)
    with edit_live(db) as conn:
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
    db.save_client("c1", "C1", emp, spatial_embeddings=[vec(512, 8)], spatial_origin=origin, temporal_embeddings=[], model_version=MODEL)
    assert db.get_client_record("c1")["spatial_origin"] == origin
    with on_disk(db_path, db._dek) as conn:
        assert conn.execute("SELECT spatial_origin FROM enrollments").fetchall() == [(origin,)]


def test_model_version_is_stored_per_template(db, db_path):
    emp, _ = two_employees(db)
    db.save_client("c1", "C1", emp, spatial_embeddings=[vec(512, 9)], spatial_origin="scan", temporal_embeddings=[vec(128, 10)], model_version="v-spatial-temporal-7")
    with on_disk(db_path, db._dek) as conn:
        rows = conn.execute("SELECT modality, model_version FROM enrollments ORDER BY id").fetchall()
    assert rows == [("spatial", "v-spatial-temporal-7"), ("temporal", "v-spatial-temporal-7")]
    assert db.get_references("c1", "temporal", model_version="other") == []


def test_spatial_and_temporal_templates_are_both_held(db):
    emp, _ = two_employees(db)
    db.save_client("tab", "Tab", emp, spatial_embeddings=[vec(512, i) for i in range(5)], spatial_origin="render", temporal_embeddings=[vec(128, 100 + i) for i in range(5)], model_version=MODEL)
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
    with on_disk(db_path, db._dek) as conn:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(employees)")]
    assert columns and not {"salt", "password_hash", "legacy_salt", "legacy_hash"} & set(columns)
    rows = [r for r in slots(db_path, db._dek) if r[0] == "employee"]
    assert [r[1] for r in rows] == [a, b]
    (_, _, salt_a, wrapped_a), (_, _, salt_b, wrapped_b) = rows
    assert len(salt_a) == 16 and salt_a != salt_b and wrapped_a != wrapped_b


def test_each_slot_records_its_scrypt_cost_and_the_default_is_2_17(db, db_path, monkeypatch):
    db.create_employee("A", "a", PW_A)                       # made at the test cost, 2**10
    monkeypatch.setattr(DatabaseManager, "KDF_N", 2 ** 11)   # raising the cost later...
    assert db.authenticate_employee("a", PW_A)[0] is not None   # ...still opens the old slot
    with on_disk(db_path, db._dek) as conn:
        assert conn.execute("SELECT kdf_n FROM key_slots WHERE kind = 'employee'").fetchone() == (2 ** 10,)
    assert SHIPPED_KDF_N == 2 ** 17                          # the shipped cost, lowered only in tests


def _another_session(db, sql, params=()):
    with edit_live(db) as conn:
        conn.execute(sql, params)


def test_create_employee_rechecks_the_creator_inside_the_transaction(db, db_path, monkeypatch):
    a, b = two_employees(db)                                 # bob is not admin; make him one, then demote
    _another_session(db, "UPDATE employees SET is_admin = 1 WHERE id = ?", (b,))
    real_wrap = db._wrap

    def wrap_then_demote(*args):
        slot = real_wrap(*args)
        _another_session(db, "UPDATE employees SET is_admin = 0 WHERE id = ?", (b,))
        return slot

    monkeypatch.setattr(db, "_wrap", wrap_then_demote)
    with pytest.raises(PermissionError):
        db.create_employee("Carol", "carol", "carol-password-1", created_by=b)
    assert [e["username"] for e in db.list_employees()] == ["alice", "bob"]


def test_update_employee_refuses_when_the_slot_changed_meanwhile(db, db_path, monkeypatch):
    a, _ = two_employees(db)
    real_wrap = db._wrap

    def wrap_then_change_slot(*args):
        slot = real_wrap(*args)
        _another_session(db, "UPDATE key_slots SET salt = ? WHERE employee_id = ?",
                                   (b"x" * 16, a))        # another session changed alice's slot
        return slot

    monkeypatch.setattr(db, "_wrap", wrap_then_change_slot)
    with pytest.raises(ValueError, match="Your password was changed meanwhile. Try again."):
        db.update_employee(a, "alice", "Alice", "alice", "new-password-33", current_password=PW_A)
    with on_disk(db_path, db._dek) as conn:
        assert conn.execute("SELECT salt FROM key_slots WHERE employee_id = ?", (a,)).fetchone() == (b"x" * 16,)


def test_only_an_active_admin_backs_up(db, tmp_path):
    a, b = two_employees(db)
    with pytest.raises(PermissionError):
        db.backup_to(b, str(tmp_path / "b.db"))
    assert not (tmp_path / "b.db").exists()
    assert db.backup_to(a, str(tmp_path / "a.db")) == str(tmp_path / "a.db")


def test_the_backup_is_created_private_and_never_over_a_file(db, tmp_path):
    a, _ = two_employees(db)
    target = tmp_path / "a.db"
    target.write_bytes(b"mine")
    with pytest.raises(FileExistsError):
        db.backup_to(a, str(target))
    assert target.read_bytes() == b"mine"


def test_new_password_rewraps_the_slot(db, db_path):
    a = db.create_employee("A", "a", "old-password-1-long")
    before = slots(db_path, db._dek)
    db.update_employee(a, "a", "A", "a", "new-password-2-long", current_password="old-password-1-long")
    after = slots(db_path, db._dek)
    assert [r for r in after if r[0] == "employee"] != [r for r in before if r[0] == "employee"]
    assert db.authenticate_employee("a", "old-password-1-long")[0] is None
    assert db.authenticate_employee("a", "new-password-2-long")[0] is not None


def test_passwords_under_15_characters_are_refused(db):
    fourteen = "abcdefghijklmn"
    with pytest.raises(ValueError, match="at least 15"):
        db.create_employee("A", "a", fourteen)
    a = db.create_employee("A", "a", PW_A)
    with pytest.raises(ValueError, match="at least 15"):
        db.update_employee(a, "a", "A", "a", fourteen, current_password=PW_A)
    with pytest.raises(ValueError, match="at least 15"):
        db.reset_password(a, "a", fourteen)
    assert db.authenticate_employee("a", PW_A)[0] is not None


@pytest.mark.parametrize("common", ["iloveyouforever", "PasswordPassword", "1qaz2wsx3edc4rfv"])
def test_common_passwords_are_refused_whatever_the_case(db, common):
    with pytest.raises(ValueError, match="too common"):
        db.create_employee("A", "a", common)


def test_a_long_passphrase_needs_no_capitals_digits_or_symbols(db):
    assert db.create_employee("A", "a", "correct horse battery staple")


def test_a_password_set_under_the_old_rule_still_logs_in(db, monkeypatch):
    import src.database.db_manager as dbm
    rule = dbm.MIN_PASSWORD_LENGTH
    monkeypatch.setattr(dbm, "MIN_PASSWORD_LENGTH", 12)
    db.create_employee("A", "a", "twelve-chars")
    monkeypatch.setattr(dbm, "MIN_PASSWORD_LENGTH", rule)
    assert db.authenticate_employee("a", "twelve-chars")[0] is not None


def test_password_problem_names_what_is_missing():
    from src.database.db_manager import password_problem
    assert "at least 15" in password_problem("short")
    assert "too common" in password_problem("passwordpassword")
    assert password_problem("correct horse battery staple") is None


def test_generated_passwords_meet_the_rule_and_differ():
    from src.database.db_manager import generate_password, password_problem
    first, second = generate_password(), generate_password()
    assert password_problem(first) is None and len(first) >= 20
    assert first != second


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
    before = slots(db_path, db._dek)
    with pytest.raises(PermissionError, match="current password"):
        db.update_employee(a, "a", "A", "a", "new-password-2-long")
    with pytest.raises(PermissionError, match="incorrect"):
        db.update_employee(a, "a", "A", "a", "new-password-2-long", current_password="wrong-password-9")
    assert slots(db_path, db._dek) == before
    assert db.authenticate_employee("a", PW_A)[0] is not None


def test_a_locked_store_changes_no_profile(db):
    a = db.create_employee("A", "a", PW_A)
    db.lock()
    with pytest.raises(DatabaseLockedError):
        db.update_employee(a, "a", "Hacked", "mallory")
    db.authenticate_employee("a", PW_A)
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
    db.save_client("shared-id", "Shared-ID", a, spatial_embeddings=[original], spatial_origin="scan", temporal_embeddings=[], model_version=MODEL)
    assert db.get_client_record("shared-id")["enrolled_by"] == "alice"
    assert db.get_client_record("shared-id", b) is None          # "enrolled by Bob": no
    assert not db.has_client("shared-id", b) and db.has_client("shared-id")
    assert [r["key"] for r in db.list_client_records()] == ["shared-id"]
    assert np.array_equal(db.get_client_record("shared-id")["spatial"][0], original)


def test_rename_keeps_clients_and_log_with_the_employee(db):
    a, b = two_employees(db)
    db.save_client("c1", "C1", a, spatial_embeddings=[vec(512, 13)], spatial_origin="scan", temporal_embeddings=[], model_version=MODEL)
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


def test_a_non_admin_cannot_change_a_clients_name(db):
    a, b = two_employees(db)
    db.save_client("c1", "Signer", a, spatial_embeddings=[vec(512, 1)], spatial_origin="scan", temporal_embeddings=[], model_version=MODEL)
    # a set the client lacks is open to everyone, but the name stays the admin's
    db.save_client("c1", "SIGNER", b, spatial_embeddings=None, spatial_origin="scan", temporal_embeddings=[vec(128, 2)], model_version=MODEL)
    assert db.get_client_record("c1")["display_name"] == "Signer"
    assert audit(db)[-1] == ("enrol", "bob", ["pen set"])


def test_a_non_admin_adds_a_scan_set_to_a_pen_only_client(db):
    a, b = two_employees(db)
    db.save_client("c1", "C1", a, spatial_embeddings=None, spatial_origin="scan", temporal_embeddings=[vec(128, 1)], model_version=MODEL)
    db.save_client("c1", "C1", b, spatial_embeddings=[vec(512, 2)], spatial_origin="scan", temporal_embeddings=None, model_version=MODEL)
    assert audit(db)[-1] == ("enrol", "bob", ["scan set"])


def test_the_admins_name_change_is_logged(db):
    a, _ = two_employees(db)
    db.save_client("c1", "Signer", a, spatial_embeddings=[vec(512, 1)], spatial_origin="scan", temporal_embeddings=[], model_version=MODEL)
    db.save_client("c1", "SIGNER", a, spatial_embeddings=None, spatial_origin="scan", temporal_embeddings=None, model_version=MODEL)
    rec = db.get_client_record("c1")
    assert rec["display_name"] == "SIGNER"
    assert audit(db)[-1] == ("re-enrol", "alice", ["display name"])
    before = len(audit(db))
    db.save_client("c1", "SIGNER", a, spatial_embeddings=None, spatial_origin="scan", temporal_embeddings=None, model_version=MODEL)   # no change
    assert len(audit(db)) == before


# ------------------------------------------------- replay of a stored reference
def test_an_enrolled_or_added_reference_given_as_a_probe_is_a_replay(db):
    a, _ = two_employees(db)
    scan, pen, added = vec(512, 1), vec(128, 2), vec(512, 3)
    db.save_client("c1", "C1", a, spatial_embeddings=[scan], spatial_origin="scan", temporal_embeddings=[pen], model_version=MODEL)
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
    db.save_client("c1", "C1", a, spatial_embeddings=first, spatial_origin="scan", temporal_embeddings=[], model_version=MODEL)
    with pytest.raises(ValueError):
        db.save_client("c1", "C1", a, spatial_embeddings=[vec(512, 50), vec(10, 51)], spatial_origin="scan", temporal_embeddings=[], model_version=MODEL)
    rec = db.get_client_record("c1", a)
    assert all(np.array_equal(x, y) for x, y in zip(rec["spatial"], first)) and len(rec["spatial"]) == 5


def test_error_inside_the_transaction_rolls_everything_back(db, db_path):
    a, _ = two_employees(db)
    first = [vec(512, i) for i in range(5)]
    db.save_client("c1", "C1", a, spatial_embeddings=first, spatial_origin="scan", temporal_embeddings=[], model_version=MODEL)
    with edit_live(db) as conn:  # a failure that only happens at INSERT time
        conn.execute("""CREATE TRIGGER boom BEFORE INSERT ON enrollments WHEN NEW.model_version = 'boom'
                        BEGIN SELECT RAISE(ABORT, 'disk on fire'); END""")
    with pytest.raises(sqlite3.DatabaseError):
        db.save_client("c1", "C1", a, spatial_embeddings=[vec(512, 60)], spatial_origin="scan", temporal_embeddings=[], model_version="boom")
    rec = db.get_client_record("c1", a)
    assert len(rec["spatial"]) == 5
    assert np.array_equal(rec["spatial"][2], first[2])
    assert [e["event"] for e in db.get_log_events(event=None) if e["client_key"]] == ["enrol"]   # the log rolled back too


def test_reenrolment_replaces_old_templates_and_keeps_enrolled_at(db, db_path):
    a, _ = two_employees(db)
    db.save_client("c1", "C1", a, spatial_embeddings=[vec(512, i) for i in range(5)], spatial_origin="scan", temporal_embeddings=[], model_version=MODEL)
    with edit_live(db) as conn:  # backdate, so "kept" cannot be confused with "same second"
        conn.execute("UPDATE users SET enrolled_at = '2020-01-02T03:04:05', updated_at = '2020-01-02T03:04:05'")
    new = [vec(512, 200 + i) for i in range(5)]
    db.save_client("c1", "c1 renamed", a, spatial_embeddings=new, spatial_origin="scan", temporal_embeddings=[], model_version=MODEL)
    rec = db.get_client_record("c1")
    assert len(rec["spatial"]) == 5
    assert np.array_equal(rec["spatial"][0], new[0])
    assert rec["enrolled_at"] == "2020-01-02T03:04:05"
    assert rec["updated_at"] != "2020-01-02T03:04:05"
    assert rec["display_name"] == "c1 renamed"


def test_no_image_path_is_stored(db, db_path):
    a, _ = two_employees(db)
    db.save_client("c1", "C1", a, spatial_embeddings=[vec(512, 1)], spatial_origin="scan", temporal_embeddings=[], model_version=MODEL)
    with on_disk(db_path, db._dek) as conn:
        columns = [r[1] for t in ("users", "enrollments")
                   for r in conn.execute(f"PRAGMA table_info({t})")]
        assert not [c for c in columns if "image" in c.lower()]
        row = conn.execute("SELECT * FROM users WHERE user_id = 'c1'").fetchone()
    assert not [v for v in row if isinstance(v, (str, bytes)) and (b"/" in v if isinstance(v, bytes) else "/" in v)]
    assert "image_references" not in db.get_client_record("c1")


def test_a_passphrase_never_creates_a_second_key_on_a_password_database(tmp_path):
    path = str(tmp_path / "s.db")
    owner = DatabaseManager(path)
    owner.create_employee("Owner", "owner", PW_A)
    before = slots(path, owner._dek)
    with pytest.raises(ValueError, match="member passwords"):
        DatabaseManager(path, passphrase="ci-secret")
    assert slots(path, owner._dek) == before
