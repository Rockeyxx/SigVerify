"""Part 1 of issue #22: encryption, key handling, passwords, ownership (temporary DB only)."""
import hashlib
import hmac
import sqlite3
 
import numpy as np
import pytest
from cryptography.exceptions import InvalidTag
 
from src.database.db_manager import DatabaseManager
 
MODEL = "test-model-v1"
 
 
@pytest.fixture(autouse=True)
def fast_kdf(monkeypatch):
    monkeypatch.setattr(DatabaseManager, "KDF_N", 2 ** 10)
 
 
@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")
 
 
@pytest.fixture
def db(db_path):
    return DatabaseManager(db_path, passphrase="pass-one")
 
 
def vec(dim, seed):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
 
 
def file_bytes(path):
    with open(path, "rb") as f:
        return f.read()
 
 
def two_employees(db):
    return db.create_employee("Alice", "alice", "pw-alice"), db.create_employee("Bob", "bob", "pw-bob")
 
 
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
    before = sqlite3.connect(db_path).execute("SELECT salt, wrapped_dek, nonce FROM meta").fetchall()
    with pytest.raises(InvalidTag):
        DatabaseManager(db_path, passphrase="not-the-passphrase")
    after = sqlite3.connect(db_path).execute("SELECT salt, wrapped_dek, nonce FROM meta").fetchall()
    assert before == after
 
 
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
    assert db._dek not in file_bytes(db_path)
 
 
# ----------------------------------------------------- origin + model version
@pytest.mark.parametrize("origin", ["scan", "render"])
def test_spatial_origin_saved_is_the_one_returned(db, db_path, origin):
    emp, _ = two_employees(db)
    db.save_client("c1", "C1", emp, [], [vec(512, 8)], origin, [], MODEL)
    assert db.get_client_record("c1")["spatial_origin"] == origin
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT spatial_origin FROM enrollments").fetchall() == [(origin,)]
    assert db.list_spatial_origins("c1") == [origin]
 
 
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
    db.create_employee("Alice", "alice", "correct horse")
    emp, reason = db.authenticate_employee("alice", "wrong horse")
    assert emp is None and reason == "bad_password"
    emp, reason = db.authenticate_employee("ALICE", "correct horse")
    assert emp["username"] == "alice" and reason == ""
    assert db.authenticate_employee("nobody", "x") == (None, "not_found")
 
 
def test_password_text_is_nowhere_in_the_database_file(db, db_path):
    secret = "S3cret-Passw0rd-marker"
    db.create_employee("Alice", "alice", secret)
    data = file_bytes(db_path)
    assert secret.encode() not in data
    assert hashlib.sha256(secret.encode()).digest() not in data
 
 
def test_password_hash_is_salted_scrypt(db, db_path):
    db.create_employee("A", "a", "same-password")
    db.create_employee("B", "b", "same-password")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT salt, password_hash FROM employees ORDER BY id").fetchall()
    (salt_a, hash_a), (salt_b, hash_b) = rows
    assert len(salt_a) == 16 and salt_a != salt_b
    assert hash_a != hash_b
    assert hash_a == hashlib.scrypt(b"same-password", salt=salt_a, n=2 ** 14, r=8, p=1, dklen=32)
 
 
def test_login_compares_with_hmac_compare_digest(db, monkeypatch):
    db.create_employee("A", "a", "pw")
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(hmac, "compare_digest", lambda x, y: calls.append(1) or real(x, y))
    db.authenticate_employee("a", "pw")
    assert calls
 
 
def test_new_password_is_rehashed_with_a_new_salt(db, db_path):
    db.create_employee("A", "a", "old")
    before = sqlite3.connect(db_path).execute("SELECT salt, password_hash FROM employees").fetchone()
    db.update_employee("a", "A", "a", "new")
    after = sqlite3.connect(db_path).execute("SELECT salt, password_hash FROM employees").fetchone()
    assert before[0] != after[0] and before[1] != after[1]
    assert db.authenticate_employee("a", "old")[0] is None
    assert db.authenticate_employee("a", "new")[0] is not None
 
 
# ----------------------------------------------------------------- ownership
def test_employee_b_cannot_overwrite_or_read_a_clients_client(db):
    a, b = two_employees(db)
    original = vec(512, 11)
    db.save_client("shared-id", "Shared-ID", a, ["/p/1.png"], [original], "scan", [], MODEL)
 
    with pytest.raises(PermissionError):
        db.save_client("shared-id", "Shared-ID", b, [], [vec(512, 12)], "render", [], MODEL)
 
    assert db.get_client_record("shared-id", b) is None
    assert not db.has_client("shared-id", b)
    assert db.list_client_records(b) == []
    kept = db.get_client_record("shared-id", a)
    assert kept["enrolled_by"] == "alice" and kept["spatial_origin"] == "scan"
    assert np.array_equal(kept["spatial"][0], original)
 
 
def test_rename_keeps_clients_and_log_with_the_employee(db):
    a, b = two_employees(db)
    db.save_client("c1", "C1", a, [], [vec(512, 13)], "scan", [], MODEL)
    db.log_verification("c1", "scan", 80.0, "Genuine", employee_id=a, display_name="C1")
    db.update_employee("alice", "Alice N", "alice.new", None)
    new_id = db.find_employee("alice.new")["id"]
    assert new_id == a
    assert [r["key"] for r in db.list_client_records(new_id)] == ["c1"]
    assert db.get_client_record("c1", new_id)["enrolled_by"] == "alice.new"
    events = db.get_log_events(employee_id=new_id)
    assert len(events) == 1 and events[0]["employee"] == "alice.new"
    assert db.find_employee("alice") is None
    assert db.list_client_records(b) == []
 
 
def test_rename_to_a_taken_username_is_refused(db):
    two_employees(db)
    with pytest.raises(ValueError):
        db.update_employee("alice", "Alice", "BOB", None)
    assert db.find_employee("alice") is not None
 
 
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
 