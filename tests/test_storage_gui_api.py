"""Part 2/4 of issue #22: src/gui/storage.py keeps its API and now runs on SQLite + AES."""
import base64
import os
import stat
import sys
from datetime import datetime
 
import numpy as np
import pytest
 
from src.database.db_manager import DatabaseManager, PASSPHRASE_ENV_VAR
from src.gui import storage
 
CLIENT_KEYS = {"display_name", "enrolled_by", "image_references", "dynamic_references",
               "spatial_embeddings", "spatial_origin", "enrolled_at", "updated_at"}
LOG_KEYS = {"client_key", "client_display_name", "decision", "confidence",
            "modality_state", "employee", "timestamp"}
 
 
@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(DatabaseManager, "KDF_N", 2 ** 10)
    monkeypatch.setenv(storage.HOME_ENV_VAR, str(tmp_path / "home"))
    monkeypatch.delenv(PASSPHRASE_ENV_VAR, raising=False)
    storage._reset()
    yield tmp_path / "home"
    storage._reset()
 
 
def b64(dim, seed):
    return base64.b64encode(np.random.default_rng(seed).standard_normal(dim).astype(np.float32).tobytes()).decode()
 
 
def spatial(n=5, base=0):
    return [b64(512, base + i) for i in range(n)]
 
 
def db_bytes(home):
    with open(home / "signatures.db", "rb") as f:
        return f.read()
 
 
@pytest.fixture
def people():
    assert storage.register_employee("Alice A", "alice", "pw-a")[0]
    assert storage.register_employee("Bob B", "bob", "pw-b")[0]
 
 
# ---------------------------------------------------------------- employees
def test_register_and_login_messages(people):
    assert storage.register_employee("X", "ALICE", "z") == (False, "This User ID already exists")
    ok, emp, msg = storage.authenticate_employee("Alice", "pw-a")
    assert (ok, msg) == (True, "") and emp == {"full_name": "Alice A", "username": "alice"}
    assert storage.authenticate_employee("alice", "nope") == (False, None, "Incorrect password")
    assert storage.authenticate_employee("ghost", "x") == (False, None, "User ID not found")
 
 
def test_employee_dicts_never_carry_secrets(people):
    assert set(storage.find_employee("alice")) == {"full_name", "username"}
    assert storage.find_employee("ghost") is None
    emp = storage.authenticate_employee("alice", "pw-a")[1]
    assert "password" not in emp and "salt" not in emp
 
 
def test_password_is_not_in_the_database_file(people, isolated_storage):
    storage.register_employee("C", "carol", "Unmistakable-Pass-123")
    assert b"Unmistakable-Pass-123" not in db_bytes(isolated_storage)
 
 
def test_update_profile_results(people):
    ok, msg, emp = storage.update_employee_profile("alice", "Alice Z", "alice", "newpw")
    assert (ok, msg, emp) == (True, "Profile updated successfully", {"full_name": "Alice Z", "username": "alice"})
    assert storage.authenticate_employee("alice", "newpw")[0]
    assert not storage.authenticate_employee("alice", "pw-a")[0]
    assert storage.update_employee_profile("alice", "A", "bob") == (False, "This User ID is already taken", None)
    assert storage.update_employee_profile("ghost", "G", "g") == (False, "Employee not found", None)
    # no new password given: the old one keeps working
    storage.update_employee_profile("alice", "Alice Z", "alice2")
    assert storage.authenticate_employee("alice2", "newpw")[0]
 
 
def test_renaming_a_username_keeps_clients_and_log_visible(people):
    assert storage.save_client_enrollment("Client One", ["/x/1.png"], [], "alice", spatial(), "scan")[0]
    storage.append_verification_log("Client One", "Genuine", 88.0, "scan", "alice")
    assert storage.update_employee_profile("alice", "Alice A", "alice.renamed")[0]
 
    clients = storage.list_clients("alice.renamed")
    assert [c["key"] for c in clients] == ["client one"]
    assert clients[0]["enrolled_by"] == "alice.renamed"
    assert storage.client_exists("client one", "alice.renamed")
    assert storage.get_client("client one", "alice.renamed") is not None
    assert storage.get_client("client one", "alice") is None
    assert storage.get_verification_stats("alice.renamed")["total"] == 1
    assert storage.get_verification_log("alice.renamed")[0]["employee"] == "alice.renamed"
    assert storage.count_clients_enrolled_this_month("alice.renamed") == 1
    # and the renamed employee can still re-enroll their own client
    assert storage.save_client_enrollment("client one", [], [], "alice.renamed", spatial(base=9), "scan")[0]
 
 
# ------------------------------------------------------------------ clients
def test_save_and_get_client_shape_and_round_trip(people):
    embs = spatial()
    assert storage.save_client_enrollment("  Client A ", ["/p/a.png"], [], "alice", embs, "scan") == (True, "Saved")
    rec = storage.get_client("CLIENT a", "alice")
    assert set(rec) == CLIENT_KEYS
    assert rec["display_name"] == "Client A"
    assert rec["enrolled_by"] == "alice"
    assert rec["spatial_embeddings"] == embs
    assert rec["image_references"] == ["/p/a.png"]
    assert rec["dynamic_references"] == []
    assert rec["spatial_origin"] == "scan"
    datetime.fromisoformat(rec["enrolled_at"])
    datetime.fromisoformat(rec["updated_at"])
    assert storage.client_exists("client a") and storage.client_exists("CLIENT A", "alice")
    assert not storage.client_exists("client a", "bob") and not storage.client_exists("nope")
 
 
def test_list_clients_is_scoped_and_has_a_key_field(people):
    storage.save_client_enrollment("Alpha", [], [], "alice", spatial(1), "scan")
    storage.save_client_enrollment("Beta", [], [], "bob", spatial(1, 5), "scan")
    assert [c["key"] for c in storage.list_clients("alice")] == ["alpha"]
    assert [c["key"] for c in storage.list_clients("bob")] == ["beta"]
    assert sorted(c["key"] for c in storage.list_clients()) == ["alpha", "beta"]
    assert set(storage.list_clients("alice")[0]) == CLIENT_KEYS | {"key"}
    assert storage.list_clients("ghost") == []
 
 
def test_other_employee_cannot_overwrite_or_read_a_client(people):
    mine = spatial()
    storage.save_client_enrollment("Shared", ["/a.png"], [], "alice", mine, "scan")
    ok, msg = storage.save_client_enrollment("SHARED", [], [], "bob", spatial(base=77), "render")
    assert ok is False and "already enrolled by another employee" in msg
    assert storage.get_client("shared", "bob") is None
    assert storage.get_client("shared", None) is not None          # existence is visible, data is not Bob's
    kept = storage.get_client("shared", "alice")
    assert kept["spatial_embeddings"] == mine and kept["spatial_origin"] == "scan"
 
 
def test_reenrolment_replaces_templates_and_keeps_enrolled_at(people):
    storage.save_client_enrollment("C", ["/1.png"], [], "alice", spatial(), "scan")
    with storage._get_db().get_connection() as conn:  # backdate: "kept" must not mean "same second"
        conn.execute("UPDATE users SET enrolled_at = '2020-01-02T03:04:05', updated_at = '2020-01-02T03:04:05'")
    first = storage.get_client("c", "alice")
    new = spatial(base=300)
    assert storage.save_client_enrollment("C", ["/2.png"], [], "alice", new, "scan")[0]
    rec = storage.get_client("c", "alice")
    assert rec["spatial_embeddings"] == new and len(rec["spatial_embeddings"]) == 5
    assert rec["image_references"] == ["/2.png"]
    assert rec["enrolled_at"] == first["enrolled_at"] == "2020-01-02T03:04:05"
    assert rec["updated_at"] != "2020-01-02T03:04:05"
 
 
def test_bad_input_is_refused_without_touching_old_data(people):
    old = spatial()
    storage.save_client_enrollment("C", [], [], "alice", old, "scan")
    assert storage.save_client_enrollment("C", [], [], "alice", [b64(10, 1)], "scan")[0] is False
    assert storage.save_client_enrollment("C", [], [], "alice", ["!!not base64!!"], "scan")[0] is False
    assert storage.save_client_enrollment("C", [], [], "alice", spatial(1), "sideways")[0] is False
    assert storage.get_client("c", "alice")["spatial_embeddings"] == old
    assert storage.save_client_enrollment("", [], [], "alice", spatial(1), "scan")[0] is False
    assert storage.save_client_enrollment("a|b", [], [], "alice", spatial(1), "scan")[0] is False
    assert storage.save_client_enrollment("C", [], [], "ghost", spatial(1), "scan") == (False, "Employee not found")
 
 
def test_failed_save_keeps_the_previous_enrollment(people):
    old = spatial()
    storage.save_client_enrollment("C", ["/old.png"], [], "alice", old, "scan")
    ok, _ = storage.save_client_enrollment("C", ["/new.png"], [], "alice", spatial(2) + [b64(7, 1)], "scan")
    assert ok is False
    rec = storage.get_client("c", "alice")
    assert rec["spatial_embeddings"] == old and rec["image_references"] == ["/old.png"]
 
 
def test_tablet_enrolment_stores_temporal_and_render(people):
    pen = [b64(128, 40 + i) for i in range(5)]
    assert storage.save_client_enrollment("Tab", [], pen, "alice", None, "render")[0]
    rec = storage.get_client("tab", "alice")
    assert rec["dynamic_references"] == pen and rec["spatial_embeddings"] == []
    assert rec["spatial_origin"] == "render"
    both = storage.save_client_enrollment("Both", ["/s.png"], pen, "alice", spatial(), "scan")
    assert both[0]
    rec = storage.get_client("both", "alice")
    assert rec["dynamic_references"] == pen and len(rec["spatial_embeddings"]) == 5
    assert rec["spatial_origin"] == "scan"
 
 
def test_spatial_origin_saved_is_the_one_returned(people):
    for origin in ("scan", "render"):
        storage.save_client_enrollment(f"c-{origin}", [], [], "alice", spatial(1), origin)
        assert storage.get_client(f"c-{origin}", "alice")["spatial_origin"] == origin
 
 
def test_delete_client_removes_data_but_keeps_the_log(people):
    storage.save_client_enrollment("Gone", [], [], "alice", spatial(), "scan")
    storage.append_verification_log("Gone", "Genuine", 90.0, "scan", "alice")
    assert storage.delete_client("Gone") is None
    assert storage.get_client("gone") is None and not storage.client_exists("gone")
    assert storage.list_clients("alice") == []
    assert storage.get_last_verification_for_client("gone", "alice") is not None
    storage.delete_client("never existed")  # no error
 
 
def test_count_clients_enrolled_this_month(people):
    storage.save_client_enrollment("A1", [], [], "alice", spatial(1), "scan")
    storage.save_client_enrollment("B1", [], [], "bob", spatial(1, 3), "scan")
    assert storage.count_clients_enrolled_this_month("alice") == 1
    assert storage.count_clients_enrolled_this_month() == 2
    assert storage.count_clients_enrolled_this_month("ghost") == 0
    db = storage._get_db()
    with db.get_connection() as conn:
        conn.execute("UPDATE users SET enrolled_at = '2001-02-03T04:05:06' WHERE user_id = 'a1'")
    assert storage.count_clients_enrolled_this_month("alice") == 0
 
 
# ---------------------------------------------------------------------- log
def test_log_events_and_stats(people):
    storage.append_verification_log("Client X", "Genuine", 87.3, "multimodal", "alice")
    storage.append_verification_log("client x", "Forged", 61.0, "scan", "alice")
    storage.append_verification_log("Client Y", "Genuine", 90.5, "scan", "bob")
    events = storage.get_verification_log("alice")
    assert [e["decision"] for e in events] == ["Genuine", "Forged"]
    assert set(events[0]) == LOG_KEYS
    assert events[0]["client_key"] == "client x" and events[0]["client_display_name"] == "Client X"
    assert events[0]["confidence"] == 87.3 and events[0]["employee"] == "alice"
    datetime.fromisoformat(events[0]["timestamp"])
    assert len(storage.get_verification_log()) == 3
    assert storage.get_last_verification_for_client("CLIENT X", "alice")["decision"] == "Forged"
    assert storage.get_last_verification_for_client("client x", "bob") is None
    assert storage.get_last_verification_for_client("client y")["employee"] == "bob"
    assert storage.get_verification_stats("alice") == {"total": 2, "genuine": 1, "genuine_rate": 50.0}
    assert storage.get_verification_stats() == {"total": 3, "genuine": 2, "genuine_rate": 66.7}
    assert storage.get_verification_stats("ghost") == {"total": 0, "genuine": 0, "genuine_rate": 0.0}
 
 
def test_empty_log_stats():
    assert storage.get_verification_stats() == {"total": 0, "genuine": 0, "genuine_rate": 0.0}
    assert storage.get_verification_log() == []
 
 
def test_sign_again_is_not_logged(people):
    storage.append_verification_log("c", "Sign again", 70.0, "multimodal", "alice")
    assert storage.get_verification_log() == []
 
 
# ------------------------------------------------------- at-rest protection
def test_embeddings_and_image_paths_are_not_readable_in_the_file(people, isolated_storage):
    embs = spatial(2)
    path = "/Users/someone/Desktop/Jane-Doe-signature.png"
    storage.save_client_enrollment("Jane Doe", [path], [b64(128, 5)], "alice", embs, "scan")
    data = db_bytes(isolated_storage)
    for e in embs + [b64(128, 5)]:
        assert base64.b64decode(e) not in data and e.encode() not in data
    assert path.encode() not in data and b"Jane-Doe-signature" not in data
 
 
def test_no_json_storage_code_is_left():
    for name in ("_load_json", "_save_json", "load_employees", "load_enrollments",
                 "EMPLOYEES_FILE", "ENROLLMENTS_FILE", "LOG_FILE"):
        assert not hasattr(storage, name)
    with open(storage.__file__, encoding="utf-8") as f:
        source = f.read()
    assert "import json" not in source and ".json" not in source
 
 
# ---------------------------------------------------------------- key / path
def test_data_location_is_fixed_not_the_working_directory(people, tmp_path, monkeypatch, isolated_storage):
    storage.save_client_enrollment("Persist", [], [], "alice", spatial(1), "scan")
    elsewhere = tmp_path / "some" / "other" / "dir"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    storage._reset()
    assert storage.get_client("persist", "alice") is not None
    assert sorted(os.listdir(elsewhere)) == []                     # nothing created in the cwd
    assert sorted(os.listdir(isolated_storage)) == ["db.key", "signatures.db"]
 
 
def test_missing_key_file_fails_loudly_and_creates_no_new_key(people, isolated_storage):
    storage.save_client_enrollment("K", [], [], "alice", spatial(1), "scan")
    os.remove(isolated_storage / "db.key")
    storage._reset()
    with pytest.raises(RuntimeError, match="key file"):
        storage.get_client("k")
    assert not (isolated_storage / "db.key").exists()
 
 
def test_wrong_key_file_fails_loudly(people, isolated_storage):
    (isolated_storage / "db.key").write_text("a-different-key")
    storage._reset()
    with pytest.raises(RuntimeError, match="does not match"):
        storage.get_client("k")
 
 
def test_passphrase_env_var_overrides_the_key_file(isolated_storage, monkeypatch):
    monkeypatch.setenv(PASSPHRASE_ENV_VAR, "from-environment")
    storage._reset()
    assert storage.register_employee("E", "env", "pw")[0]
    assert not (isolated_storage / "db.key").exists()
    storage._reset()
    assert storage.find_employee("env") is not None
 
 
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_key_file_is_private(people, isolated_storage):
    assert stat.S_IMODE(os.stat(isolated_storage / "db.key").st_mode) == 0o600
    assert stat.S_IMODE(os.stat(isolated_storage).st_mode) == 0o700
 