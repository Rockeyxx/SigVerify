"""src/gui/storage.py: the GUI's storage API on SQLite + AES, unlocked by employee logins."""
import base64
import os
import shutil
import sqlite3
import sys
from contextlib import closing
from datetime import datetime

import numpy as np
import pytest

from config.settings import MAX_REFERENCES
from src.database.db_manager import PASSPHRASE_ENV_VAR, DatabaseManager, probe_fingerprint
from src.gui import storage
from tests.test_database import OLD_B, _make_v4

CLIENT_KEYS = {"display_name", "enrolled_by", "dynamic_references",
               "spatial_embeddings", "spatial_origin", "enrolled_at", "updated_at",
               "spatial_sets", "temporal_set"}
SET_KEYS = {"embeddings", "model_version", "enrolled_at", "stale", "capture_device"}
LOG_KEYS = {"client_key", "client_display_name", "decision", "confidence",
            "modality_state", "employee", "timestamp",
            "routing", "degraded", "reasons", "model_version", "threshold", "event"}
PW_A = "alice-password-1"
PW_B = "bob-password-123"
XPPEN = {"device_model": "XPPen Deco 640", "os": "Windows", "qt_platform": "windows"}
COMPANY = "Test Co"


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


def pen(n=5, base=300):
    return [b64(128, base + i) for i in range(n)]


def company_dir(home, slug="test-co"):
    return home / "companies" / slug


def db_bytes(home):
    with open(company_dir(home) / "signatures.db", "rb") as f:
        return f.read()


@pytest.fixture
def people():
    """Alice registers the company Test Co (its admin, logged in); she adds Bob."""
    assert storage.create_company(COMPANY, "Alice A", "alice", PW_A)[0]
    assert storage.register_employee("Bob B", "bob", PW_B)[0]


def login(username, password):
    storage.open_company(COMPANY)
    assert storage.authenticate_employee(username, password)[0]


# ---------------------------------------------------------------- employees
def test_registering_a_company_makes_its_admin_who_is_told_passwords_cannot_be_recovered():
    assert storage.is_first_run() and storage.current_employee() is None and storage.current_company() is None
    ok, msg = storage.create_company(COMPANY, "Alice A", "alice", PW_A)
    assert ok and msg.startswith("Admin account created") and storage.ADMIN_WARNING in msg
    assert storage.current_employee() == {"full_name": "Alice A", "username": "alice", "is_admin": True}
    assert "cannot be recovered" in storage.ADMIN_WARNING and not storage.is_first_run()
    assert storage.current_company() == COMPANY
    assert storage.find_employee("alice")["is_admin"] is True
    assert storage.register_employee("Bob B", "bob", PW_B) == (True, "Account created successfully")
    assert storage.find_employee("bob")["is_admin"] is False


def test_only_a_logged_in_admin_adds_employees(people):
    storage.logout()
    ok, msg = storage.register_employee("Carol", "carol", "carol-password-1")
    assert not ok and "Only the admin" in msg
    login("bob", PW_B)
    assert not storage.register_employee("Carol", "carol", "carol-password-1")[0]
    login("alice", PW_A)
    assert storage.register_employee("Carol", "carol", "carol-password-1")[0]


def test_register_and_login_messages(people):
    assert storage.register_employee("X", "ALICE", "some-password-1") == (False, "This User ID already exists")
    assert storage.register_employee("X", "x", "short") == (False, "The password must have at least 12 characters")
    ok, emp, msg = storage.authenticate_employee("Alice", PW_A)
    assert (ok, msg) == (True, "") and emp == {"full_name": "Alice A", "username": "alice", "is_admin": True}
    assert storage.authenticate_employee("alice", "nope") == (False, None, "Incorrect password")
    assert storage.authenticate_employee("ghost", "x") == (False, None, "User ID not found")


def test_employee_dicts_never_carry_secrets(people):
    assert set(storage.find_employee("alice")) == {"full_name", "username", "is_admin"}
    assert storage.find_employee("ghost") is None
    emp = storage.authenticate_employee("alice", PW_A)[1]
    assert not {"password", "salt", "id"} & set(emp)
    assert [e["username"] for e in storage.list_employees()] == ["alice", "bob"]


def test_password_is_not_in_the_database_file(people, isolated_storage):
    storage.register_employee("C", "carol", "Unmistakable-Pass-123")
    assert b"Unmistakable-Pass-123" not in db_bytes(isolated_storage)


def test_update_profile_results(people):
    ok, msg, emp = storage.update_employee_profile("alice", "Alice Z", "alice", "new-password-1",
                                                   current_password=PW_A)
    assert (ok, msg) == (True, "Profile updated successfully")
    assert emp == {"full_name": "Alice Z", "username": "alice", "is_admin": True}
    assert storage.authenticate_employee("alice", "new-password-1")[0]
    assert not storage.authenticate_employee("alice", PW_A)[0]
    login("alice", "new-password-1")
    assert storage.update_employee_profile("alice", "A", "bob") == (False, "This User ID is already taken", None)
    assert storage.update_employee_profile("ghost", "G", "g") == (False, "You can only change your own profile", None)
    assert storage.update_employee_profile("alice", "A", "alice", "short",
                                           current_password="new-password-1")[0] is False
    # no new password given: the old one keeps working
    storage.update_employee_profile("alice", "Alice Z", "alice2")
    assert storage.find_employee("alice2")["full_name"] == "Alice Z"
    assert storage.authenticate_employee("alice2", "new-password-1")[0]


def test_nobody_changes_another_employees_profile_or_password(people):
    login("bob", PW_B)
    for args in (("alice", "Alice", "alice", "bob-owns-alice-1"), ("alice", "Hacked", "mallory")):
        assert storage.update_employee_profile(*args, current_password=PW_B) == (
            False, "You can only change your own profile", None)
    login("alice", PW_A)                                       # not even the admin
    assert storage.update_employee_profile("bob", "B", "bob", "alice-owns-bob-1")[0] is False
    storage.logout()                                           # nor anyone logged out
    assert storage.update_employee_profile("alice", "Hacked", "mallory")[0] is False
    assert storage.find_employee("mallory") is None
    assert storage.authenticate_employee("alice", PW_A)[0] and storage.authenticate_employee("bob", PW_B)[0]


def test_a_new_password_needs_the_current_one(people):
    login("bob", PW_B)
    ok, msg, _ = storage.update_employee_profile("bob", "Bob B", "bob", "bob-new-password")
    assert not ok and "current password" in msg
    ok, msg, _ = storage.update_employee_profile("bob", "Bob B", "bob", "bob-new-password",
                                                 current_password="wrong-password-1")
    assert not ok and "incorrect" in msg
    assert storage.authenticate_employee("bob", PW_B)[0]


def test_writes_run_as_the_logged_in_employee_never_a_name_passed_in(people, tmp_path):
    """Bob, logged in, passes the admin's name to every write: each is refused."""
    storage.save_client_enrollment("C", None, "alice", spatial_embeddings=spatial(), spatial_origin="scan")
    login("bob", PW_B)
    assert storage.reset_employee_password("alice", "alice", "taken-over-pass")[0] is False
    assert storage.reset_employee_password("alice", "bob", "bob-new-password")[0] is False
    assert storage.remove_employee("alice", "bob")[0] is False
    assert storage.register_employee("Mallory", "mallory", "mallory-pass-1")[0] is False
    assert storage.save_client_enrollment("C", None, "alice", spatial_embeddings=spatial(base=9), spatial_origin="scan")[0] is False
    assert storage.delete_client("C", "alice")[0] is False
    assert storage.add_confirmed_reference("c", b64(512, 5), "alice") is False
    assert storage.backup_database(str(tmp_path), "alice")[0] is False
    with pytest.raises(storage.NotLoggedInAsError):
        storage.append_verification_log("C", "Genuine", 90.0, "scan", "alice")
    assert storage.get_client("c")["spatial_embeddings"] == spatial()
    assert storage.authenticate_employee("alice", PW_A)[0]
    assert [(e["event"], e["employee"]) for e in storage.get_audit_log() if e["client_key"]] == [("enrol", "alice")]


def test_writes_need_a_login(people):
    storage.save_client_enrollment("C", None, "alice", spatial_embeddings=spatial(1), spatial_origin="scan")
    storage.logout()
    with pytest.raises(storage.StorageLockedError):
        storage.append_verification_log("C", "Genuine", 90.0, "scan", "alice")
    assert storage.delete_client("C", "alice") == (False, "Log in first.")
    assert storage.reset_employee_password("alice", "bob", "bob-new-password")[0] is False


def test_admin_resets_a_forgotten_password_and_removes_an_employee(people):
    assert storage.reset_employee_password("bob", "alice", "taken-over-pass")[0] is False
    assert storage.reset_employee_password("alice", "bob", "bob-new-password") == (True, "Done")
    assert storage.authenticate_employee("bob", "bob-new-password")[0]
    login("alice", PW_A)
    assert storage.remove_employee("alice", "bob") == (True, "Done")
    assert storage.authenticate_employee("bob", "bob-new-password") == (False, None, "User ID not found")


def test_renaming_a_username_keeps_clients_and_log_visible(people):
    assert storage.save_client_enrollment("Client One", [], "alice", spatial_embeddings=spatial(), spatial_origin="scan")[0]
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
    assert storage.save_client_enrollment("client one", [], "alice.renamed", spatial_embeddings=spatial(base=9), spatial_origin="scan")[0]


# ------------------------------------------------------------------ clients
def test_save_and_get_client_shape_and_round_trip(people):
    embs = spatial()
    assert storage.save_client_enrollment("  Client A ", [], "alice", spatial_embeddings=embs, spatial_origin="scan") == (True, "Saved")
    rec = storage.get_client("CLIENT a", "alice")
    assert set(rec) == CLIENT_KEYS and set(rec["spatial_sets"]["scan"]) == SET_KEYS
    assert rec["display_name"] == "Client A"
    assert rec["enrolled_by"] == "alice"
    assert rec["spatial_embeddings"] == embs
    assert rec["dynamic_references"] == [] and rec["temporal_set"] is None
    assert rec["spatial_origin"] == "scan"
    datetime.fromisoformat(rec["enrolled_at"])
    datetime.fromisoformat(rec["updated_at"])
    assert storage.client_exists("client a") and storage.client_exists("CLIENT A", "alice")
    assert not storage.client_exists("client a", "bob") and not storage.client_exists("nope")


def test_the_client_list_is_the_companys_and_can_be_narrowed(people):
    storage.save_client_enrollment("Alpha", [], "alice", spatial_embeddings=spatial(1), spatial_origin="scan")
    login("bob", PW_B)
    storage.save_client_enrollment("Beta", [], "bob", spatial_embeddings=spatial(1, 5), spatial_origin="scan")
    assert sorted(c["key"] for c in storage.list_clients()) == ["alpha", "beta"]
    assert [c["key"] for c in storage.list_clients("alice")] == ["alpha"]
    assert [c["key"] for c in storage.list_clients("bob")] == ["beta"]
    assert set(storage.list_clients("alice")[0]) == CLIENT_KEYS | {"key"}
    assert storage.list_clients("ghost") == []
    assert storage.get_client("alpha")["enrolled_by"] == "alice"     # Bob can find and verify it


def test_a_non_admin_cannot_replace_a_clients_references(people):
    mine = spatial()
    storage.save_client_enrollment("Shared", [], "alice", spatial_embeddings=mine, spatial_origin="scan")
    login("bob", PW_B)
    ok, msg = storage.save_client_enrollment("SHARED", [], "bob", spatial_embeddings=spatial(base=77), spatial_origin="scan")
    assert ok is False and "Only the admin" in msg
    assert storage.get_client("shared")["spatial_embeddings"] == mine
    # adding a set the client does not have yet is an enrolment, open to everyone
    assert storage.save_client_enrollment("shared", pen(), "bob", spatial_embeddings=None, spatial_origin="scan")[0]
    events = storage.get_audit_log()
    assert [(e["event"], e["employee"]) for e in events if e["client_key"]] == [
        ("enrol", "alice"), ("re-enrol refused", "bob"), ("enrol", "bob")]


def test_reenrolment_replaces_templates_and_keeps_enrolled_at(people):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(), spatial_origin="scan")
    with storage._get_db().get_connection() as conn:  # backdate: "kept" must not mean "same second"
        conn.execute("UPDATE users SET enrolled_at = '2020-01-02T03:04:05', updated_at = '2020-01-02T03:04:05'")
    first = storage.get_client("c", "alice")
    new = spatial(base=300)
    assert storage.save_client_enrollment("C", [], "alice", spatial_embeddings=new, spatial_origin="scan")[0]
    rec = storage.get_client("c", "alice")
    assert rec["spatial_embeddings"] == new and len(rec["spatial_embeddings"]) == 5
    assert rec["enrolled_at"] == first["enrolled_at"] == "2020-01-02T03:04:05"
    assert rec["updated_at"] != "2020-01-02T03:04:05"


def test_bad_input_is_refused_without_touching_old_data(people):
    old = spatial()
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=old, spatial_origin="scan")
    assert storage.save_client_enrollment("C", [], "alice", spatial_embeddings=[b64(10, 1)], spatial_origin="scan")[0] is False
    assert storage.save_client_enrollment("C", [], "alice", spatial_embeddings=["!!not base64!!"], spatial_origin="scan")[0] is False
    assert storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(1), spatial_origin="sideways")[0] is False
    assert storage.get_client("c", "alice")["spatial_embeddings"] == old
    assert storage.save_client_enrollment("", [], "alice", spatial_embeddings=spatial(1), spatial_origin="scan")[0] is False
    assert storage.save_client_enrollment("a|b", [], "alice", spatial_embeddings=spatial(1), spatial_origin="scan")[0] is False
    assert storage.save_client_enrollment("C", [], "ghost", spatial_embeddings=spatial(1), spatial_origin="scan") == (False, "Employee not found")
    assert storage.save_client_enrollment("New", None, "alice", spatial_embeddings=None)[0] is False   # nothing to save


def test_failed_save_keeps_the_previous_enrollment(people):
    old = spatial()
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=old, spatial_origin="scan")
    ok, _ = storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(2) + [b64(7, 1)], spatial_origin="scan")
    assert ok is False
    rec = storage.get_client("c", "alice")
    assert rec["spatial_embeddings"] == old


def test_tablet_enrolment_stores_temporal_and_the_tablet(people):
    assert storage.save_client_enrollment("Tab", pen(), "alice", spatial_embeddings=None, spatial_origin="render", capture_device=XPPEN)[0]
    rec = storage.get_client("tab", "alice")
    assert rec["dynamic_references"] == pen() and rec["spatial_embeddings"] == []
    assert rec["spatial_origin"] is None and rec["spatial_sets"] == {}   # pen only: no spatial set
    assert rec["temporal_set"]["capture_device"] == XPPEN
    assert storage.save_client_enrollment("Both", pen(), "alice", spatial_embeddings=spatial(), spatial_origin="scan")[0]
    rec = storage.get_client("both", "alice")
    assert rec["dynamic_references"] == pen() and len(rec["spatial_embeddings"]) == 5
    assert rec["spatial_origin"] == "scan"


def test_spatial_origin_saved_is_the_one_returned(people):
    for origin in ("scan", "render"):
        storage.save_client_enrollment(f"c-{origin}", [], "alice", spatial_embeddings=spatial(1), spatial_origin=origin)
        assert storage.get_client(f"c-{origin}", "alice")["spatial_origin"] == origin


def test_only_the_admin_deletes_and_the_log_is_kept(people):
    login("bob", PW_B)
    storage.save_client_enrollment("Gone", [], "bob", spatial_embeddings=spatial(), spatial_origin="scan")
    storage.append_verification_log("Gone", "Genuine", 90.0, "scan", "bob")
    assert storage.delete_client("Gone", "bob") == (False, "Only the admin can delete a client")
    assert storage.delete_client("Gone", "ghost")[0] is False
    assert storage.client_exists("gone")
    login("alice", PW_A)
    assert storage.delete_client("Gone", "alice") == (True, "Deleted client 'Gone'")
    assert storage.get_client("gone") is None and not storage.client_exists("gone")
    assert storage.get_last_verification_for_client("gone", "bob") is not None
    assert storage.delete_client("never existed", "alice") == (False, "No client 'never existed'")
    assert [e["event"] for e in storage.get_audit_log() if e["client_key"]] == [
        "enrol", "verify", "delete refused", "delete"]


def test_count_clients_enrolled_this_month(people):
    storage.save_client_enrollment("A1", [], "alice", spatial_embeddings=spatial(1), spatial_origin="scan")
    login("bob", PW_B)
    storage.save_client_enrollment("B1", [], "bob", spatial_embeddings=spatial(1, 3), spatial_origin="scan")
    assert storage.count_clients_enrolled_this_month("alice") == 1
    assert storage.count_clients_enrolled_this_month() == 2
    assert storage.count_clients_enrolled_this_month("ghost") == 0
    with storage._get_db().get_connection() as conn:
        conn.execute("UPDATE users SET enrolled_at = '2001-02-03T04:05:06' WHERE user_id = 'a1'")
    assert storage.count_clients_enrolled_this_month("alice") == 0


# ---------------------------------------------------------------------- log
def test_log_events_and_stats(people):
    storage.append_verification_log("Client X", "Genuine", 87.3, "multimodal", "alice")
    storage.append_verification_log("client x", "Forged", 61.0, "scan", "alice")
    login("bob", PW_B)
    storage.append_verification_log("Client Y", "Genuine", 90.5, "scan", "bob")
    events = storage.get_verification_log("alice")
    assert [e["decision"] for e in events] == ["Genuine", "Forged"]
    assert set(events[0]) == LOG_KEYS and events[0]["event"] == "verify"
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


def test_empty_log_stats(people):
    assert storage.get_verification_stats() == {"total": 0, "genuine": 0, "genuine_rate": 0.0}
    assert storage.get_verification_log() == []


def test_sign_again_is_not_logged(people):
    storage.append_verification_log("c", "Sign again", 70.0, "multimodal", "alice")
    assert storage.get_verification_log() == []


def test_non_final_decisions_are_not_logged_or_counted(people):
    storage.append_verification_log("c", "Error", 0.0, "multimodal", "alice")
    storage.append_verification_log("c", "Forged", 10.0, "multimodal", "alice")
    assert [e["decision"] for e in storage.get_verification_log("alice")] == ["Forged"]
    assert storage.get_verification_stats("alice")["total"] == 1


def test_enrolment_events_do_not_count_as_verifications(people):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(1), spatial_origin="scan")
    assert storage.get_verification_stats()["total"] == 0 and storage.get_verification_log() == []
    assert storage.get_last_verification_for_client("c") is None


def test_audit_log_keeps_routing_degraded_reasons_model_version_threshold(people):
    storage.append_verification_log("X", "Forged", 61.0, "multimodal", "alice", routing="fused",
                                    degraded=True, reasons=["spatial gate failed", "x"],
                                    model_version="v0.10.0", threshold=0.7564)
    storage.append_verification_log("X", "Genuine", 90.0, "scan", "alice")
    first, second = storage.get_verification_log("alice")
    assert first["routing"] == "fused" and first["degraded"] is True
    assert first["reasons"] == ["spatial gate failed", "x"]
    assert first["model_version"] == "v0.10.0" and first["threshold"] == 0.7564
    assert second["routing"] is None and second["degraded"] is None and second["reasons"] is None
    rows = sqlite3.connect(storage._get_db().db_path).execute(
        "SELECT routing, degraded, model_version, threshold FROM audit_log WHERE event = 'verify' ORDER BY id"
    ).fetchall()
    assert rows[0] == ("fused", 1, "v0.10.0", 0.7564)       # plaintext and queryable


# ------------------------------------------------------- at-rest protection
def test_embeddings_are_not_readable_in_the_file(people, isolated_storage):
    embs = spatial(2)
    storage.save_client_enrollment("Jane Doe", [b64(128, 5)], "alice", spatial_embeddings=embs, spatial_origin="scan")
    data = db_bytes(isolated_storage)
    for e in embs + [b64(128, 5)]:
        assert base64.b64decode(e) not in data and e.encode() not in data


def test_no_json_storage_code_is_left():
    for name in ("_load_json", "_save_json", "load_employees", "load_enrollments",
                 "EMPLOYEES_FILE", "ENROLLMENTS_FILE", "LOG_FILE"):
        assert not hasattr(storage, name)
    with open(storage.__file__, encoding="utf-8") as f:
        source = f.read()
    assert "import json" not in source and ".json" not in source


# ------------------------------------------------- key: logins only (item 6)
def test_no_key_file_is_ever_written_and_the_cwd_stays_clean(people, tmp_path, monkeypatch, isolated_storage):
    storage.save_client_enrollment("Persist", [], "alice", spatial_embeddings=spatial(1), spatial_origin="scan")
    elsewhere = tmp_path / "some" / "other" / "dir"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    storage._reset()
    login("alice", PW_A)
    assert storage.get_client("persist", "alice") is not None
    assert sorted(os.listdir(elsewhere)) == []                     # nothing created in the cwd
    assert sorted(os.listdir(isolated_storage)) == ["companies"]
    assert sorted(os.listdir(company_dir(isolated_storage))) == ["company.txt", "signatures.db"]   # no db.key


def test_nothing_decrypts_before_login_or_after_logout(people):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(1), spatial_origin="scan")
    storage.logout()
    with pytest.raises(storage.StorageLockedError):
        storage.get_client("c")
    assert storage.save_client_enrollment("C2", [], "alice", spatial_embeddings=spatial(1), spatial_origin="scan")[0] is False
    storage._reset()                                                # the app restarts
    with pytest.raises(storage.StorageLockedError):
        storage.list_clients()
    login("bob", PW_B)                                              # any employee's login opens it
    assert storage.get_client("c")["spatial_embeddings"] == spatial(1)


def test_a_copied_data_folder_cannot_be_opened(people, isolated_storage, tmp_path, monkeypatch):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(1), spatial_origin="scan")
    stolen = tmp_path / "stolen"
    shutil.copytree(isolated_storage, stolen)
    monkeypatch.setenv(storage.HOME_ENV_VAR, str(stolen))
    storage._reset()
    storage.open_company(COMPANY)
    with pytest.raises(storage.StorageLockedError):
        storage.get_client("c")
    assert storage.authenticate_employee("alice", "guess-guess-1")[0] is False


def test_passphrase_env_var_unlocks_for_tests_and_ci_without_a_file(isolated_storage, monkeypatch):
    monkeypatch.setenv(PASSPHRASE_ENV_VAR, "from-environment")
    storage._reset()
    assert storage.create_company(COMPANY, "E", "env", "env-password-1")[0]
    storage.save_client_enrollment("C", [], "env", spatial_embeddings=spatial(1), spatial_origin="scan")
    storage._reset()
    storage.open_company(COMPANY)
    assert storage.get_client("c") is not None                      # no login needed
    storage.logout()
    assert storage.get_client("c") is not None                      # and logout does not lock it
    assert sorted(os.listdir(company_dir(isolated_storage))) == ["company.txt", "signatures.db"]
    monkeypatch.setenv(PASSPHRASE_ENV_VAR, "a-different-one")
    storage._reset()
    storage.open_company(COMPANY)
    with pytest.raises(storage.StorageKeyError):
        storage.get_client("c")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_data_folder_is_private(people, isolated_storage):
    for folder in (isolated_storage, isolated_storage / "companies", company_dir(isolated_storage)):
        assert (os.stat(folder).st_mode & 0o777) == 0o700


def test_an_old_database_is_refused_with_the_folder_to_move(isolated_storage):
    folder = company_dir(isolated_storage)
    folder.mkdir(parents=True)
    (folder / "company.txt").write_text(COMPANY)
    _make_v4(str(folder / "signatures.db"))
    storage.open_company(COMPANY)
    with pytest.raises(storage.StorageError, match="older version") as e:
        storage.authenticate_employee("bob", OLD_B)
    assert str(folder) in str(e.value)


# ------------------------------------------------- several sets (issue #33, item 3)
def test_scan_then_render_are_stored_side_by_side(people):
    scan, render = spatial(5, 0), spatial(5, 100)
    assert storage.save_client_enrollment("C", [], "alice", spatial_embeddings=scan, spatial_origin="scan")[0]
    assert storage.save_client_enrollment("C", pen(), "alice", spatial_embeddings=render, spatial_origin="render")[0]
    rec = storage.get_client("c", "alice")
    assert rec["spatial_sets"]["scan"]["embeddings"] == scan
    assert rec["spatial_sets"]["render"]["embeddings"] == render
    assert rec["temporal_set"]["embeddings"] == pen()
    assert rec["spatial_embeddings"] == scan and rec["spatial_origin"] == "scan"   # old keys: scan first
    assert storage.get_reference_embeddings("c", spatial_origin="render") == (render, "render")


def test_reenrolling_one_set_leaves_the_others_untouched(people):
    scan, render = spatial(5, 0), spatial(5, 100)
    storage.save_client_enrollment("C", pen(), "alice", spatial_embeddings=scan, spatial_origin="scan")
    storage.save_client_enrollment("C", None, "alice", spatial_embeddings=render, spatial_origin="render")
    new_scan = spatial(5, 500)
    storage.save_client_enrollment("C", None, "alice", spatial_embeddings=new_scan, spatial_origin="scan")
    rec = storage.get_client("c", "alice")
    assert rec["spatial_sets"]["scan"]["embeddings"] == new_scan and len(new_scan) == 5
    assert rec["spatial_sets"]["render"]["embeddings"] == render
    assert rec["temporal_set"]["embeddings"] == pen()


def test_reenrolling_the_pen_set_replaces_it_and_keeps_the_spatial_sets(people):
    scan = spatial(5, 0)
    storage.save_client_enrollment("C", pen(base=700), "alice", spatial_embeddings=scan, spatial_origin="scan")
    storage.save_client_enrollment("C", pen(base=800), "alice", spatial_embeddings=None, spatial_origin="render")
    rec = storage.get_client("c", "alice")
    assert rec["temporal_set"]["embeddings"] == pen(base=800)      # replaced, 5 not 10
    assert rec["spatial_sets"]["scan"]["embeddings"] == scan       # untouched


def test_resaving_without_new_pen_signatures_keeps_the_pen_set_and_its_added_signings(people, monkeypatch):
    storage.save_client_enrollment("C", pen(), "alice", spatial_embeddings=spatial(), spatial_origin="scan")
    assert storage.add_confirmed_reference("c", b64(128, 999), "alice", modality="temporal")
    monkeypatch.setattr(storage, "MODEL_VERSION", "v-next")
    assert storage.save_client_enrollment("C", None, "alice", spatial_embeddings=spatial(base=50), spatial_origin="scan")[0]
    t = storage.get_client("c")["temporal_set"]
    assert t["embeddings"] == pen() + [b64(128, 999)]
    assert t["stale"] is True                               # still labelled with the old version
    assert storage.add_confirmed_reference("c", b64(128, 1000), "alice", modality="temporal") is False
    assert len(storage.get_client("c")["dynamic_references"]) == 6     # a stale set takes no more rows


def _pen_rows():
    with sqlite3.connect(storage._get_db().db_path) as conn:
        return conn.execute("SELECT source, model_version FROM enrollments WHERE modality = 'temporal' "
                            "ORDER BY id").fetchall()


def test_resaving_without_new_pen_keeps_pen_rows_and_the_next_signing_is_kept(people):
    storage.save_client_enrollment("C", pen(), "alice", spatial_embeddings=spatial(), spatial_origin="scan")
    assert storage.add_confirmed_reference("c", b64(128, 999), "alice", modality="temporal")
    before = _pen_rows()
    assert storage.save_client_enrollment("C", None, "alice", spatial_embeddings=spatial(base=50), spatial_origin="scan")[0]
    assert _pen_rows() == before and before[-1][0] == "added"
    assert storage.get_client("c")["temporal_set"]["stale"] is False
    assert storage.add_confirmed_reference("c", b64(128, 1000), "alice", modality="temporal") is True
    assert storage.get_client("c")["temporal_set"]["embeddings"][-1] == b64(128, 1000)


def test_each_set_has_its_own_model_version_and_stale_flag(people, monkeypatch):
    old_version = storage.MODEL_VERSION
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(1), spatial_origin="scan")
    monkeypatch.setattr(storage, "MODEL_VERSION", "v-next")
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(1, 9), spatial_origin="render")
    sets = storage.get_client("c", "alice")["spatial_sets"]
    assert sets["render"]["model_version"] == "v-next" and sets["render"]["stale"] is False
    assert sets["scan"]["model_version"] == old_version != "v-next"
    assert sets["scan"]["stale"] is True
    datetime.fromisoformat(sets["scan"]["enrolled_at"])


def test_one_row_from_other_weights_makes_the_set_stale(people):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(5), spatial_origin="scan")
    assert storage.add_confirmed_reference("c", b64(512, 99), "alice")
    _set_last_row_version("v-next")
    assert storage.get_client("c")["spatial_sets"]["scan"]["stale"] is True


# ----------------------------------------- checks before verifying (items 7, 11)
def test_stale_references_are_not_verified_against(people, monkeypatch):
    storage.save_client_enrollment("C", pen(), "alice", spatial_embeddings=spatial(), spatial_origin="scan")
    assert storage.check_references("c", "scan", use_pen=True) == []
    monkeypatch.setattr(storage, "MODEL_VERSION", "v-next")
    problems = storage.check_references("c", "scan", use_pen=True)
    assert len(problems) == 2 and all("re-enrol" in p for p in problems)
    assert "scan" in problems[0] and "pen" in problems[1]
    assert storage.check_references("c", None, use_pen=False) == []   # nothing used, nothing to check


def test_a_pen_signing_from_another_tablet_or_system_is_refused(people):
    storage.save_client_enrollment("C", pen(), "alice", spatial_embeddings=None, capture_device=XPPEN)
    assert storage.check_references("c", use_pen=True, capture_device=dict(XPPEN)) == []
    assert storage.check_references("c", use_pen=True,
                                    capture_device={**XPPEN, "device_model": "xppen deco 640 "}) == []
    other = storage.check_references("c", use_pen=True, capture_device={**XPPEN, "device_model": "Wacom One"})
    assert len(other) == 1 and "Wacom One" in other[0] and "XPPen Deco 640" in other[0]
    assert len(storage.check_references("c", use_pen=True, capture_device={**XPPEN, "os": "Linux"})) == 1
    assert storage.check_references("c", use_pen=False, capture_device={"device_model": "Wacom"}) == []


def test_a_pen_set_enrolled_before_devices_were_recorded_is_not_checked(people):
    storage.save_client_enrollment("C", pen(), "alice", spatial_embeddings=None)
    assert storage.check_references("c", use_pen=True, capture_device={"device_model": "Any"}) == []


# ------------------------------------------------------------ replay (item 8)
def test_a_replayed_probe_is_refused_and_two_real_signings_are_not(people):
    first = probe_fingerprint(base64.b64decode(b64(512, 1)))
    second = probe_fingerprint(base64.b64decode(b64(512, 2)))
    assert not storage.is_replayed_probe("C", [first])
    storage.append_verification_log("C", "Genuine", 90.0, "scan", "alice", probe_fingerprints=[first])
    assert storage.is_replayed_probe("c", [first])            # the same file again
    assert not storage.is_replayed_probe("c", [second])       # a new signing
    storage.append_verification_log("C", "Genuine", 91.0, "scan", "alice", probe_fingerprints=[second])
    assert storage.get_verification_stats()["genuine"] == 2


def test_an_enrolment_scan_given_again_as_the_probe_is_a_replay(people):
    refs = spatial(5)
    storage.save_client_enrollment("C", pen(), "alice", spatial_embeddings=refs, spatial_origin="scan")
    for emb in (refs[0], pen()[2]):
        assert storage.is_replayed_probe("c", [probe_fingerprint(base64.b64decode(emb))])
    assert not storage.is_replayed_probe("c", [probe_fingerprint(base64.b64decode(b64(512, 4242)))])


def test_a_non_admin_cannot_swap_a_clients_name(people):
    storage.save_client_enrollment("Client", None, "alice", spatial_embeddings=spatial(1), spatial_origin="scan")
    login("bob", PW_B)
    storage.save_client_enrollment("CLIENT", pen(), "bob", spatial_embeddings=None)
    assert storage.get_client("client")["display_name"] == "Client"


# --------------------------------------------------- confirmed signings (restored)
def test_reference_embeddings_are_enrolled_then_added(people):
    enrolled = spatial(5)
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=enrolled)
    added = b64(512, 99)
    assert storage.add_confirmed_reference("c", added, "alice") is True
    refs, origin = storage.get_reference_embeddings("C", "alice")
    assert (refs, origin) == (enrolled + [added], "scan")


def test_reference_embeddings_of_unknown_or_other_employees_client(people):
    storage.save_client_enrollment("c", [], "alice", spatial_embeddings=spatial(5))
    assert storage.get_reference_embeddings("ghost", "alice") == ([], "scan")
    assert storage.get_reference_embeddings("c", "bob") == ([], "scan")        # narrowed to Bob's clients
    assert storage.get_reference_embeddings("c") == (spatial(5), "scan")       # the company list


def test_any_employee_adds_a_confirmed_signing_but_not_to_an_unknown_client(people):
    storage.save_client_enrollment("c", [], "alice", spatial_embeddings=spatial(5))
    login("bob", PW_B)
    assert storage.add_confirmed_reference("c", b64(512, 7), "bob") is True
    assert storage.add_confirmed_reference("c", b64(512, 8), "ghost") is False
    assert storage.add_confirmed_reference("nobody", b64(512, 7), "alice") is False
    assert storage.get_reference_embeddings("c")[0] == spatial(5) + [b64(512, 7)]


def test_confirmed_references_cap_drops_oldest_added_never_enrolled(people):
    enrolled = spatial(5)
    storage.save_client_enrollment("c", [], "alice", spatial_embeddings=enrolled)
    added = [b64(512, 100 + i) for i in range(5)]
    for emb in added:
        assert storage.add_confirmed_reference("c", emb, "alice")
    refs = storage.get_reference_embeddings("c", "alice")[0]
    assert len(refs) == MAX_REFERENCES == 8
    assert refs == enrolled + added[2:]
    assert not hasattr(storage, "MAX_SPATIAL_REFERENCES") and not hasattr(storage, "MAX_TEMPORAL_REFERENCES")


def test_confirmed_reference_goes_into_the_set_of_its_own_origin(people):
    storage.save_client_enrollment("c", [b64(128, 1)], "alice", spatial_embeddings=spatial(5), spatial_origin="render")
    added = b64(512, 50)
    assert storage.add_confirmed_reference("c", added, "alice") is False      # a scan: no scan set
    assert storage.add_confirmed_reference("c", added, "alice", spatial_origin="render")
    assert storage.get_reference_embeddings("c", "alice") == (spatial(5) + [added], "render")
    with pytest.raises(ValueError):
        storage.add_confirmed_reference("c", added, "alice", spatial_origin="sideways")


def test_a_rendered_signing_never_lands_in_the_scan_set(people):
    scan, render = spatial(5, 0), spatial(5, 100)
    storage.save_client_enrollment("C", None, "alice", spatial_embeddings=scan, spatial_origin="scan")
    storage.save_client_enrollment("C", None, "alice", spatial_embeddings=render, spatial_origin="render")
    drawn = b64(512, 999)
    assert storage.add_confirmed_reference("c", drawn, "alice", spatial_origin="render")
    sets = storage.get_client("c")["spatial_sets"]
    assert sets["scan"]["embeddings"] == scan
    assert sets["render"]["embeddings"] == render + [drawn]


def test_confirmed_reference_insert_and_trim_are_one_transaction(people):
    db = storage._get_db()
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(5), spatial_origin="scan")
    for i in range(6):
        db.add_confirmed_reference("c", "spatial", np.full(512, i, dtype=np.float32), storage.MODEL_VERSION, "scan", max_per_set=7)
    assert len(db.get_references("c", "spatial", "scan")) == 7      # 5 enrolled kept, oldest added dropped
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("""CREATE TRIGGER boom BEFORE DELETE ON enrollments BEGIN
                        SELECT RAISE(ABORT, 'no trim'); END""")
    with pytest.raises(sqlite3.DatabaseError):
        db.add_confirmed_reference("c", "spatial", np.full(512, 99, dtype=np.float32), storage.MODEL_VERSION, "scan", max_per_set=7)
    assert len(db.get_references("c", "spatial", "scan")) == 7      # the insert was rolled back too


def test_an_ownerless_row_is_started_clean_by_the_first_save(people):
    db = storage._get_db()
    db._add_references("orphan", "spatial", [np.zeros(512, dtype=np.float32)], "old", spatial_origin="render")
    assert not storage.client_exists("orphan")                  # invisible while ownerless
    login("bob", PW_B)
    assert storage.save_client_enrollment("orphan", [], "bob", spatial_embeddings=spatial(1), spatial_origin="scan")[0]
    rec = storage.get_client("orphan")
    assert rec["enrolled_by"] == "bob" and set(rec["spatial_sets"]) == {"scan"}


# ------------------------------------------------- backup / restore (item 9)
def test_backup_and_restore(people, tmp_path):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(5), spatial_origin="scan")
    folder = tmp_path / "usb"
    folder.mkdir()
    login("bob", PW_B)                                             # only the admin backs up
    assert storage.backup_database(str(folder), "bob")[0] is False
    login("alice", PW_A)
    ok, path = storage.backup_database(str(folder), "alice")
    assert ok and os.path.dirname(path) == str(folder) and os.path.basename(path).endswith(".db")
    assert storage.backup_database(str(tmp_path / "missing"), "alice")[0] is False
    assert storage.delete_client("C", "alice")[0]

    login("bob", PW_B)
    assert storage.restore_database(path, "bob", PW_B) == (False, "Only the admin can restore a backup.")
    login("alice", PW_A)
    ok, msg = storage.restore_database(path, "alice", "wrong-password-1")
    assert not ok and "does not open" in msg and storage.get_client("c") is None    # nothing replaced
    assert storage.restore_database(path, "alice", PW_A) == (True, "Backup restored")
    assert storage.get_client("c")["spatial_embeddings"] == spatial(5)


def test_a_backup_can_be_restored_on_a_new_install(people, tmp_path, monkeypatch):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(2), spatial_origin="scan")
    folder = tmp_path / "usb"
    folder.mkdir()
    path = storage.backup_database(str(folder), "alice")[1]
    monkeypatch.setenv(storage.HOME_ENV_VAR, str(tmp_path / "new-pc"))
    storage._reset()
    assert storage.is_first_run()
    assert storage.restore_company(COMPANY, path, "alice", PW_A) == (True, "Backup restored")
    assert storage.get_client("c")["spatial_embeddings"] == spatial(2)
    storage._reset()
    storage.open_company(COMPANY)
    assert storage.authenticate_employee("bob", PW_B)[0]           # same passwords as before


def _backup_of_c(tmp_path):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(5), spatial_origin="scan")
    folder = tmp_path / "usb"
    folder.mkdir()
    return storage.backup_database(str(folder), "alice")[1]


def test_a_non_admin_session_cannot_restore_and_nothing_changes(people, tmp_path, isolated_storage):
    path = _backup_of_c(tmp_path)
    login("bob", PW_B)
    live = db_bytes(isolated_storage)
    ok, msg = storage.restore_database(path, "alice", PW_A)
    assert not ok and "admin" in msg and db_bytes(isolated_storage) == live


def test_a_failing_copy_of_the_live_database_is_reported_and_nothing_changes(people, tmp_path, isolated_storage,
                                                                             monkeypatch):
    path = _backup_of_c(tmp_path)
    storage.save_client_enrollment("D", [], "alice", spatial_embeddings=spatial(5, 9), spatial_origin="scan")
    live, session = db_bytes(isolated_storage), dict(storage._session)

    def disk_full(self, actor_id, dest):
        raise OSError("No space left on device")
    monkeypatch.setattr(DatabaseManager, "backup_to", disk_full)
    ok, msg = storage.restore_database(path, "alice", PW_A)
    assert not ok and "No space left on device" in msg
    assert db_bytes(isolated_storage) == live and storage._session == session
    assert storage.client_exists("d") and storage.get_client("c")["spatial_embeddings"] == spatial(5)


def test_a_database_error_during_restore_is_reported(people, tmp_path, monkeypatch):
    path = _backup_of_c(tmp_path)

    def locked(self, *args):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(DatabaseManager, "restore_from", locked)
    ok, msg = storage.restore_database(path, "alice", PW_A)
    assert not ok and "database is locked" in msg and storage._session["username"] == "alice"


def test_restore_passes_the_session_employee_to_the_database(people, tmp_path, monkeypatch):
    path = _backup_of_c(tmp_path)
    seen = []
    real = DatabaseManager.restore_from

    def spy(self, actor_id, *args):
        seen.append(actor_id)
        return real(self, actor_id, *args)
    monkeypatch.setattr(DatabaseManager, "restore_from", spy)
    alice = storage._session["id"]
    assert storage.restore_database(path, "alice", PW_A) == (True, "Backup restored")
    assert seen == [alice]


def test_backup_needs_a_login(people, tmp_path):
    storage.logout()
    ok, msg = storage.backup_database(str(tmp_path), "alice")
    assert not ok and "Log in" in msg


def test_a_damaged_backup_is_refused_before_anything_is_replaced(people, tmp_path):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(5), spatial_origin="scan")
    backup = storage._get_db().backup_to(storage._session["id"], str(tmp_path / "b.db"))
    with sqlite3.connect(backup) as conn:   # an index that no longer matches its table
        conn.execute("CREATE INDEX damaged ON audit_log(event)")
        conn.execute("PRAGMA writable_schema = ON")
        conn.execute("UPDATE sqlite_master SET sql = 'CREATE INDEX damaged ON audit_log(decision)' "
                     "WHERE name = 'damaged'")
    with sqlite3.connect(backup) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok"   # opens, but is damaged
    storage.save_client_enrollment("D", [], "alice", spatial_embeddings=spatial(5, 9), spatial_origin="scan")
    ok, _ = storage.restore_database(backup, "alice", PW_A)
    assert not ok and storage.client_exists("d")             # live database untouched


def test_an_older_added_row_makes_a_current_set_stale(people):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(5), spatial_origin="scan")   # enrolled at MODEL_VERSION
    assert storage.add_confirmed_reference("c", b64(512, 99), "alice")
    _set_last_row_version("v-older")
    found = storage.get_client("c")["spatial_sets"]["scan"]
    assert found["model_version"] == storage.MODEL_VERSION and found["stale"] is True


def _set_last_row_version(version):
    with sqlite3.connect(storage._get_db().db_path) as conn:
        conn.execute("UPDATE enrollments SET model_version = ? WHERE id = (SELECT MAX(id) FROM enrollments)",
                     (version,))


def test_a_confirmed_signing_to_a_stale_set_is_refused_and_writes_nothing(people):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(5), spatial_origin="scan")
    _set_last_row_version("v-older")
    assert storage.add_confirmed_reference("c", b64(512, 99), "alice") is False
    assert storage.get_client("c")["spatial_embeddings"] == spatial(5)


def test_a_confirmed_signing_to_a_missing_set_is_refused_by_the_database(people):
    db = storage._get_db()
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(5), spatial_origin="scan")
    with pytest.raises(LookupError):
        db.add_confirmed_reference("c", "spatial", np.zeros(512, dtype=np.float32), storage.MODEL_VERSION, "render")
    with pytest.raises(LookupError):
        db.add_confirmed_reference("c", "temporal", np.zeros(128, dtype=np.float32), storage.MODEL_VERSION)
    with pytest.raises(LookupError):
        db.add_confirmed_reference("ghost", "spatial", np.zeros(512, dtype=np.float32), storage.MODEL_VERSION, "scan")


@pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX only")
def test_the_backup_file_is_private_to_its_owner(people, tmp_path):
    ok, path = storage.backup_database(str(tmp_path), "alice")
    assert ok and os.stat(path).st_mode & 0o777 == 0o600


def test_a_pen_set_with_a_stored_device_needs_the_current_device(people):
    storage.save_client_enrollment("C", pen(), "alice", spatial_embeddings=None, spatial_origin="render",
                                   capture_device=XPPEN)
    (message,) = storage.check_references("c", None, use_pen=True, capture_device=None)
    assert message == ("This pen signing did not say which tablet it came from; the pen references were "
                       "signed on XPPen Deco 640 (Windows). Sign again on the tablet.")
    assert storage.check_references("c", None, use_pen=True, capture_device=XPPEN) == []


def test_a_confirmed_pen_signing_needs_a_pen_set(people):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(5), spatial_origin="scan")
    assert storage.add_confirmed_reference("c", b64(128, 5), "alice", modality="temporal") is False
    assert storage.get_client("c")["temporal_set"] is None


# ------------------------------------------------- companies: one encrypted folder each
PW_BETA = "beta-admin-pass-1"


def test_two_companies_cannot_see_each_others_clients(people, isolated_storage):
    storage.save_client_enrollment("C", [], "alice", spatial_embeddings=spatial(1), spatial_origin="scan")
    ok, _ = storage.create_company("Beta Ltd", "Alice Beta", "alice", PW_BETA)   # same User ID, other company
    assert ok and storage.current_company() == "Beta Ltd"
    assert storage.current_employee() == {"full_name": "Alice Beta", "username": "alice", "is_admin": True}
    assert storage.list_clients() == [] and storage.get_client("c") is None and not storage.client_exists("c")
    assert storage.get_reference_embeddings("c") == ([], "scan")
    assert storage.list_companies() == ["Beta Ltd", COMPANY]
    storage.logout()
    assert storage.authenticate_employee("alice", PW_A) == (False, None, "Incorrect password")
    assert storage.authenticate_employee("bob", PW_B) == (False, None, "User ID not found")
    a_file = DatabaseManager(str(company_dir(isolated_storage) / "signatures.db"))
    assert a_file.authenticate_employee("alice", PW_BETA)[0] is None             # B's password opens nothing in A
    storage.open_company("test co")                                              # any case
    assert storage.current_company() == COMPANY and storage.current_employee() is None
    assert storage.authenticate_employee("alice", PW_A)[0]
    assert storage.get_client("c") is not None


def test_company_names_are_checked_and_unique(isolated_storage):
    for bad in ("", "   ", "A", "x" * 41, "!!", " - "):
        ok, msg = storage.create_company(bad, "Alice A", "alice", PW_A)
        assert not ok and "2 to 40 characters" in msg
    assert storage.is_first_run() and storage.list_companies() == []
    assert storage.create_company("  Acme Inc.  ", "Alice A", "alice", PW_A)[0]
    assert storage.list_companies() == ["Acme Inc."]
    assert (company_dir(isolated_storage, "acme-inc") / "company.txt").read_text() == "Acme Inc."
    assert storage.create_company("ACME inc", "Bob B", "bob", PW_B) == (
        False, "A company with this name already exists")
    assert storage.current_company() == "Acme Inc." and storage.current_employee()["username"] == "alice"


def test_marks_stay_inside_a_word_of_a_company_name():
    # Arabic diacritics and Devanagari vowel signs are part of the word, not separators.
    assert storage._slug("مُحَمَّد") == "مُحَمَّد"
    assert storage._slug("हिन्दी") == "हिन्दी"
    assert storage._slug("Acme  Bank / Riyadh") == "acme-bank-riyadh"
    assert storage._slug("../..") == ""


def test_company_names_in_any_script_get_their_own_folder(isolated_storage):
    for name in ("شركة النور", "Café", "Cafe"):                    # é is not e: two companies
        ok, msg = storage.create_company(name, "Alice A", "alice", PW_A)
        assert ok, msg
    assert storage.list_companies() == ["Cafe", "Café", "شركة النور"]
    assert sorted(os.listdir(isolated_storage / "companies")) == ["cafe", "café", "شركة-النور"]
    storage.open_company("شركة النور")
    assert storage.authenticate_employee("alice", PW_A)[0]


def test_a_company_name_never_leaves_the_companies_folder(isolated_storage):
    for name, slug in (("../x", "x"), ("/etc", "etc"), ("a/../../b", "a-b")):
        assert storage.create_company(name, "Alice A", "alice", PW_A)[0]
        assert (company_dir(isolated_storage, slug) / "company.txt").read_text() == name
    assert sorted(os.listdir(isolated_storage)) == ["companies"]
    for bad in ("..", "./", "__"):
        assert storage.create_company(bad, "Alice A", "alice", PW_A) == (
            False, "The company name must have 2 to 40 characters, with at least one letter or digit.")


def test_windows_device_names_are_refused_as_company_names(isolated_storage):
    for bad in ("CON", "prn", " Aux ", "nul.", "COM1", "lpt9"):
        ok, msg = storage.create_company(bad, "Alice A", "alice", PW_A)
        assert not ok and msg == f"'{bad.strip()}' is a name Windows keeps for a device. Choose another company name."
    assert storage.is_first_run()
    assert storage.create_company("Con Ltd", "Alice A", "alice", PW_A)[0]
    assert storage.create_company("COM10", "Alice A", "alice", PW_A)[0]


def test_an_unreadable_companies_folder_is_a_storage_error(isolated_storage):
    isolated_storage.mkdir()
    (isolated_storage / "companies").write_text("not a folder")
    with pytest.raises(storage.StorageError, match="cannot be read"):
        storage.is_first_run()


def test_a_long_company_file_is_read_only_in_part(people, isolated_storage):
    (company_dir(isolated_storage) / "company.txt").write_text("x" * 5000)
    assert storage.list_companies() == ["x" * 200]


def test_a_failed_registration_leaves_no_company_folder(isolated_storage):
    ok, msg = storage.create_company("Acme", "Alice A", "alice", "short")
    assert not ok and "12 characters" in msg
    assert storage.is_first_run() and storage.current_company() is None
    assert not company_dir(isolated_storage, "acme").exists()
    assert storage.create_company("Acme", "Alice A", "alice", PW_A)[0]           # the name is free again


def test_a_company_is_listed_only_after_its_admin_exists(isolated_storage, monkeypatch):
    seen = []
    real = DatabaseManager.create_employee

    def spy(self, *args, **kw):                      # the scrypt step: a crash here must list nothing
        seen.append((storage.list_companies(), (company_dir(isolated_storage, "acme") / "company.txt").exists()))
        return real(self, *args, **kw)
    monkeypatch.setattr(DatabaseManager, "create_employee", spy)
    assert storage.create_company("Acme", "Alice A", "alice", PW_A)[0]
    assert seen == [([], False)] and storage.list_companies() == ["Acme"]


def test_a_database_error_while_registering_is_reported_and_leaves_no_folder(isolated_storage, monkeypatch):
    def disk_full(self, *args, **kw):
        raise sqlite3.OperationalError("database or disk is full")
    monkeypatch.setattr(DatabaseManager, "create_employee", disk_full)
    assert storage.create_company("Acme", "Alice A", "alice", PW_A) == (False, "database or disk is full")
    assert storage.is_first_run() and storage.current_company() is None
    assert not company_dir(isolated_storage, "acme").exists()


def test_unknown_company_and_no_company_chosen(people):
    with pytest.raises(storage.StorageError, match="No company named 'Nope'"):
        storage.open_company("Nope")
    assert storage.current_company() == COMPANY                                  # still selected
    storage._reset()
    assert storage.current_company() is None and not storage.is_first_run()
    for call in (storage.list_clients, lambda: storage.authenticate_employee("alice", PW_A),
                 storage.list_employees):
        with pytest.raises(storage.StorageLockedError, match="Choose a company first"):
            call()


def test_logout_keeps_the_company_selected(people):
    storage.logout()
    assert storage.current_company() == COMPANY and storage.current_employee() is None
    assert storage.authenticate_employee("bob", PW_B)[0]


def test_restore_company_makes_a_new_company_from_a_backup(people, tmp_path, isolated_storage):
    path = _backup_of_c(tmp_path)
    new = company_dir(isolated_storage, "branch-two")
    for user, password in (("bob", PW_B), ("alice", "wrong-password-1")):        # not an admin / wrong password
        ok, msg = storage.restore_company("Branch Two", path, user, password)
        assert not ok and msg and not new.exists() and storage.current_company() == COMPANY
    assert storage.restore_company("Branch Two", str(tmp_path / "missing.db"), "alice", PW_A)[0] is False
    assert not new.exists()
    assert storage.restore_company(COMPANY, path, "alice", PW_A) == (
        False, "A company with this name already exists")
    assert storage.restore_company("Branch Two", path, "alice", PW_A) == (     # Test Co's backup
        False, "This backup belongs to company 'Test Co'.")
    assert not new.exists()
    with closing(sqlite3.connect(path)) as conn:                                 # a backup with no name recorded
        conn.execute("DELETE FROM company_info")
        conn.commit()
    assert storage.restore_company("Branch Two", path, "alice", PW_A) == (True, "Backup restored")
    assert storage._get_db().company_name() == "Branch Two"
    assert storage.current_company() == "Branch Two" and storage.current_employee()["username"] == "alice"
    assert storage.get_client("c")["spatial_embeddings"] == spatial(5)
    assert sorted(os.listdir(new)) == ["company.txt", "signatures.db"]
    login("alice", PW_A)                                                         # Test Co is untouched
    assert storage.current_company() == COMPANY and storage.get_client("c") is not None
    storage.open_company("Branch Two")
    assert storage.authenticate_employee("bob", PW_B)[0]                         # the backup's passwords


def test_a_company_records_its_name_and_refuses_another_companys_backup(people, tmp_path, isolated_storage):
    assert storage._get_db().company_name() == COMPANY
    assert storage.create_company("Beta Ltd", "Alice Beta", "alice", PW_BETA)[0]
    assert storage._get_db().company_name() == "Beta Ltd"
    beta_backup = storage.backup_database(str(tmp_path), "alice")[1]
    login("alice", PW_A)
    live = db_bytes(isolated_storage)
    assert storage.restore_database(beta_backup, "alice", PW_BETA) == (
        False, "This backup belongs to company 'Beta Ltd'.")
    assert db_bytes(isolated_storage) == live and storage.current_employee()["username"] == "alice"


def test_last_event_for_backup_and_delete_and_survives_restore(people, tmp_path):
    assert storage.last_event(DatabaseManager.EVENT_BACKUP) is None
    assert storage.last_event(DatabaseManager.EVENT_DELETE) is None

    storage.save_client_enrollment("c", pen(), "alice", spatial_embeddings=spatial(5))

    # Perform a backup
    folder = tmp_path / "bak"
    folder.mkdir()
    ok, path = storage.backup_database(str(folder), "alice")
    assert ok
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("UPDATE audit_log SET timestamp = '2026-01-01T00:00:00'")
        conn.commit()
    with closing(sqlite3.connect(storage._get_db().db_path)) as conn:
        conn.execute("UPDATE audit_log SET timestamp = '2026-01-01T00:00:00'")
        conn.commit()

    ev_bak = storage.last_event(DatabaseManager.EVENT_BACKUP)
    assert ev_bak is not None
    assert ev_bak["employee"] == "alice"
    assert ev_bak["timestamp"]
    assert ev_bak["client"] is None

    # Delete client c
    ok, _ = storage.delete_client("c")
    assert ok
    ev_del = storage.last_event(DatabaseManager.EVENT_DELETE)
    assert ev_del is not None
    assert ev_del["employee"] == "alice"
    assert ev_del["client"] == "c"
    assert ev_del["timestamp"]

    # Restore the backup
    ok, msg = storage.restore_database(path, "alice", PW_A)
    assert ok

    # After restore, last_event(delete) still returns the pre-restore delete with the right username
    ev_del_after = storage.last_event(DatabaseManager.EVENT_DELETE)
    assert ev_del_after is not None
    assert ev_del_after["employee"] == "alice"
    assert ev_del_after["client"] == "c"

