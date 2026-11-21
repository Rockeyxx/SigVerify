"""The whole vault file is encrypted and authenticated at rest (temporary files only)."""
import json
import os
import sqlite3
import stat
import sys
import tempfile

import numpy as np
import pytest

from src.database import db_manager as dm
from src.database.db_manager import DatabaseManager, VaultFileError, VaultInUseError
from tests.vault_file import header, on_disk, read_bytes

MODEL = "test-model-v1"
PW_A = "alice-password-1"
PW_B = "bob-password-22"


@pytest.fixture(autouse=True)
def fast_kdf(monkeypatch):
    monkeypatch.setattr(DatabaseManager, "KDF_N", 2 ** 10)


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def vault(path):
    """Owner alice ("Alice Smith"), member bob, signer client-777 with one verification."""
    db = DatabaseManager(path)
    a = db.create_employee("Alice Smith", "alice", PW_A)
    db.create_employee("Bob Jones", "bob", PW_B, created_by=a)
    rng = np.random.default_rng(0)
    db.save_client("client-777", "Client Seven", a, spatial_embeddings=[rng.standard_normal(512).astype(np.float32)],
                   spatial_origin="scan", temporal_embeddings=[], model_version=MODEL)
    db.log_verification("client-777", "fused", 87.1, "Genuine", a)
    return db


def files(folder):
    return sorted(os.listdir(folder))


def test_the_file_is_not_sqlite_and_holds_no_readable_name(vault, path):
    data = read_bytes(path)
    assert not data.startswith(b"SQLite format 3")
    with pytest.raises(sqlite3.DatabaseError), sqlite3.connect(path) as conn:
        conn.execute("SELECT name FROM sqlite_master").fetchall()
    for secret in (b"client-777", b"Client Seven", b"Alice Smith", b"Bob Jones", b"audit_log",
                   b"enrollments", b"Genuine", b"fused"):
        assert secret not in data


def test_the_header_holds_only_user_ids_and_key_slots(vault, path):
    head = header(path)
    assert set(head) == {"kdf", "r", "p", "slots", "passphrase"}
    assert set(head["slots"]) == {"alice", "bob"} and head["passphrase"] is None
    assert all(set(s) == {"salt", "n", "nonce", "wrapped"} for s in head["slots"].values())
    text = json.dumps(head)
    for word in ("Alice Smith", "Bob Jones", "admin", "owner", "client-777", "audit"):
        assert word not in text


def test_the_stored_body_is_the_real_database(vault, path):
    with on_disk(path, vault._dek) as conn:
        assert conn.execute("SELECT user_id, display_name FROM users").fetchall() == [("client-777", "Client Seven")]
        assert conn.execute("SELECT decision FROM audit_log WHERE event = 'verify'").fetchall() == [("Genuine",)]


def test_files_are_owner_only(vault, path, tmp_path):
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600 or sys.platform == "win32"
    backup = vault.backup_to(vault.find_employee("alice")["id"], str(tmp_path / "b.db"))
    assert stat.S_IMODE(os.stat(backup).st_mode) == 0o600 or sys.platform == "win32"


def _flip(path, offset):
    data = bytearray(read_bytes(path))
    data[offset] ^= 1
    with open(path, "wb") as f:
        f.write(data)


def test_a_changed_body_byte_is_refused_and_nothing_loads(vault, path):
    vault.lock()
    _flip(path, len(read_bytes(path)) - 40)
    db = DatabaseManager(path)
    with pytest.raises(VaultFileError, match="damaged or was changed"):
        db.authenticate_employee("alice", PW_A)
    assert not db.unlocked and db._conn is None


def test_a_changed_header_byte_is_refused_and_nothing_loads(vault, path):
    vault.lock()
    salt = header(path)["slots"]["bob"]["salt"].encode()
    data = read_bytes(path)
    at = data.index(salt)
    swapped = b"A" if data[at:at + 1] != b"A" else b"B"
    with open(path, "wb") as f:                     # bob's salt changed, alice logs in: the AAD catches it
        f.write(data[:at] + swapped + data[at + 1:])
    db = DatabaseManager(path)
    with pytest.raises(VaultFileError, match="damaged or was changed"):
        db.authenticate_employee("alice", PW_A)
    assert not db.unlocked and db._conn is None


def test_no_journal_wal_or_plaintext_temp_file_during_or_after_writes(vault, path, tmp_path, monkeypatch):
    seen = []
    real_replace = os.replace

    def watch(src, dst):
        seen.append((files(tmp_path), read_bytes(src)))
        real_replace(src, dst)

    def no_temp(*_a, **_k):
        raise AssertionError("a system temp file was made")

    monkeypatch.setattr(dm.os, "replace", watch)
    monkeypatch.setattr(tempfile, "mkstemp", no_temp)
    monkeypatch.setattr(tempfile, "mkdtemp", no_temp)
    vault.log_verification("client-777", "fused", 52.9, "Forged", vault.find_employee("alice")["id"])
    assert seen
    for listing, written in seen:
        assert not [f for f in listing if f.endswith(("-journal", "-wal", "-shm"))]
        assert b"client-777" not in written and not written.startswith(b"SQLite format 3")
    assert files(tmp_path) == ["test.db", "test.db.lock"]


def test_a_crash_before_the_replace_keeps_the_previous_file(vault, path, monkeypatch):
    before = read_bytes(path)
    alice = vault.find_employee("alice")["id"]

    def crash(*_a):
        raise OSError("power cut")

    monkeypatch.setattr(dm.os, "replace", crash)
    with pytest.raises(OSError, match="power cut"):
        vault.log_verification("client-777", "fused", 52.9, "Forged", alice)
    monkeypatch.undo()
    assert read_bytes(path) == before
    assert vault.count_log_events() == (1, 1)                 # memory went back to the file
    vault.lock()
    db = DatabaseManager(path)
    assert db.authenticate_employee("alice", PW_A)[1] == ""
    assert db.count_log_events() == (1, 1)


def test_a_temp_file_left_by_a_crash_is_removed_at_the_next_login(vault, path, tmp_path):
    vault.lock()
    (tmp_path / ".test.db.0123456789abcdef.tmp").write_bytes(b"left by a crash")
    DatabaseManager(path).authenticate_employee("alice", PW_A)
    assert files(tmp_path) == ["test.db", "test.db.lock"]


def test_login_reads_the_header_and_refuses_a_wrong_password(vault, path):
    vault.lock()
    db = DatabaseManager(path)
    assert db.has_employees() and not db.unlocked
    assert db.authenticate_employee("bob", "wrong-password-1") == (None, "bad_password")
    assert db.authenticate_employee("carol", PW_A) == (None, "not_found")
    assert not db.unlocked
    emp, reason = db.authenticate_employee("BOB", PW_B)
    assert reason == "" and emp["full_name"] == "Bob Jones" and not emp["is_admin"]


def test_a_removed_member_leaves_the_header_and_cannot_log_in(vault, path):
    vault.remove_employee(vault.find_employee("alice")["id"], "bob")
    assert set(header(path)["slots"]) == {"alice"}
    vault.lock()
    assert DatabaseManager(path).authenticate_employee("bob", PW_B) == (None, "not_found")


def test_reads_need_a_login(vault, path):
    vault.lock()
    db = DatabaseManager(path)
    with pytest.raises(dm.DatabaseLockedError):
        db.list_employees()
    with pytest.raises(dm.DatabaseLockedError):
        db.company_name()


def test_backup_and_restore_round_trip(vault, path, tmp_path):
    alice = vault.find_employee("alice")["id"]
    backup = vault.backup_to(alice, str(tmp_path / "b.db"))
    assert not read_bytes(backup).startswith(b"SQLite format 3") and b"client-777" not in read_bytes(backup)
    (tmp_path / "new").mkdir()
    fresh = DatabaseManager(str(tmp_path / "new" / "test.db"))
    emp = fresh.restore_from(None, backup, "alice", PW_A)
    assert emp["username"] == "alice" and fresh.has_client("client-777")
    fresh.lock()
    again = DatabaseManager(str(tmp_path / "new" / "test.db"))
    assert again.authenticate_employee("bob", PW_B)[1] == "" and again.has_client("client-777")


def test_a_failed_restore_leaves_the_live_file_byte_identical(vault, path, tmp_path, monkeypatch):
    alice = vault.find_employee("alice")["id"]
    backup = vault.backup_to(alice, str(tmp_path / "b.db"))
    live = read_bytes(path)
    with pytest.raises(PermissionError):
        vault.restore_from(alice, backup, "alice", "wrong-password-1")
    assert read_bytes(path) == live

    def full_disk(*_a):
        raise OSError("disk full")

    monkeypatch.setattr(dm, "_atomic_write", full_disk)      # fails at the very last step
    with pytest.raises(OSError, match="disk full"):
        vault.restore_from(alice, backup, "alice", PW_A)
    assert read_bytes(path) == live
    assert vault.unlocked and vault.has_client("client-777")   # the open store still works


def test_an_old_plaintext_database_is_refused_with_the_message(path):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE enrollments (id INTEGER)")
        conn.execute("PRAGMA user_version = 5")
    before = read_bytes(path)
    with pytest.raises(RuntimeError, match="older version of the app.*is older than"):
        DatabaseManager(path)
    assert read_bytes(path) == before


def test_a_second_instance_is_refused(vault, path):
    other = DatabaseManager(path)
    with pytest.raises(VaultInUseError, match="open in another window"):
        other.authenticate_employee("bob", PW_B)
    assert not other.unlocked
    vault.lock()
    assert other.authenticate_employee("bob", PW_B)[1] == ""


def test_view_database_tool_works_after_login(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SIGVER_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SIGVER_DB_PASSPHRASE", raising=False)
    from src.gui import storage
    from tools import view_database
    storage._reset()
    assert storage.create_company("Acme", "Alice Smith", "alice", PW_A)[0]
    storage.logout()
    folder = tmp_path / "home" / "companies" / "acme"
    before = {f: read_bytes(folder / f) for f in files(folder)}
    answers = iter(["Acme", "alice"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    monkeypatch.setattr(view_database.getpass, "getpass", lambda _p="": PW_A)
    try:
        view_database.main()
    finally:
        storage._reset()
    out = capsys.readouterr().out
    assert "employee added" in out and "Total clients: 0" in out
    assert {f: read_bytes(folder / f) for f in files(folder)} == before     # read only


def test_a_backup_too_large_to_be_a_vault_is_refused_before_it_is_read(vault, path, tmp_path, monkeypatch):
    alice = vault.find_employee("alice")["id"]
    live = read_bytes(path)
    backup = tmp_path / "huge.db"
    backup.write_bytes(live + bytes(200))
    monkeypatch.setattr(dm, "MAX_VAULT_BYTES", len(live) + 100)     # the live vault fits, the file does not
    with pytest.raises(ValueError, match="too large"):
        vault.restore_from(alice, backup, "alice", PW_A)
    assert read_bytes(path) == live


def test_sqlite_temp_structures_stay_in_memory(vault):
    assert vault._conn.execute("PRAGMA temp_store").fetchone() == (2,)      # MEMORY: no plaintext temp file


@pytest.mark.parametrize("where", ["magic", "version", "length", "nonce"])
def test_a_changed_frame_byte_is_refused(vault, path, where):
    vault.lock()
    length = int.from_bytes(read_bytes(path)[10:14], "big")
    _flip(path, {"magic": 0, "version": 9, "length": 13, "nonce": 14 + length}[where])
    with pytest.raises(VaultFileError):
        DatabaseManager(path).authenticate_employee("alice", PW_A)


def test_a_write_by_another_instance_during_login_is_not_lost(vault, path, monkeypatch):
    """B logs in while A still holds the vault; A saves and logs out during B's scrypt."""
    other = DatabaseManager(path)
    real_unwrap = other._unwrap

    def unwrap_while_a_writes(*args):
        dek = real_unwrap(*args)
        vault.log_verification("client-777", "fused", 52.9, "Forged", vault.find_employee("alice")["id"])
        vault.lock()
        return dek

    monkeypatch.setattr(other, "_unwrap", unwrap_while_a_writes)
    assert other.authenticate_employee("bob", PW_B)[1] == ""
    assert other.count_log_events() == (2, 1)                 # A's verification is there


def test_a_failed_restore_drops_the_backups_key_and_body(vault, path, tmp_path, monkeypatch):
    alice = vault.find_employee("alice")["id"]
    backup = vault.backup_to(alice, str(tmp_path / "b.db"))
    scratch = []
    real_lock = DatabaseManager.lock

    def spy(self):
        if self._backup is not None:
            scratch.append(self)
        real_lock(self)

    monkeypatch.setattr(DatabaseManager, "lock", spy)
    with pytest.raises(PermissionError):
        vault.restore_from(alice, backup, "bob", PW_B)          # bob opens it but is not its owner
    assert scratch and all(c._conn is None and c._dek is None for c in scratch)


def test_a_malformed_slot_nonce_is_refused_as_damaged(path):
    db = DatabaseManager(path, passphrase="pass-one")
    db.lock()
    data = read_bytes(path)
    head = header(path)
    bad = dict(head, passphrase=dict(head["passphrase"], nonce="AAAA"))
    raw = json.dumps(bad).encode()
    length = int.from_bytes(data[10:14], "big")
    with open(path, "wb") as f:
        f.write(data[:10] + len(raw).to_bytes(4, "big") + raw + data[14 + length:])
    with pytest.raises(VaultFileError, match="damaged"):
        DatabaseManager(path, passphrase="pass-one")
