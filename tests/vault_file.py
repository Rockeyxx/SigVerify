"""Test helper: read or edit a vault file's encrypted body through the production format code,
so tests keep checking the values really stored. Nothing here writes plaintext to disk."""
from contextlib import contextmanager

from src.database import db_manager as dm


def read_bytes(path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def header(path) -> dict:
    """The plaintext header of the vault file at `path`."""
    return dm._parse_vault(read_bytes(path))[0]


@contextmanager
def on_disk(path, dek):
    """The body stored in the file at `path`, decrypted into an in-memory connection."""
    conn = dm._memory_db(dm._decrypt_vault(read_bytes(path), dek))
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def edit_file(path, dek):
    """Edit the body of the file at `path` as someone holding the key would (a backup, say),
    then encrypt it again with a fresh nonce."""
    with on_disk(path, dek) as conn:
        yield conn
        dm._atomic_write(path, dm._encrypt_vault(conn, dek))


@contextmanager
def edit_live(db):
    """Edit an open store's in-memory body as someone holding the key would, then save it."""
    with db._lock:
        yield db._require_conn()
        db._persist()
