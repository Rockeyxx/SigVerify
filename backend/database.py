import sqlite3
import os
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_FILE = "secret.key"

EXPECTED_DIMENSIONS = {
    "spatial": 512,
    "temporal": 128,
}


def get_or_create_key():
    """ISO/IEC 18033-1: Manage AES-256 key outside the database store."""
    if not os.path.exists(KEY_FILE):
        key = AESGCM.generate_key(bit_length=256)
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        return key
    with open(KEY_FILE, "rb") as f:
        return f.read()

class DatabaseManager:
    def __init__(self, db_path="signatures.db"):
        self.db_path = db_path
        self.key = get_or_create_key()
        self.aesgcm = AESGCM(self.key)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self):
        """ISO/IEC 9075-1: Relational SQLite schema initialization."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enrollments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    modality TEXT,
                    encrypted_embedding BLOB,
                    nonce BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    modality_state TEXT,
                    similarity_score REAL,
                    decision TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def encrypt_vector(self, vector: np.ndarray) -> tuple:
        """Encrypt continuous float32 embedding vector as BLOB."""
        raw_bytes = vector.astype(np.float32).tobytes()
        nonce = os.urandom(12)
        encrypted_bytes = self.aesgcm.encrypt(nonce, raw_bytes, None)
        return encrypted_bytes, nonce

    def decrypt_vector(self, encrypted_bytes: bytes, nonce: bytes) -> np.ndarray:
        decrypted_bytes = self.aesgcm.decrypt(nonce, encrypted_bytes, None)
        return np.frombuffer(decrypted_bytes, dtype=np.float32)

    def add_reference(self, user_id: str, modality: str, embedding: np.ndarray):
        if modality not in EXPECTED_DIMENSIONS:
            raise ValueError(f"Unknown modality '{modality}'. Expected one of {list(EXPECTED_DIMENSIONS)}")

        expected_dim = EXPECTED_DIMENSIONS[modality]
        if embedding.shape[0] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim}-d embedding for modality '{modality}', "
                f"got shape {embedding.shape}"
            )
        
        enc_blob, nonce = self.encrypt_vector(embedding)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
            cursor.execute("""
                INSERT INTO enrollments (user_id, modality, encrypted_embedding, nonce)
                VALUES (?, ?, ?, ?)
            """, (user_id, modality, enc_blob, nonce))
            conn.commit()

    def get_references(self, user_id: str, modality: str) -> list:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT encrypted_embedding, nonce FROM enrollments 
                WHERE user_id = ? AND modality = ?
            """, (user_id, modality))
            rows = cursor.fetchall()
            return [self.decrypt_vector(row[0], row[1]) for row in rows]

    def log_verification(self, user_id: str, modality_state: str, score: float, decision: str):
        """IEEE 730: Record verification event into plaintext queryable audit log."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_log (user_id, modality_state, similarity_score, decision)
                VALUES (?, ?, ?, ?)
            """, (user_id, modality_state, score, decision))
            conn.commit()