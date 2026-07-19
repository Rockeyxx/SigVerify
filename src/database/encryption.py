"""
Encryption Module — Signature Verification System

AES-256-GCM application-layer encryption (ISO/IEC 18033-1) for stored
embedding payloads. The encryption key is held outside the template
store (protected key file / OS keyring / operator-credential derivation).

No SQLCipher. No whole-file DB encryption.
"""

# TODO: Implement key generation and secure key file management
# TODO: Implement AES-256-GCM encrypt/decrypt for embedding payloads
# TODO: Implement key loading from external storage (never from the database)
