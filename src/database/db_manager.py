"""
Database Manager — Signature Verification System

SQLite relational schema (ISO/IEC 9075-1) for:
  - User identity and enrollment metadata
  - Encrypted reference embeddings (BLOB storage)
  - Verification audit log (IEEE 730)

Only derived embeddings are stored — never raw images or coordinate
streams (ISO/IEC 24745 compliance).
"""

# TODO: Implement database schema creation (users, embeddings, audit_log tables)
# TODO: Implement user CRUD operations
# TODO: Implement embedding storage/retrieval as BLOBs (float32, zero truncation)
# TODO: Implement audit log writer (timestamp, claimed ID, decision, score, routing)
# TODO: Implement replay mitigation (duplicate submission detection)
