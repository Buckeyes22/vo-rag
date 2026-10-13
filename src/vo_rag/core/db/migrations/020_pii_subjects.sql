-- Reverse-lookup table from a subject hash to the envelope_ids and Qdrant
-- point_ids recorded for that subject.
--
-- The helpers in core/db/queries/pii_subjects.py read and write it, but
-- nothing calls them yet: purge is by source, not by data subject.

CREATE TABLE IF NOT EXISTS pii_subjects (
    subject_hash      TEXT PRIMARY KEY,
    first_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    envelope_ids      TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    qdrant_point_ids  JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_pii_subjects_first_seen
    ON pii_subjects(first_seen_at DESC);
