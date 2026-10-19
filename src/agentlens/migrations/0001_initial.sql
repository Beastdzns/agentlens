-- AgentLens initial schema migration
-- Version: 0001
-- Creates: events, traces tables with supporting indexes

-- ── events ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    trace_id        TEXT NOT NULL,
    parent_id       TEXT,
    agent_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    status          TEXT NOT NULL,
    timestamp       TEXT NOT NULL,   -- UTC ISO-8601, fixed precision, TEXT-sortable
    input_data      TEXT,            -- JSON
    output_data     TEXT,            -- JSON
    metadata        TEXT,            -- JSON
    latency_ms      REAL,
    importance_score REAL NOT NULL DEFAULT 0.5,
    observation_level TEXT NOT NULL DEFAULT 'full'
);

-- ── traces ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS traces (
    trace_id        TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    start_time      TEXT NOT NULL,   -- UTC ISO-8601, TEXT-sortable
    end_time        TEXT,
    event_count     INTEGER NOT NULL DEFAULT 0,
    total_latency_ms REAL NOT NULL DEFAULT 0.0,
    metadata        TEXT             -- JSON
);

-- ── indexes on events ──────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_events_trace_timestamp
    ON events (trace_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_events_agent_id
    ON events (agent_id);

CREATE INDEX IF NOT EXISTS idx_events_status
    ON events (status);

CREATE INDEX IF NOT EXISTS idx_events_timestamp
    ON events (timestamp);

-- ── indexes on traces ──────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_traces_agent_id
    ON traces (agent_id);

CREATE INDEX IF NOT EXISTS idx_traces_status
    ON traces (status);

CREATE INDEX IF NOT EXISTS idx_traces_start_time_desc
    ON traces (start_time DESC, trace_id DESC);
