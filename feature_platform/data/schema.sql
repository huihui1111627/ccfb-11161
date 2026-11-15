-- Source-of-truth schema for the synthetic consumer-lending dataset.
-- Timestamps are naive UTC wall-clock values.  `*_local` columns carry the
-- business timezone (UTC+8) as an ISO-8601 string and exist specifically to
-- create the classic "which column is event time?" trap.

CREATE TABLE IF NOT EXISTS users (
    user_id            VARCHAR PRIMARY KEY,
    register_time      TIMESTAMP NOT NULL,
    channel            VARCHAR NOT NULL,
    city_tier          VARCHAR NOT NULL,
    age                INTEGER NOT NULL,
    gender             VARCHAR NOT NULL,
    occupation         VARCHAR NOT NULL,
    education          VARCHAR NOT NULL,
    income_level       INTEGER NOT NULL,
    device_type        VARCHAR NOT NULL,
    is_employee        BOOLEAN NOT NULL
);

-- Slowly-changing profile snapshots.  A feature must take the latest
-- snapshot with snapshot_time < as_of, never the latest snapshot overall.
CREATE TABLE IF NOT EXISTS user_profile_history (
    profile_id         VARCHAR PRIMARY KEY,
    user_id            VARCHAR NOT NULL,
    snapshot_time      TIMESTAMP NOT NULL,
    credit_score       INTEGER NOT NULL,
    risk_level         VARCHAR NOT NULL,
    income_level       INTEGER NOT NULL,
    device_type        VARCHAR NOT NULL,
    app_version        VARCHAR NOT NULL,
    is_blacklist       BOOLEAN NOT NULL,
    ingested_at        TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS applications (
    application_id     VARCHAR PRIMARY KEY,
    user_id            VARCHAR NOT NULL,
    apply_time         TIMESTAMP NOT NULL,       -- canonical event time (UTC-naive)
    apply_time_local   VARCHAR NOT NULL,         -- ISO-8601 with +08:00 offset
    channel            VARCHAR NOT NULL,
    device_type        VARCHAR NOT NULL,
    app_version        VARCHAR NOT NULL,
    product_id         VARCHAR NOT NULL,
    apply_amount       DOUBLE NOT NULL,
    apply_status       VARCHAR NOT NULL,         -- final status; known only at decision_time
    reject_reason      VARCHAR,
    decision_time      TIMESTAMP,                -- when the status became known
    ip_city            VARCHAR,                  -- nullable: ~1.5%-9% missing, by design
    ingested_at        TIMESTAMP NOT NULL
);

-- `loans` is an append-only ingestion log: the same loan_id can appear more
-- than once after a re-ingest.  Consumers must dedupe by ingested_at.
CREATE TABLE IF NOT EXISTS loans (
    loan_id            VARCHAR NOT NULL,
    application_id     VARCHAR NOT NULL,
    user_id            VARCHAR NOT NULL,
    loan_time          TIMESTAMP NOT NULL,
    principal          DOUBLE NOT NULL,
    term_days          INTEGER NOT NULL,
    interest_rate      DOUBLE NOT NULL,
    product_id         VARCHAR NOT NULL,
    loan_status        VARCHAR NOT NULL,
    cancelled_time     TIMESTAMP,
    final_dpd          INTEGER,                  -- outcome, not a feature
    ingested_at        TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS repayments (
    repayment_id       VARCHAR PRIMARY KEY,
    loan_id            VARCHAR NOT NULL,
    user_id            VARCHAR NOT NULL,
    installment_no     INTEGER NOT NULL,
    due_time           TIMESTAMP NOT NULL,
    repay_time         TIMESTAMP,                -- NULL while unpaid
    amount_due         DOUBLE NOT NULL,
    amount_paid        DOUBLE NOT NULL,
    dpd_days           INTEGER,
    status             VARCHAR NOT NULL,
    reversal_time      TIMESTAMP,                -- payment reversal / chargeback
    schedule_known_at  TIMESTAMP NOT NULL,       -- when the instalment plan was created
    ingested_at        TIMESTAMP NOT NULL
);

-- Labels are derived from post-application outcomes and are only ever used
-- as targets.  Every label carries the time at which it became observable.
CREATE TABLE IF NOT EXISTS labels (
    application_id     VARCHAR PRIMARY KEY,
    loan_id            VARCHAR,
    label              INTEGER,                  -- 1 = 30+ DPD within 90 days
    max_dpd_90d        INTEGER,
    label_ready_time   TIMESTAMP,
    observation_end    TIMESTAMP,
    label_reason       VARCHAR
);
