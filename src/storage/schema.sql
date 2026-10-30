-- Blueprint of the optional history database (PostgreSQL).
-- One row per analyzed clip, one row per shot in it, and the feedback for it.
-- history.py runs this automatically before the first save; IF NOT EXISTS
-- makes it safe to run again.

CREATE TABLE IF NOT EXISTS clips (
    id          SERIAL PRIMARY KEY,
    player      TEXT        NOT NULL,
    video_name  TEXT        NOT NULL,
    played_on   DATE        NOT NULL,
    discipline  TEXT        NOT NULL CHECK (discipline IN ('singles', 'doubles')),
    racket_arm  TEXT        NOT NULL CHECK (racket_arm IN ('right', 'left')),
    shot_count  INTEGER     NOT NULL,
    saved_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- saving the same video again for the same player and day replaces it
    UNIQUE (player, video_name, played_on)
);

CREATE TABLE IF NOT EXISTS shots (
    id                    SERIAL PRIMARY KEY,
    clip_id               INTEGER NOT NULL REFERENCES clips (id) ON DELETE CASCADE,
    time_s                REAL    NOT NULL,   -- seconds into the clip
    shot_type             TEXT    NOT NULL,   -- e.g. 'clear', 'lift', 'serve'
    confidence            REAL,               -- 0 to 1, from the shot classifier
    alternatives          TEXT,               -- other possible names, comma-separated
    elbow_angle           REAL,               -- degrees at contact, 180 = straight
    knee_angle_min        REAL,               -- deepest knee bend before the shot, degrees
    wrist_height          TEXT,               -- racket hand at the fastest moment
    swing                 REAL,               -- racket-wrist travel, shoulder widths
    shuttle_rise          REAL,               -- shuttle climb above head height afterwards, shoulder widths
    shuttle_incoming_speed REAL,              -- shoulder widths per second
    shuttle_outgoing_speed REAL,              -- shoulder widths per second
    details               JSONB   NOT NULL    -- every measurement for the shot, as saved in shot_events.json
);

CREATE INDEX IF NOT EXISTS shots_clip_id   ON shots (clip_id);
CREATE INDEX IF NOT EXISTS shots_shot_type ON shots (shot_type);

CREATE TABLE IF NOT EXISTS feedback (
    id            SERIAL PRIMARY KEY,
    clip_id       INTEGER NOT NULL REFERENCES clips (id) ON DELETE CASCADE,
    window_label  TEXT    NOT NULL,   -- 'whole clip', or e.g. '0-25s' for long clips
    text          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS feedback_clip_id ON feedback (clip_id);
