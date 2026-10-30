-- Example questions about a player's progress, answered from the history
-- database. Change 'player' to the name used with --player.

-- 1. Has my elbow at contact on clears changed across sessions?
--    (180 degrees = a straight arm. A slight bend is the usual advice.)
SELECT c.played_on,
       count(*)                            AS clears,
       round(avg(s.elbow_angle)::numeric, 1) AS avg_elbow_at_contact
FROM shots s
JOIN clips c ON c.id = s.clip_id
WHERE c.player = 'player' AND s.shot_type = 'clear' AND s.elbow_angle IS NOT NULL
GROUP BY c.played_on
ORDER BY c.played_on;

-- 2. What did I play in each session?
SELECT c.played_on, c.video_name, s.shot_type, count(*) AS shots
FROM shots s
JOIN clips c ON c.id = s.clip_id
WHERE c.player = 'player'
GROUP BY c.played_on, c.video_name, s.shot_type
ORDER BY c.played_on, c.video_name, shots DESC;

-- 3. Which shots was the tool unsure how to name? (confidence below 0.7)
SELECT c.played_on, c.video_name, s.time_s, s.shot_type, s.alternatives, s.confidence
FROM shots s
JOIN clips c ON c.id = s.clip_id
WHERE c.player = 'player' AND s.confidence < 0.7
ORDER BY s.confidence, c.played_on;

-- 4. The feedback for my most recent session
SELECT c.played_on, c.video_name, f.window_label, f.text
FROM feedback f
JOIN clips c ON c.id = f.clip_id
WHERE c.player = 'player'
ORDER BY c.played_on DESC, c.saved_at DESC
LIMIT 1;
