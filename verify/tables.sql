-- Rebuild both published tables from results/stage1.csv and results/stage2.csv.
--
-- The tables in README.md are medians over three seeds. They were produced once,
-- by statistics.median in Python, and scripts/check_numbers.py then looks for
-- each figure somewhere in the prose. That catches drift in a number but not a
-- wrong aggregation: if the median were computed over the wrong grouping, the
-- checker would recompute it the same wrong way and agree with itself.
--
-- This derives every cell in SQLite instead and prints each table row in a
-- canonical form. verify/verify.sh requires each printed row to appear in the
-- README, cell for cell and in order, so a number in the right table but the
-- wrong row is a failure here.
--
-- Run: sqlite3 -init verify/tables.sql :memory: ""

.mode csv
.headers off
.import --csv results/stage1.csv s1
.import --csv results/stage2.csv s2

-- printf('%.0f') does not round the same way on every sqlite build: the runner
-- prints 670 where this laptop prints 671. The seconds columns therefore round
-- with ROUND and cast to an integer, which does mean the same thing everywhere.
--
-- Everything arrives as text, so every comparison and every median casts first.
-- Ordering strings would put 113.67 before 83.71 and quietly return the wrong
-- middle value.
CREATE TEMP VIEW long1 AS
    SELECT CAST(f AS INT) AS f, 'psnr' AS metric, CAST(psnr AS REAL) AS v FROM s1
    UNION ALL SELECT CAST(f AS INT), 'rfid', CAST(rfid AS REAL) FROM s1
    UNION ALL SELECT CAST(f AS INT), 'compression', CAST(compression AS REAL) FROM s1
    UNION ALL SELECT CAST(f AS INT), 'wall_s', CAST(wall_s AS REAL) FROM s1;

-- Median of an odd count is the middle element and of an even count the mean of
-- the two middle ones, which is what statistics.median does.
CREATE TEMP VIEW med1 AS
    SELECT f, metric, AVG(v) AS v FROM (
        SELECT f, metric, v,
               ROW_NUMBER() OVER (PARTITION BY f, metric ORDER BY v) AS rn,
               COUNT(*)     OVER (PARTITION BY f, metric)            AS c
        FROM long1)
    WHERE rn IN ((c + 1) / 2, (c + 2) / 2)
    GROUP BY f, metric;

CREATE TEMP VIEW range1 AS
    SELECT CAST(f AS INT) AS f,
           MIN(CAST(rfid AS REAL)) AS lo,
           MAX(CAST(rfid AS REAL)) AS hi,
           MIN(latent_shape)       AS shape,
           COUNT(*)                AS seeds
    FROM s1 GROUP BY CAST(f AS INT);

.mode list
.headers off
SELECT 'stage1|' || r.f
       || '|' || r.shape
       || '|' || printf('%.1f', (SELECT v FROM med1 WHERE f = r.f AND metric = 'compression')) || 'x'
       || '|' || printf('%.2f', (SELECT v FROM med1 WHERE f = r.f AND metric = 'psnr')) || ' dB'
       || '|' || printf('%.3f', (SELECT v FROM med1 WHERE f = r.f AND metric = 'rfid'))
       || '|' || printf('%.3f', r.lo) || ' to ' || printf('%.3f', r.hi)
       || '|' || CAST(ROUND((SELECT v FROM med1 WHERE f = r.f AND metric = 'wall_s')) AS INT) || ' s'
FROM range1 r ORDER BY r.f;

CREATE TEMP VIEW long2 AS
    SELECT model, CAST(nfe AS INT) AS nfe, 'cfid' AS metric, CAST(cfid AS REAL) AS v FROM s2
    UNION ALL SELECT model, CAST(nfe AS INT), 'sw2', CAST(sw2 AS REAL) FROM s2
    UNION ALL SELECT model, -1, 'train_s', CAST(train_s AS REAL) FROM s2;

CREATE TEMP VIEW med2 AS
    SELECT model, nfe, metric, AVG(v) AS v FROM (
        SELECT model, nfe, metric, v,
               ROW_NUMBER() OVER (PARTITION BY model, nfe, metric ORDER BY v) AS rn,
               COUNT(*)     OVER (PARTITION BY model, nfe, metric)            AS c
        FROM long2)
    WHERE rn IN ((c + 1) / 2, (c + 2) / 2)
    GROUP BY model, nfe, metric;

-- Ordered the way the table is, cheapest latent last.
CREATE TEMP VIEW models AS
    SELECT model, MIN(CAST(latent_elems AS INT)) AS elems, MIN(CAST(f AS INT)) AS f
    FROM s2 GROUP BY model;

SELECT 'stage2|' || m.model
       || '|' || m.elems
       || '|' || CAST(ROUND((SELECT v FROM med2 WHERE model = m.model AND metric = 'train_s')) AS INT) || ' s'
       || '|' || printf('%.2f', (SELECT v FROM med2 WHERE model = m.model AND metric = 'cfid' AND nfe = 10))
       || '|' || printf('%.2f', (SELECT v FROM med2 WHERE model = m.model AND metric = 'cfid' AND nfe = 25))
       || '|' || printf('%.2f', (SELECT v FROM med2 WHERE model = m.model AND metric = 'cfid' AND nfe = 50))
       || '|' || printf('%.4f', (SELECT v FROM med2 WHERE model = m.model AND metric = 'sw2'  AND nfe = 50))
FROM models m ORDER BY m.f;
