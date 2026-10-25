-- feature_engineering.sql
-- Aggregates raw Delta + AA transactions into one feature row per user.
-- Window functions used: SUM OVER (rolling 3-mo), SUM OVER (cumulative),
-- LAG (MoM growth), ROW_NUMBER (latest snapshot per user x program).

WITH all_tx AS (
    SELECT user_id, month, flights, points, 'DELTA' AS program FROM delta_transactions
    UNION ALL
    SELECT user_id, month, flights, points, 'AA'    AS program FROM aa_transactions
),

monthly AS (  -- one row per user x program x month
    SELECT user_id, program, month,
           SUM(points)  AS pts,
           SUM(flights) AS flts
    FROM all_tx
    GROUP BY 1, 2, 3
),

windowed AS (
    SELECT
        user_id, program, month, pts, flts,
        -- 3-month rolling average points: "Points Velocity"
        AVG(pts) OVER (PARTITION BY user_id, program ORDER BY month
                       ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)       AS velocity_3mo,
        -- cumulative points -> loyalty tier
        SUM(pts) OVER (PARTITION BY user_id, program ORDER BY month
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_pts,
        -- month-over-month growth
        pts - LAG(pts) OVER (PARTITION BY user_id, program ORDER BY month) AS mom_growth,
        -- most recent activity row per user x program
        ROW_NUMBER() OVER (PARTITION BY user_id, program ORDER BY month DESC) AS rn,
        COUNT(*) OVER (PARTITION BY user_id, program)                  AS active_months,
        MAX(month) OVER (PARTITION BY user_id, program)                AS last_active_month
    FROM monthly
),

latest AS (  -- snapshot at each user's latest active month, per program
    SELECT
        user_id, program,
        velocity_3mo,
        cum_pts,
        COALESCE(mom_growth, 0)            AS mom_growth,
        active_months,
        12 - last_active_month             AS recency_months,
        CASE WHEN cum_pts >= 50000 THEN 3   -- Platinum
             WHEN cum_pts >= 25000 THEN 2   -- Gold
             WHEN cum_pts >= 10000 THEN 1   -- Silver
             ELSE 0 END                    AS tier
    FROM windowed
    WHERE rn = 1
),

pivoted AS (  -- one row per user
    SELECT
        user_id,
        MAX(CASE WHEN program = 'DELTA' THEN velocity_3mo END) AS delta_velocity,
        MAX(CASE WHEN program = 'AA'    THEN velocity_3mo END) AS aa_velocity,
        MAX(CASE WHEN program = 'DELTA' THEN tier END)         AS delta_tier,
        MAX(CASE WHEN program = 'AA'    THEN tier END)         AS aa_tier,
        MAX(CASE WHEN program = 'DELTA' THEN recency_months END) AS delta_recency,
        MAX(CASE WHEN program = 'AA'    THEN recency_months END) AS aa_recency,
        SUM(active_months)                                     AS total_active_months,
        SUM(mom_growth)                                        AS mom_growth_total,
        COUNT(DISTINCT program)                                AS n_programs
    FROM latest
    GROUP BY user_id
)

SELECT
    u.user_id, u.age, u.income, u.region, u.tenure_months,
    COALESCE(p.delta_velocity, 0)            AS delta_velocity,
    COALESCE(p.aa_velocity, 0)               AS aa_velocity,
    COALESCE(p.delta_velocity, 0)
      + COALESCE(p.aa_velocity, 0)           AS total_velocity,
    COALESCE(p.delta_tier, -1)               AS delta_tier,   -- -1 = not a member
    COALESCE(p.aa_tier, -1)                  AS aa_tier,
    GREATEST(COALESCE(p.delta_tier, 0),
             COALESCE(p.aa_tier, 0))         AS max_tier,
    CASE WHEN p.n_programs = 2 THEN 1 ELSE 0 END AS cross_program,
    COALESCE(p.total_active_months, 0)       AS total_active_months,
    COALESCE(p.mom_growth_total, 0)          AS mom_growth_total,
    LEAST(COALESCE(p.delta_recency, 12),
          COALESCE(p.aa_recency, 12))        AS recency_months
FROM users u
LEFT JOIN pivoted p USING (user_id)
ORDER BY u.user_id;
