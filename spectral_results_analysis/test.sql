WITH base AS (
    SELECT unit_idx, v AS v_a
    FROM rankings.rk_t5_A_tau20_rho0_m0110_chi50_alpha85
    WHERE unit_type = 'S'
),
sm AS (
    SELECT source_idx, source_name, field_subset
    FROM '/home/lc/m/working/econ_bus/parquet/source_master.parquet'
    WHERE field_subset IS NULL
)
SELECT sm.source_name, ROUND(base.v_a, 3) AS v_a
FROM base
JOIN sm ON base.unit_idx = sm.source_idx
ORDER BY v_a DESC
LIMIT 30;

-- WITH base AS (
--     SELECT unit_idx, v AS v_a
--     FROM rankings.rk_t5_A_tau20_rho0_m0110_chi50_alpha85
--     WHERE unit_type = 'S'
-- ),
-- fb AS (
--     SELECT unit_idx, v AS v_b
--     FROM rankings.rk_t5_B_tau20_rho0_m0110_chi50_alpha85
--     WHERE unit_type = 'S'
-- ),
-- fe AS (
--     SELECT unit_idx, v AS v_e
--     FROM rankings.rk_t5_E_tau20_rho0_m0110_chi50_alpha85
--     WHERE unit_type = 'S'
-- ),
-- sm AS (
--     SELECT source_idx, source_name, field_subset
--     FROM '/home/lc/m/working/econ_bus/parquet/source_master.parquet'
-- )
-- SELECT
--     sm.source_name,
--     sm.field_subset,
--     base.v_a,
--     fb.v_b,
--     fe.v_e,
--     ROUND(fb.v_b / base.v_a, 3) AS ratio_b_over_a,
--     ROUND(fe.v_e / base.v_a, 3) AS ratio_e_over_a
-- FROM base
-- JOIN sm ON base.unit_idx = sm.source_idx
-- LEFT JOIN fb ON base.unit_idx = fb.unit_idx
-- LEFT JOIN fe ON base.unit_idx = fe.unit_idx
-- ORDER BY ratio_b_over_a ASC NULLS LAST
-- LIMIT 40;
