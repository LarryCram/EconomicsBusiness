-- DESCRIBE SELECT * FROM '/home/lc/m/working/econ_bus/parquet/source_master.parquet' LIMIT 0;

SELECT DISTINCT type, count(*) FROM '/home/lc/m/openalex_feb26/parquet/sources.parquet' GROUP BY ALL;


-- SELECT unit_type, COUNT(*) AS n, MAX(v) AS max_v,
--        SUM(CASE WHEN v > 1 THEN 1 ELSE 0 END) AS n_over_1
-- FROM rk_t5_B_tau10_rho0_m0110_chi50_alpha85
-- GROUP BY unit_type ORDER BY unit_type;