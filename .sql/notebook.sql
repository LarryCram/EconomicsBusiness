"-- 0 CONVERT THE apenalex CLI return for econ_bus
-- ++++++++++++++++++++++++++++++++++++++++++++https://openalex.org/S83271458
SET preserve_insertion_order=FALSE;

CREATE OR REPLACE TEMP TABLE works AS (
  WITH 
    loader AS
    (SELECT id AS work_id,
         doi,
         title,
         institutions_distinct_count,
         publication_year, 
         referenced_works_count, 
         cited_by_count, 
         type,          
         is_retracted,  
         is_paratext,
         biblio.*, 
         primary_location.source.id AS source_id,  
         primary_location.source.display_name AS source_name, 
         primary_location.source.host_organization AS source_host,
         referenced_works,
         authorships,    
      FROM read_json_auto('/home/lc/m/openalex_feb26/json/**/*.json', ignore_errors = true)  
      -- LIMIT 16
      )
SELECT * FROM loader);

COPY works TO '/home/lc/m/openalex_feb26/parquet/works.parquet' (FORMAT PARQUET);

COPY (
  WITH 
    authorship_reducer AS
    (SELECT work_id, author_id, author_name,
            institution.id AS institution_id,
            institution.display_name AS institution_name,
            institution.ror AS ror,
            institution.country_code
      FROM 
        (SELECT work_id,
                authorship.author.id AS author_id,
                authorship.author.display_name AS author_name, unnest(authorship.institutions) AS institution
            FROM (SELECT work_id, unnest(authorships) AS authorship FROM works))
    )
  SELECT * FROM authorship_reducer)
TO '/home/lc/m/openalex_feb26/parquet/authorships.parquet' (FORMAT PARQUET); 

SELECT * FROM works LIMIT 4;
SHOW TABLES;"
"-- 1 SELECT works, sources, authors and instutions in the journal and institution sets
-- =================================================================================
CREATE OR REPLACE TEMP TABLE wsai AS
  SELECT work_id, source_id, author_id, id AS institution_id, publication_year, title, source_name, author_name, institution_name, country_code, referenced_works
  FROM '/home/lc/Projects/EconomicsBusiness/2026_study/DATA/econ_bus_ror_oa_SAVE.csv' incites
  LEFT JOIN
    (SELECT work_id, source_id, author_id, institution_id, title, source_name, author_name, institution_name, institutions_distinct_count, publication_year, referenced_works
      FROM '/home/lc/m/openalex_feb26/parquet/authorships.parquet'
      JOIN (SELECT * EXCLUDE (authorships) 
              FROM '/home/lc/m/openalex_feb26/parquet/works.parquet' 
              WHERE (publication_year BETWEEN 2020 AND 2024) AND list_contains(['article', 'report'], type) AND is_retracted=false AND is_paratext=false)
              USING (work_id)
      ) oa
   ON incites.id = oa.institution_id
   WHERE TRY_CAST(index AS INT) = 0 AND author_id NOT NULL;

SELECT * FROM wsai LIMIT 4;
SHOW TABLES;"
"-- 2 Compute the full WORK item including the per-row journal, author and institution weight
-- =======================================================================================
CREATE OR REPLACE TEMP TABLE work_items AS
    SELECT 
        work_id, 
        source_id,
        author_id,
        institution_id, 
        publication_year,
        referenced_works,
        
        -- Source weight: 1.0 total per work, split across rows
        1.0 / COUNT(*) OVER (PARTITION BY work_id) AS source_weight,
        
        -- Author weight: 1.0/(authors per work) total per author, split across their institution rows
        (1.0 / COUNT(DISTINCT author_id) OVER (PARTITION BY work_id)) 
        / COUNT(*) OVER (PARTITION BY work_id, author_id) AS author_weight,
        
        -- Institution weight: Same as author_weight in this formulation
        -- Because author's share is already split among institutions
        (1.0 / COUNT(DISTINCT author_id) OVER (PARTITION BY work_id)) 
        / COUNT(*) OVER (PARTITION BY work_id, author_id) AS institution_weight   
    FROM wsai;

SELECT * FROM work_items LIMIT 4;
SHOW TABLES;
"
"-- 3 ASSEMBLE THE unprojected TABLE using the REFERENCE table
-- =========================================================
CREATE OR REPLACE TEMP TABLE unprojected AS
WITH refs AS (
    SELECT 
        DISTINCT work_id AS citer_work, 
        -- User's specific parsing logic
        r.reference AS cited_work
    FROM work_items 
    LEFT JOIN LATERAL UNNEST(referenced_works) AS r(reference) ON true
    -- Filter: keep only valid cited_idx that exist in our dataset
    WHERE r.reference IN (SELECT work_id FROM work_items)
  ),
  unprojected AS
    (SELECT 
        citer_work, 
        i1.source_id AS citer_source, i1.author_id AS citer_author, i1.institution_id AS citer_institution, 
        i1.source_weight AS citer_source_weight, i1.author_weight AS citer_author_weight, i1.institution_weight AS citer_institution_weight, 
        
        cited_work, 
        i2.source_id AS cited_source, i2.author_id AS cited_author, i2.institution_id AS cited_institution, 
        i2.source_weight AS cited_source_weight, i2.author_weight AS cited_author_weight, i2.institution_weight AS cited_institution_weight

      FROM refs 
      -- INNER JOIN is safe because we know work_items is clean now
      INNER JOIN work_items i1 ON i1.work_id = citer_work
      INNER JOIN work_items i2 ON i2.work_id = cited_work
      WHERE i2.publication_year = 2020
    )
  SELECT * FROM unprojected;

SELECT * FROM unprojected LIMIT 4;
SHOW TABLES;"
"-- 4 BUILD THE edge_list tables 
-- Includes mono-modal (s, a, i) and the combined (si. sa and ai) super-matrices
-- ===========================================================================

CREATE OR REPLACE TEMP TABLE edge_list AS
  SELECT 
      v.projection_type,
      v.source_id AS citer,
      v.target_id AS cited,
      SUM(v.weight) as weight
  FROM unprojected u
  CROSS JOIN LATERAL (
      VALUES 
          -- =========================================================
          -- 1. MONO-MODAL MATRICES (Keep weight = 1.0)
          -- Total system weight = Total Citations
          -- =========================================================
          ('s', u.citer_source,      u.cited_source,      (u.citer_source_weight * u.cited_source_weight)),
          ('a', u.citer_author,      u.cited_author,      (u.citer_author_weight * u.cited_author_weight)),
          ('i', u.citer_institution, u.cited_institution, (u.citer_institution_weight * u.cited_institution_weight)),

          -- =========================================================
          -- 2. COMPOSITE BLOCK MATRICES (Scale weight by 0.25)
          -- Total system weight = Total Citations (Conserved)
          -- =========================================================
          
          -- SI (Source + Institution)
          ('si', u.citer_source,      u.cited_source,      (u.citer_source_weight * u.cited_source_weight) * 0.25),
          ('si', u.citer_source,      u.cited_institution, (u.citer_source_weight * u.cited_institution_weight) * 0.25),
          ('si', u.citer_institution, u.cited_source,      (u.citer_institution_weight * u.cited_source_weight) * 0.25),
          ('si', u.citer_institution, u.cited_institution, (u.citer_institution_weight * u.cited_institution_weight) * 0.25),

          -- SA (Source + Author)
          ('sa', u.citer_source,      u.cited_source,      (u.citer_source_weight * u.cited_source_weight) * 0.25),
          ('sa', u.citer_source,      u.cited_author,      (u.citer_source_weight * u.cited_author_weight) * 0.25),
          ('sa', u.citer_author,      u.cited_source,      (u.citer_author_weight * u.cited_source_weight) * 0.25),
          ('sa', u.citer_author,      u.cited_author,      (u.citer_author_weight * u.cited_author_weight) * 0.25),

          -- AI (Author + Institution)
          ('ai', u.citer_author,      u.cited_author,      (u.citer_author_weight * u.cited_author_weight) * 0.25),
          ('ai', u.citer_author,      u.cited_institution, (u.citer_author_weight * u.cited_institution_weight) * 0.25),
          ('ai', u.citer_institution, u.cited_author,      (u.citer_institution_weight * u.cited_author_weight) * 0.25),
          ('ai', u.citer_institution, u.cited_institution, (u.citer_institution_weight * u.cited_institution_weight) * 0.25)

      ) AS v(projection_type, source_id, target_id, weight)
      
  WHERE v.source_id IS NOT NULL 
    AND v.target_id IS NOT NULL
  GROUP BY 1, 2, 3;
 
  COPY (SELECT * FROM edge_list) TO '/home/lc/m/working/econ_bus_edge_lists.parquet' (FORMAT PARQUET);"
"-- x TRANSFER THIS NOTEBOOK TO AN .sql TEXT FILE FOR INGESTION INTO PYTHON
-- =====================================================================
-- SELECT * FROM duckdb_tables();
SELECT * FROM _duckdb_ui.current_notebook_id;
-- SELECT * FROM _duckdb_ui.notebooks WHERE id = 'aca8f7a9-996e-4767-b5eb-02266d707437';
-- SELECT json FROM _duckdb_ui.notebook_versions WHERE notebook_id = 'aca8f7a9-996e-4767-b5eb-02266d707437' ORDER BY version DESC LIMIT 2

COPY (
    SELECT q.sql_line
    FROM (
        SELECT CAST(json AS JSON) AS notebook_json
        FROM _duckdb_ui.notebook_versions 
        WHERE notebook_id = 'aca8f7a9-996e-4767-b5eb-02266d707437' 
        ORDER BY version DESC 
        LIMIT 1
    ) n,
    LATERAL unnest(
        json_transform(
            json_extract(n.notebook_json, '$.cells'), 
            '[""JSON""]'
        )
    ) AS c(cell),
    LATERAL (VALUES (
        json_extract_string(c.cell, '$.query'),
        json_extract(c.cell, '$.cellId')::INTEGER
    )) AS q(sql_line, cell_id)
    WHERE q.sql_line IS NOT NULL
    ORDER BY q.cell_id
) TO '/home/lc/Projects/EconomicsBusiness/.sql/notebook.sql' (FORMAT csv, HEADER false, OVERWRITE_OR_IGNORE);

SELECT * FROM read_csv('/home/lc/Projects/EconomicsBusiness/.sql/notebook.sql') LIMIT 4"
"-- x Validator for Scaled (0.25) Matrices
-- ======================================
WITH matrix_stats AS (
    SELECT 
        projection_type, 
        SUM(weight) as total_weight
    FROM edge_list
    GROUP BY 1
),
baseline AS (
    -- Use 's' as the baseline unit
    SELECT total_weight as base_val 
    FROM matrix_stats 
    WHERE projection_type = 's'
)
SELECT 
    m.projection_type,
    CAST(m.total_weight AS BIGINT) AS total_weight,
    ROUND(m.total_weight / b.base_val, 2) AS ratio_to_source,
    
    -- We now expect ALL ratios to be 1.0 because of the 0.25 scaling
    1.0 AS expected_ratio,

    CASE 
        WHEN ROUND(m.total_weight / b.base_val, 2) = 1.00 THEN 'PASS' 
        ELSE 'FAIL' 
    END AS status
FROM matrix_stats m, baseline b
ORDER BY length(m.projection_type), m.projection_type;"
"-- x BUILD supporting tables - works counts author filters, cross-matching names
-- =============================================================================
-- SELECT * FROM read_csv('/home/lc/Projects/EconomicsBusiness/2026_study/DATA/econ_bus_journal_oa.csv')
SELECT DISTINCT 's' AS kind, source_id, source_name, work_count
  FROM wsai
  JOIN (SELECT source_id, sum(source_weight)::INT AS work_count
          FROM work_items GROUP BY ALL)
  USING (source_id)
  ORDER BY work_count DESC

-- ATTACH IF NOT EXISTS '/home/lc/m/working/econ_backup.duckdb' AS project;
-- WITH
--     candidates_CTE AS
--     (SELECT author_id,
--             author_name,
--             Research_Profile,
--             ""Group"",
--             ""Class"",
--         FROM project.candidates
--         WHERE list_contains(['endogenous', 'exogenous', 'matched_difficult', 'openalex'], kind)
--     ),
--     works_authors_CTE AS
--     (SELECT DISTINCT work_id,
--             author_id,
--             author_name,
--             Research_Profile,
--             ""Group"",
--             ""Class"",
--         FROM candidates_CTE
--         LEFT JOIN project.authorships
--         USING (author_id)
--     )

-- SELECT DISTINCT * EXCLUDE (work_id) FROM works_authors_CTE;

-- SELECT DISTINCT 'a' AS kind, author_id AS oa_id, author_name AS oa_name, work_count, ""Group""
--   FROM wsai
--   JOIN (SELECT author_id, sum(author_weight)::INT AS work_count
--           FROM work_items GROUP BY ALL)
--   USING (author_id)
--   LEFT JOIN read_csv('/home/lc/Projects/EconomicsBusiness/2026_study/DATA/matched_authors.csv')
--   ON author_id = id
--   WHERE ""Group"" NOT NULL
--   ORDER BY work_count DESC
"
SELECT DISTINCT source_id FROM works
"SELECT DISTINCT column0 AS source_id_IN, j.source_id, j.source_name, count(id) AS works_count
  FROM read_csv('/home/lc/Projects/EconomicsBusiness/2026_study/sources.txt', header=false) s
  LEFT JOIN
    (SELECT  DISTINCT primary_location.source.id AS source_id,  
          primary_location.source.display_name AS source_name,
          id
    FROM read_json_auto('/home/lc/m/openalex_feb26/json/**/*.json', ignore_errors = true)) j
  ON column0 = j.source_id
  GROUP BY ALL
  ORDER BY works_count DESC"
