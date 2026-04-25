from pathlib import Path
import duckdb
import yaml
import pandas as pd

# Load configuration relative to project root
config_path = Path('./config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)
    PROJECT_FOLDER = Path(config['PROJECT_ROOT'])
    DATA = PROJECT_FOLDER / Path(config.get('DATA'))
    WORKING = Path(config.get('WORKING'))
    OPENALEX = Path(config.get('OPENALEX'))

PARQUET = WORKING / 'parquet'


def filter_and_match(db):
    sql = f"""
        -- 1. CROSS MATCH journals by ISSN
        -- ================================================================
        -- For registry rows that match exactly one OA source: keep it.
        -- For registry rows that match multiple OA sources (multi-match):
        --   - keep the first name-matching OA source (case-insensitive equality
        --     against any of era/harzing/wos name), ranked by works_count DESC.
        --   - if no OA source matches by name: drop all (treat as unmatched).
        -- After filtering to at most one OA source per registry row, apply
        -- DISTINCT ON (s.id) to handle the reverse case (one OA source matching
        -- multiple registry rows).
        CREATE OR REPLACE TEMP TABLE source_list AS
        WITH all_matches AS (
            SELECT
                j.unique_issn_list,
                j.era_journal_name, j.era_for_codes,
                j.harzing_journal_name, j.harzing_field,
                j.wos_journal_name, j.wos_categories,
                s.id AS oa_id,
                s.issn_l AS issn,
                s.display_name AS source_name,
                s.works_count, s.cited_by_count,
                (   (j.era_journal_name     IS NOT NULL AND LOWER(s.display_name) = LOWER(j.era_journal_name))
                 OR (j.harzing_journal_name IS NOT NULL AND LOWER(s.display_name) = LOWER(j.harzing_journal_name))
                 OR (j.wos_journal_name     IS NOT NULL AND LOWER(s.display_name) = LOWER(j.wos_journal_name))
                ) AS name_match
            FROM '{PARQUET}/comprehensive_journal_list.parquet' j
            JOIN '{OPENALEX}/sources.parquet' s
                ON list_has_any(j.unique_issn_list, s.issn)
               AND j.unique_issn_list IS NOT NULL
               AND s.issn IS NOT NULL
               AND s.type = 'journal'
        ),
        per_reg AS (
            SELECT unique_issn_list,
                   COUNT(*)        AS n_oa,
                   BOOL_OR(name_match) AS any_name_match
            FROM all_matches
            GROUP BY unique_issn_list
        ),
        filtered AS (
            SELECT m.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY m.unique_issn_list
                       ORDER BY m.name_match DESC, m.works_count DESC
                   ) AS rk_reg
            FROM all_matches m
            JOIN per_reg p USING (unique_issn_list)
            WHERE p.n_oa = 1                                           -- single match: always keep
               OR (p.n_oa > 1 AND p.any_name_match AND m.name_match)  -- multi: name-matched only
            -- multi + no name match: dropped (p.any_name_match = FALSE)
        )
        SELECT DISTINCT ON (oa_id)
            CAST(regexp_replace(oa_id, 'https://openalex.org/S', '') AS BIGINT) AS source_idx,
            source_name, works_count, cited_by_count, issn,
            era_journal_name, era_for_codes,
            harzing_journal_name, harzing_field,
            wos_journal_name, wos_categories
        FROM filtered
        WHERE rk_reg = 1
        ORDER BY oa_id, works_count DESC;

        -- 1b. Exact title matches for registry journals not matched by ISSN
        CREATE OR REPLACE TEMP TABLE title_matches AS
            WITH unmatched AS (
                SELECT DISTINCT
                    COALESCE(cjl.era_journal_name, cjl.wos_journal_name, cjl.harzing_journal_name) AS ref_name,
                    cjl.era_journal_name, cjl.era_for_codes,
                    cjl.harzing_journal_name, cjl.harzing_field,
                    cjl.wos_journal_name, cjl.wos_categories
                FROM '{PARQUET}/comprehensive_journal_list.parquet' cjl
                WHERE NOT EXISTS (
                    SELECT 1 FROM source_list sl
                    WHERE (cjl.era_journal_name    IS NOT NULL AND sl.era_journal_name    = cjl.era_journal_name)
                       OR (cjl.harzing_journal_name IS NOT NULL AND sl.harzing_journal_name = cjl.harzing_journal_name)
                       OR (cjl.wos_journal_name    IS NOT NULL AND sl.wos_journal_name    = cjl.wos_journal_name)
                )
                AND COALESCE(cjl.era_journal_name, cjl.wos_journal_name, cjl.harzing_journal_name) IS NOT NULL
            )
            SELECT DISTINCT ON (s.id)
                CAST(regexp_replace(s.id, 'https://openalex.org/S', '') AS BIGINT) AS source_idx,
                s.display_name AS source_name,
                s.works_count, s.cited_by_count, s.issn_l AS issn,
                u.era_journal_name, u.era_for_codes,
                u.harzing_journal_name, u.harzing_field,
                u.wos_journal_name, u.wos_categories
            FROM unmatched u
            JOIN '{OPENALEX}/sources.parquet' s
                ON LOWER(s.display_name) = LOWER(u.ref_name)
               AND s.type = 'journal';

        INSERT INTO source_list
            SELECT * FROM title_matches
            WHERE source_idx NOT IN (SELECT source_idx FROM source_list);

        -- Save OAS* (ISSN matches + exact title matches) for table reporting
        COPY (SELECT * FROM source_list) TO '{PARQUET}/oas_star.parquet' (FORMAT PARQUET);

        -- 2. Get ALL topics for matched sources
        -- ============================================================
        CREATE OR REPLACE TEMP TABLE source_topics AS
            SELECT s.source_idx, s.source_name, oa.works_count, oa.cited_by_count,
                    s.issn,
                    s.era_journal_name, s.era_for_codes,
                    s.harzing_journal_name, s.harzing_field,
                    s.wos_journal_name, s.wos_categories,
                unnest(oa.topics).count AS topic_count,
                unnest(oa.topics).subfield.display_name AS subfield_name,
                unnest(oa.topics).field.display_name AS field_name
            FROM source_list s
            LEFT JOIN '{OPENALEX}/sources.parquet' oa
                ON s.source_idx = CAST(regexp_replace(oa.id, 'https://openalex.org/S', '') AS BIGINT);

        -- 3. Source topic densities (all OAS*, saved for reporting/diagnostics)
        -- ====================================================================
        -- Density = SUM(econ/bus topic counts) / SUM(all topic counts)
        -- NULL density means no topic assignments.
        CREATE OR REPLACE TEMP TABLE source_densities AS
            SELECT source_idx,
                COALESCE(SUM(topic_count) FILTER (WHERE field_name IN (
                    'Economics, Econometrics and Finance',
                    'Business, Management and Accounting'
                )), 0) * 1.0 / NULLIF(SUM(topic_count), 0) AS econ_bus_density
            FROM source_topics
            GROUP BY source_idx;

        COPY (SELECT * FROM source_densities) TO '{PARQUET}/source_densities.parquet' (FORMAT PARQUET);

        -- 4. Final: top topic per source, with density and E/B/A/X field label
        -- =============================================================================
        -- field_eb scoring: ERA uses count of FoR codes (35xx=bus, 38xx=econ);
        --   harzing_field uses token lookup (split on ', '; E={'Economics','F&A'}, B=all others except extraneous);
        --   wos_categories and field_name each contribute 0/1 by keyword.
        --   'E'  econ_score >= 2 AND bus_score < 2   (economics-dominant)
        --   'B'  bus_score  >= 2 AND econ_score < 2  (business-dominant)
        --   'A'  econ_score >= 2 AND bus_score >= 2  (genuinely ambiguous: strong in both)
        --   'X'  econ_score < 2  AND bus_score < 2   (weak signals in both)
        CREATE OR REPLACE TEMP TABLE source_master AS
        WITH ranked_topics AS (
            SELECT
                st.source_idx, st.source_name, st.issn,
                st.era_journal_name, st.era_for_codes,
                st.harzing_journal_name, st.harzing_field,
                st.works_count, st.cited_by_count,
                st.wos_journal_name, st.wos_categories,
                st.topic_count, st.subfield_name, st.field_name,
                sd.econ_bus_density,
                ROW_NUMBER() OVER (PARTITION BY st.source_idx ORDER BY st.topic_count DESC, st.subfield_name ASC) as rn
            FROM source_topics st
            LEFT JOIN source_densities sd ON st.source_idx = sd.source_idx
        ),
        top_topic AS (SELECT * EXCLUDE rn FROM ranked_topics WHERE rn = 1),
        scored AS (
            SELECT *,
                (COALESCE(len(list_filter(era_for_codes, x -> LEFT(x, 2) = '38')), 0)
                   +  CASE WHEN len(list_filter(string_split(COALESCE(harzing_field, ''), ', '),
                                x -> x IN ('Economics', 'F&A'))) > 0 THEN 1 ELSE 0 END
                   +  CASE WHEN regexp_matches(LOWER(array_to_string(wos_categories, ' ')), 'econom|financ|banking') THEN 1 ELSE 0 END
                   +  CASE WHEN regexp_matches(LOWER(field_name),                       'econom|financ|banking') THEN 1 ELSE 0 END
                ) AS econ_score,
                (COALESCE(len(list_filter(era_for_codes, x -> LEFT(x, 2) = '35')), 0)
                   +  CASE WHEN len(list_filter(string_split(COALESCE(harzing_field, ''), ', '),
                                x -> x IN ('Bus Hist','Comm','Entrep','Gen & Strat','IB','Innovation',
                                           'Marketing','MIS,KM','OS/OB,HRM/IR','OR,MS,POM','PSM','Tourism'))) > 0 THEN 1 ELSE 0 END
                   +  CASE WHEN regexp_matches(LOWER(array_to_string(wos_categories, ' ')), 'business|commerc|management|tourism|transport') THEN 1 ELSE 0 END
                   +  CASE WHEN regexp_matches(LOWER(field_name),                       'business|commerc|management|tourism|transport') THEN 1 ELSE 0 END
                ) AS bus_score
            FROM top_topic
        )
        SELECT *,
            CASE
                WHEN econ_score >= 2 AND bus_score  >= 2 THEN 'A'
                WHEN econ_score >= 2                     THEN 'E'
                WHEN bus_score  >= 2                     THEN 'B'
                ELSE 'X'
            END AS field_eb
        FROM scored;

        COPY (
            SELECT * FROM source_master
            WHERE econ_bus_density >= 0.4 OR econ_bus_density IS NULL
        ) TO '{PARQUET}/source_master.parquet' (FORMAT PARQUET);
        """
    db.sql(sql)
    sql = f"""
        SELECT *
        FROM '{PARQUET}/source_master.parquet'
        ORDER BY works_count DESC
        """
    db.sql(sql).show()
    db.sql(sql).df().to_csv(DATA / 'source_master.csv')

    # 5. Unmatched registry journals: find those not in OAS* and fuzzy-match titles
    # =============================================================================
    db.sql(f"""
        CREATE OR REPLACE TEMP TABLE unmatched AS
        SELECT
            COALESCE(cjl.era_journal_name, cjl.wos_journal_name, cjl.harzing_journal_name) AS ref_name,
            cjl.era_journal_name, cjl.era_for_codes,
            cjl.harzing_journal_name, cjl.harzing_field,
            cjl.wos_journal_name, cjl.wos_categories
        FROM '{PARQUET}/comprehensive_journal_list.parquet' cjl
        WHERE NOT EXISTS (
            SELECT 1 FROM '{PARQUET}/oas_star.parquet' o
            WHERE list_contains(cjl.unique_issn_list, o.issn)
        )
        AND cjl.unique_issn_list IS NOT NULL
        AND COALESCE(cjl.era_journal_name, cjl.wos_journal_name, cjl.harzing_journal_name) IS NOT NULL
    """)
    print(f"Unmatched registry journals: {db.sql('SELECT COUNT(*) FROM unmatched').fetchone()[0]}")

    db.sql(f"""
        COPY (
            WITH candidates AS (
                SELECT u.ref_name,
                       u.era_journal_name, u.era_for_codes,
                       u.harzing_journal_name, u.harzing_field,
                       u.wos_journal_name, u.wos_categories,
                       CAST(regexp_replace(oa.id, 'https://openalex.org/S', '') AS BIGINT) AS candidate_source_idx,
                       oa.display_name AS candidate_name,
                       jaro_winkler_similarity(LOWER(u.ref_name), LOWER(oa.display_name)) AS similarity
                FROM unmatched u
                CROSS JOIN '{OPENALEX}/sources.parquet' oa
                WHERE oa.type = 'journal' AND oa.works_count > 100
            ),
            best AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY ref_name ORDER BY similarity DESC) AS rn
                FROM candidates
            )
            SELECT ref_name,
                   era_journal_name, era_for_codes,
                   harzing_journal_name, harzing_field,
                   wos_journal_name, wos_categories,
                   candidate_source_idx, candidate_name, ROUND(similarity, 3) AS similarity,
                   similarity = 1.0 AS likely_title_match
            FROM best WHERE rn = 1
        ) TO '{PARQUET}/unmatched_journals.parquet' (FORMAT PARQUET)
    """)
    print("Unmatched journals with title match candidates saved.")
    return

def main():
    with duckdb.connect() as db:
        filter_and_match(db)

if __name__ == "__main__":
    main()
    print("FINISHED !")
