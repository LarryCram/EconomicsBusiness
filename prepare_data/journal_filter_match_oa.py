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
        CREATE OR REPLACE TEMP TABLE source_list AS
            SELECT DISTINCT ON (s.id) s.id AS source_id, s.display_name AS source_name,
                                works_count, cited_by_count,
                                s.issn_l AS issn, j.era_journal_name, j.harzing_journal_name, j.wos_journal_name
            FROM '{PARQUET}/comprehensive_journal_list.parquet' j
            LEFT JOIN '{OPENALEX}/sources.parquet' s
            ON list_has_any(j.unique_issn_list, s.issn)
            AND j.unique_issn_list IS NOT NULL AND s.issn IS NOT NULL;

        -- 1b. Exact title matches for registry journals not matched by ISSN
        CREATE OR REPLACE TEMP TABLE title_matches AS
            WITH unmatched AS (
                SELECT DISTINCT
                    COALESCE(cjl.era_journal_name, cjl.wos_journal_name, cjl.harzing_journal_name) AS ref_name,
                    cjl.era_journal_name, cjl.harzing_journal_name, cjl.wos_journal_name
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
                s.id AS source_id, s.display_name AS source_name,
                s.works_count, s.cited_by_count, s.issn_l AS issn,
                u.era_journal_name, u.harzing_journal_name, u.wos_journal_name
            FROM unmatched u
            JOIN '{OPENALEX}/sources.parquet' s
                ON LOWER(s.display_name) = LOWER(u.ref_name);

        INSERT INTO source_list
            SELECT * FROM title_matches
            WHERE source_id NOT IN (SELECT source_id FROM source_list);

        -- Save OAS* (ISSN matches + exact title matches) for table reporting
        COPY (SELECT * FROM source_list) TO '{PARQUET}/oas_star.parquet' (FORMAT PARQUET);

        -- 2. Get ALL topics for matched sources
        -- ============================================================
        CREATE OR REPLACE TEMP TABLE source_topics AS
            SELECT s.source_id, s.source_name, oa.works_count, oa.cited_by_count,
                    s.issn, s.era_journal_name, s.harzing_journal_name, s.wos_journal_name,
                unnest(oa.topics).count AS topic_count,
                unnest(oa.topics).subfield.display_name AS subfield_name,
                unnest(oa.topics).field.display_name AS field_name
            FROM source_list s
            LEFT JOIN '{OPENALEX}/sources.parquet' oa ON s.source_id = oa.id;

        -- 3. Source topic densities (all OAS*, saved for reporting/diagnostics)
        -- ====================================================================
        -- Density = SUM(econ/bus topic counts) / SUM(all topic counts)
        -- NULL density means no topic assignments; those sources are kept.
        CREATE OR REPLACE TEMP TABLE source_densities AS
            SELECT source_id,
                COALESCE(SUM(topic_count) FILTER (WHERE field_name IN (
                    'Economics, Econometrics and Finance',
                    'Business, Management and Accounting'
                )), 0) * 1.0 / NULLIF(SUM(topic_count), 0) AS econ_bus_density
            FROM source_topics
            GROUP BY source_id;

        COPY (SELECT * FROM source_densities) TO '{PARQUET}/source_densities.parquet' (FORMAT PARQUET);

        CREATE OR REPLACE TEMP TABLE approved_sources AS
            SELECT source_id, econ_bus_density
            FROM source_densities
            WHERE econ_bus_density >= 0.4 OR econ_bus_density IS NULL;

        -- 3c. Dropped sources: top topic for sources excluded by density filter
        COPY (
            WITH ranked AS (
                SELECT st.source_id, st.source_name, st.works_count,
                       st.era_journal_name, st.harzing_journal_name, st.wos_journal_name,
                       sd.econ_bus_density,
                       st.field_name, st.subfield_name, st.topic_count,
                       ROW_NUMBER() OVER (PARTITION BY st.source_id ORDER BY st.topic_count DESC, st.subfield_name ASC) AS rn
                FROM source_topics st
                JOIN source_densities sd ON st.source_id = sd.source_id
                WHERE sd.econ_bus_density < 0.4
            )
            SELECT * EXCLUDE rn FROM ranked WHERE rn = 1
        ) TO '{PARQUET}/dropped_sources.parquet' (FORMAT PARQUET);

        -- 4. Final: top topic per approved source, with density
        -- ======================================================
        CREATE OR REPLACE TEMP TABLE source_master AS
        WITH ranked_topics AS (
            SELECT
                st.source_id, st.source_name, st.issn, st.era_journal_name, st.harzing_journal_name,
                st.works_count, st.cited_by_count,
                st.wos_journal_name, st.topic_count, st.subfield_name, st.field_name,
                ap.econ_bus_density,
                ROW_NUMBER() OVER (PARTITION BY st.source_id ORDER BY st.topic_count DESC, st.subfield_name ASC) as rn
            FROM source_topics st
            JOIN approved_sources ap ON st.source_id = ap.source_id
        )
        SELECT * EXCLUDE rn FROM ranked_topics WHERE rn = 1;

        COPY (SELECT * FROM source_master) TO '{PARQUET}/source_master.parquet' (FORMAT PARQUET);
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
            cjl.era_journal_name, cjl.harzing_journal_name, cjl.wos_journal_name
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
                SELECT u.ref_name, u.era_journal_name, u.harzing_journal_name, u.wos_journal_name,
                       oa.id AS candidate_source_id,
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
            SELECT ref_name, era_journal_name, harzing_journal_name, wos_journal_name,
                   candidate_source_id, candidate_name, ROUND(similarity, 3) AS similarity,
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
