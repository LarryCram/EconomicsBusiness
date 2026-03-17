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

        -- Save OAS* (long-list before topic filtering) for table reporting
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

        -- 3. Sources whose TOP subfield is approved
        -- =========================================
        CREATE OR REPLACE TEMP TABLE approved_sources AS
            SELECT DISTINCT source_id
            FROM (
                SELECT source_id, subfield_name,
                    ROW_NUMBER() OVER (PARTITION BY source_id ORDER BY topic_count DESC, subfield_name ASC) as rn
                FROM source_topics
            ) ranked
            WHERE rn = 1
            AND subfield_name IN (
                'Economics and Econometrics','Accounting','General Economics, Econometrics and Finance',
                'Strategy and Management','Organizational Behavior and Human Resource Management',
                'Management Information Systems','Finance','Information Systems and Management',
                'Marketing','Management of Technology and Innovation','Management Science and Operations Research',
                'General Decision Sciences', 'Tourism, Leisure and Hospitality Management',
                'Business and International Management'
            );

        -- 4. Final: main topic for approved sources only
        -- ==============================================
        CREATE OR REPLACE TEMP TABLE source_master AS
        WITH ranked_topics AS (
        SELECT
            source_id, source_name, issn, era_journal_name, harzing_journal_name,
            works_count, cited_by_count,
            wos_journal_name, topic_count, subfield_name, field_name,
            ROW_NUMBER() OVER (PARTITION BY source_id ORDER BY topic_count DESC, subfield_name ASC) as rn
        FROM source_topics
        WHERE source_id IN (SELECT source_id FROM approved_sources)
        )

        SELECT * FROM ranked_topics WHERE rn = 1;

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
    return

def main():
    with duckdb.connect() as db:
        filter_and_match(db)

if __name__ == "__main__":
    main()
    print("FINISHED !")
