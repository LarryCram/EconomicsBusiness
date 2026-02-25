from pathlib import Path
import duckdb

def filter_and_match(db):
    sql = """
        -- FILTER OA source list by ON subfield USING oa_topic_df.csv then MATCH to journal list on ISSN
        -- ==============================================================================================
        COPY (
        WITH
        source_topics AS
            (SELECT issn, source_id, display_name AS source_name, works_count,
                    unnest.subfield.display_name AS subfield_name, 
                    unnest.field.display_name AS field_name
            FROM read_parquet('/home/lc/m/working/econ_bus/parquet/sources.parquet')
            CROSS JOIN UNNEST(topics) AS topic
            ),
        filtered_source_topics AS
            (SELECT *
            FROM source_topics
            LEFT JOIN read_csv('/home/lc/Projects/EconomicsBusiness/data/source_masters/oa_topic_df.csv')
            USING (subfield_name)
            WHERE keep = 1
            )

        SELECT DISTINCT ISSN, source_id, works_count, list(subfield_name) AS subfield_list, source_name, era_journal_name, harzing_journal_name, wos_journal_name
        FROM filtered_source_topics
        JOIN (SELECT *
                FROM read_parquet('/home/lc/Projects/EconomicsBusiness/data/comprehensive_journal_list.parquet'))
        ON list_has_any(unique_issn_list, issn) = true
        GROUP BY ALL
        ORDER BY source_id
        ) TO '/home/lc/Projects/EconomicsBusiness/data/journal_list_matched.parquet' (FORMAT PARQUET); 
        """
    db.sql(sql)
    sql = """        SELECT count(source_id) AS source_count, subfield
        FROM (SELECT source_id, unnest(subfield_list) AS subfield
                FROM read_parquet('/home/lc/Projects/EconomicsBusiness/data/journal_list_matched.parquet'))
        GROUP BY ALL
        ORDER BY source_count DESC;
        """
    db.sql(sql).show()
    return

def main():
    with duckdb.connect() as db:
        filter_and_match(db)

if __name__ == "__main__":
    main()
    print("FINISHED !")