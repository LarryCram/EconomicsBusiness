SELECT DISTINCT ON (s.id) 
                s.id AS source_id, 
                s.display_name AS source_name,
                works_count, cited_by_count,
                s.issn_l AS issn, j.era_journal_name, j.harzing_journal_name, j.wos_journal_name
    FROM '/home/lc/m/working/econ_bus/parquet/comprehensive_journal_list.parquet' j
    LEFT JOIN '/home/lc/m/openalex_feb26/parquet/sources.parquet' s
    ON list_has_any(j.unique_issn_list, s.issn)
    AND j.unique_issn_list IS NOT NULL AND s.issn IS NOT NULL;