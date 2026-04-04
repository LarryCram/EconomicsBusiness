COPY (
    WITH scores AS (
        SELECT
            *,
            (CASE WHEN regexp_matches(LOWER(era_field),       'econom|financ|banking') THEN 1 ELSE 0 END
               +  CASE WHEN regexp_matches(LOWER(harzing_field),  'econom|financ|banking') THEN 1 ELSE 0 END
               +  CASE WHEN regexp_matches(LOWER(wos_categories), 'econom|financ|banking') THEN 1 ELSE 0 END
               +  CASE WHEN regexp_matches(LOWER(field_name),     'econom|financ|banking') THEN 1 ELSE 0 END
            ) AS econ_score,
            (CASE WHEN regexp_matches(LOWER(era_field),       'business|commerc|management|tourism|transport') THEN 1 ELSE 0 END
               +  CASE WHEN regexp_matches(LOWER(harzing_field),  'business|commerc|management|tourism|transport') THEN 1 ELSE 0 END
               +  CASE WHEN regexp_matches(LOWER(wos_categories), 'business|commerc|management|tourism|transport') THEN 1 ELSE 0 END
               +  CASE WHEN regexp_matches(LOWER(field_name),     'business|commerc|management|tourism|transport') THEN 1 ELSE 0 END
            ) AS bus_score
        FROM 'data/source_master.csv'
    )
    SELECT *,
        CASE
            WHEN econ_score >= 2 AND bus_score  < 1 THEN 'E'
            WHEN bus_score  >= 2 AND econ_score < 1 THEN 'B'
            WHEN bus_score <= 1 and econ_score <= 1 THEN NULL
            ELSE 'A'
        END AS field_eb
    FROM scores
) TO 'data/source_master_new.csv' (FORMAT CSV, HEADER true);
