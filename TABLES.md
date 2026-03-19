# TABLES.md — EconomicsBusiness Project

## Project root
`/home/lc/Projects/EconomicsBusiness` — VS Code workspace, synced to GitHub.

## Tables
- ultimately provide tables in Latex layout
- Tables 1 2 and 3 relate to data and corpus and are made by the table_make.py in prepare_data


## Table 1
- Name: Model Parameters (done)

## Table 2
- Name: Registry source matches to OpenAlex source identifiers (done)

## Table 3
- Name: Corpus features
- First row of columns blank & 2000 & 2024 & 2000-24 \\
- Second row of columns Quantity & then count under each year of Column 1. The count is over the unique items in the corpus after filtering by works_per_year as defined below and for all works in the column header range.
- Rows are
-- Works
-- Sources
-- Institutions
-- Reference counts (out degree)
-- Citation counts (in degree)
-- \hline
Then a second section of two columns per year column divided into the edge if the upper Q4 (75%) and lower Q1 (25%) quartiles of the following quantities computed per work then analysed. For 2020-24 pool the whole population rather then seggregate by year as an internediate step.
-- count(distinct works)/source
-- count(distinct institutions)/work
-- count(distinct references)/work (out-degree)
-- count(distinct citations)/work (in-degree)
- Construct corpus_institutions.parquet table: construct an institutions table using the institutions in the corpus authorships. Join it to the openalex_feb26 institutions parquet. Add a column of unique works per year groupby institution called works_per_year. Prepare this from the entire corpus 2000-2024.
- Construct the table using a institution filter with works_per_year > 10. This can be hard coded at present but we plan to rerun with 5 and 15. This is explained in main.pdf.
