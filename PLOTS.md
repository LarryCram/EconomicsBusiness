# PLOTS.md — EconomicsBusiness Project

## Project root
`/home/lc/Projects/EconomicsBusiness` — VS Code workspace, synced to GitHub.

## Plotting
- prefer seaborn OO version.
- use mathplotlib where seaborn is incapable.
- make a plot in VS Code and persist to disk as a latex-compatible format

## Plot 1
- Prepare an long-tail elbow plot I can use to determine the institution work_count cut off.
- Perplexity suggests the following duckdb SQL for an abstact work-institution table. 
- Apply an SQL like this to our table and make the elbow plot

### draft SQL
WITH institution_works AS (
    -- Step 1: Works per institution
    SELECT 
        institution_id, 
        COUNT(DISTINCT work_id) AS works_count
    FROM works_institutions 
    GROUP BY institution_id
),
work_count_stats AS (
    -- Step 2: Frequency distribution (institutions per works_count)
    SELECT 
        works_count, 
        COUNT(*) AS institutions_count
    FROM institution_works
    GROUP BY works_count
)
SELECT 
    works_count,
    institutions_count,
    -- Cumulative institutions up to this works_count
    SUM(institutions_count) OVER (ORDER BY works_count ROWS UNBOUNDED PRECEDING) AS cum_institutions,
    -- Cumulative works up to this works_count  
    SUM(institutions_count * works_count) OVER (ORDER BY works_count ROWS UNBOUNDED PRECEDING) AS cum_works,
    -- % of total works retained
    SUM(institutions_count * works_count) OVER (ORDER BY works_count ROWS UNBOUNDED PRECEDING) * 100.0 / 
    SUM(institutions_count * works_count) OVER () AS pct_works_retained
FROM work_count_stats
ORDER BY works_count;

