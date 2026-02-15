import duckdb
import glob

# 1. Find all xlsx files
files = glob.glob('data/*.xlsx')

# 2. Create the table from the first file
duckdb.sql(f"CREATE TABLE all_data AS SELECT * FROM read_xlsx('{files[0]}')")

# 3. Insert the rest
for file in files[1:]:
    duckdb.sql(f"INSERT INTO all_data SELECT * FROM read_xlsx('{file}')")

# Verify
duckdb.sql("SELECT count(*) FROM all_data").show()