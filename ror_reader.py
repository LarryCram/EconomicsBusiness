import pandas as pd
from collections import defaultdict
import duckdb

# df = pd.read_json('./DATA/v1.34-2023-10-12-ror-data.json')[['id', 'name', 'aliases', 'country', 'relationships']].set_index(['id', 'name'])[:32]
# print(f'{df.shape = }\n{df.head()}\n{df.info()}')
# df = df.explode('aliases').set_index(['aliases'], append=True).explode('relationships')
# df['country_code'] = [x.get('country_code') for x in df.country]
# df['relation_name'] = [x.get('label') if isinstance(x, dict) else pd.NA for x in df.relationships]
# df = df.drop(columns=['country', 'relationships']).reset_index(['name', 'aliases'])

df = pd.read_json('./DATA/v1.34-2023-10-12-ror-data.json')[['id', 'name', 'aliases', 'country']].set_index(['id', 'name'])[:32]
print(f'{df.shape = }\n{df.head()}\n{df.info()}')
df = df.explode('aliases').set_index(['aliases'], append=True)
df['country_code'] = [x.get('country_code') for x in df.country]
df = df.drop(columns=['country']).reset_index(['name', 'aliases'])

print(df)
dd = defaultdict(list)
for row in df.itertuples():
    dd[row.Index].append(row.name)
    dd[row.Index].append(row.aliases)
    # dd[row.Index].append(row.relation_name)
for k, v in dd.items():
    dd[k] = list({x for x in v if isinstance(x, str)})

for k, v in dd.items():
    print(k, v)

d1 = {v1: {'institution_id': k} for k, v in dd.items() for v1 in v}
df = pd.DataFrame.from_dict(d1, orient='index').reset_index().rename(columns={'index': 'name'})
print(f'{df.shape = }\n{df.head()}')
db = duckdb.connect('//home/lc/m/openalex_june25/institutions.duckdb')
db.sql("SELECT * FROM institutions.institutions").show()
db.sql("CREATE OR REPLACE TABLE institutions.ror AS SELECT * FROM df")
db.sql("SELECT * FROM institutions.ror").show()
db.close()
