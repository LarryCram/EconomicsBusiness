import pandas as pd
import pyarrow

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 144)

pd.set_option("mode.copy_on_write", True)
# pd.options.future.infer_string = True

def pandas_setup():
    return

