import pandas as pd 

# df = pd.read_sas("./data/raw/GATS/GATS_Greece_National_2013_SAS/GREECE_PUBLIC_USE_11Mar2015.sas7bdat")

# df.to_csv("./data/processed/GATS_Greece_National_2013_SAS.csv")
# print(df.head())

import pandas_access as mdb

# Listing the tables.
for tbl in mdb.list_tables("./data/raw/"):
print(tbl)

# Read a small table.
df = pandas_access.read_table("my.mdb", "MyTable")