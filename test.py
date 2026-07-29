import pandas as pd

df = pd.read_parquet("data/processed/model_table.parquet")
print(df.shape)
print(df.head())
print(df["high_pollution_next_hour"].value_counts(normalize=True))