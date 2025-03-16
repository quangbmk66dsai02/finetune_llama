import pandas as pd

# Replace 'your_file.parquet' with the path to your Parquet file
df = pd.read_parquet('vi-alpaca.parquet')
print(df.head())
df.to_json('output.json', orient='records', lines=True, force_ascii=False)
