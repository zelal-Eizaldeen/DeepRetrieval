import pandas as pd

# path = '/home/azureuser/cloudfiles/code/DeepRetrieval/code/data/sql/bird/train.parquet'
path = '/home/azureuser/cloudfiles/code/DeepRetrieval/code/data/sql/bird/test.parquet'
df = pd.read_parquet(path)
print(f"Loaded {len(df)} rows")
# print(df.head(3))

# for i in range(len(df)):
#     print(df.iloc[i]['prompt'])
#     print('--------------------------------')
#     break
