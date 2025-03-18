import pandas as pd
from tqdm import tqdm

from src.utils.gpt_azure import gpt_chat_4o

dataset = 'bird'
# dataset = 'spider'


parquet_path = f'/home/azureuser/cloudfiles/code/DeepRetrieval/code/data/sql/{dataset}/train.parquet'

df = pd.read_parquet(parquet_path)


data = []
for i in tqdm(range(len(df))):
    row = df.iloc[i]
    data_id = i
    original_prompt = row['prompt'][0]['content']
    gt_sql = row['sql']

    new_prompt = [
        {
            'role': 'system',
            'content': original_prompt.split('<|im_start|>system')[1].split('<|im_end|>')[0].strip()
        },
        {
            'role': 'user',
            'content': original_prompt.split('<|im_start|>user')[1].split('<|im_end|>')[0].strip()
        }
    ]

    D = {
        'id': data_id,
        'original_prompt': original_prompt,
        'gt_sql': gt_sql,
    }
    data.append(D)

    print(original_prompt)
    print('-'*100)
    break



print(gpt_chat_4o("What is the capital of France?"))