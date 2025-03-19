import json
import random

# total: 8_841_823
# groundtruth inclusion: 62_841


collection = []
with open('collections/msmarco-passage/collection.tsv', 'r') as f:
    for line in f:
        pid, text = line.strip().split('\t')
        collection.append((pid, text))


neccessary_pids = set()

for topic in ['health', 'science', 'tech']:
    for split in ['train', 'dev']:
        with open(f'code/data/raw_data/msmarco/msmarco_{topic}/{split}.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line)
                question = data['question']
                pids = data['docs_id']
                for pid in pids:
                    neccessary_pids.add(pid)

neccessary_collection = []
other_collection = []

for pid, text in collection:
    if pid in neccessary_pids:
        neccessary_collection.append((pid, text))
    else:
        other_collection.append((pid, text))



print(f'total collection: {len(collection)}')
print(f'groundtruth inclusion: {len(neccessary_pids)}')
print(f'neccessary collection: {len(neccessary_collection)}')
print(f'other collection: {len(other_collection)}')

old_total_num = len(collection)
new_total_num = 800_000
noised_num = new_total_num - len(neccessary_collection)

print(f'candidate noise num: {len(other_collection)}, noised num: {noised_num}')
random.seed(42) 
noised_collection = random.sample(other_collection, noised_num)
new_collection = neccessary_collection + noised_collection


print(f'final collection: {len(new_collection)}')

with open('collections/msmarco-passage/collection_800k.tsv', 'w') as f:
    for pid, text in new_collection:
        f.write(f'{pid}\t{text}\n')

