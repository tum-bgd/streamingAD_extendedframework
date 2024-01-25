import os
import json

datasets_json_output_path = '../custom_datasets.json'
collection_paths = [
    '../data/univariate/KDD-TSAD',
    '../data/univariate/NASA-MSL',
]

with open(datasets_json_output_path, 'r') as f:
    output_json = json.load(f)

for collection_path in collection_paths:
    for dataset_json_path in sorted([x for x in os.listdir(collection_path) if x.endswith('.json')]):
        with open(f'{collection_path}/{dataset_json_path}', 'r') as f:
            collection_name, dataset_name = json.load(f)[0]['dataset_id']
            output_json[dataset_name] = {
                'test_path': f'{collection_path[3:]}/{dataset_name}.test.csv',
                'train_path': f'{collection_path[3:]}/{dataset_name}.train.csv',
                'type': 'real',
            }
with open(datasets_json_output_path, 'w') as f:
    json.dump(output_json, f)

