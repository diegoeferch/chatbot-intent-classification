import json
import os
import pandas as pd

PIPELINES_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATASETS_FOLDER = os.path.join(PIPELINES_FOLDER, '../datasets')
SINGLE_DATASETS_FOLDER = os.path.join(DATASETS_FOLDER, 'singles')


single_datasets_files = os.listdir(SINGLE_DATASETS_FOLDER)
single_datasets = [f for f in single_datasets_files if os.path.isfile(os.path.join(SINGLE_DATASETS_FOLDER, f))]


def aggregate_datasets():
    dfs = []
    for dataset_name in single_datasets:
        print(f'Processing {dataset_name}...')
        with open(os.path.join(SINGLE_DATASETS_FOLDER, dataset_name), 'r', encoding='utf-8') as dataset:
            data = json.load(dataset)
            dataset_df = pd.DataFrame(data)
            dfs.append(dataset_df)

    df = pd.concat(dfs, ignore_index=True)
    print('Writing to new file...')
    df.to_json(os.path.join(DATASETS_FOLDER, 'intent-dataset.json'), orient='records')
    print('Done!')


if __name__ == '__main__':
    aggregate_datasets()
