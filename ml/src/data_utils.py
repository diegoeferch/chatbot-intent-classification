import json
import os
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from config import INTENT_DATASET_PATH, SINGLE_DATASETS_PATH, DATASET_ROOT_PATH


def load_intent_dataset() -> pd.DataFrame:
    return pd.read_json(INTENT_DATASET_PATH)


def create_intent_dataset() -> None:
    single_datasets_files = os.listdir(SINGLE_DATASETS_PATH)
    single_datasets = [f for f in single_datasets_files if os.path.isfile(os.path.join(SINGLE_DATASETS_PATH, f))]

    dfs = []
    for dataset_name in single_datasets:
        print(f'Processing {dataset_name}...')
        with open(os.path.join(SINGLE_DATASETS_PATH, dataset_name), 'r', encoding='utf-8') as dataset:
            data = json.load(dataset)
            dataset_df = pd.DataFrame(data)
            dfs.append(dataset_df)

    df = pd.concat(dfs, ignore_index=True)
    print(f'Writing to new file: {INTENT_DATASET_PATH}')
    df.to_json(os.path.join(DATASET_ROOT_PATH, 'intent-dataset.json'), orient='records')
    print('Done!')


def train_test_split_data(df: pd.DataFrame) -> List:
    return train_test_split(df, train_size=0.8, random_state=42)


def split_data(df: pd.DataFrame):
    x = df.drop('intent', axis=1)
    y = df['intent']
    return x, y
