import os
from pathlib import Path

DATASET_ROOT_PATH = str(Path(__file__).parent.parent / "datasets")
SINGLE_DATASETS_PATH = os.path.join(DATASET_ROOT_PATH, 'singles')
INTENT_DATASET_PATH = os.path.join(DATASET_ROOT_PATH, 'intent-dataset.json')
