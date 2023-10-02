import os.path
import pickle
from pathlib import Path
import numpy as np

from ..utils.features import normalize_intent


async def format_intent(vectorizer, intent: str):
    normalized_str = normalize_intent(intent)
    train_features = vectorizer.transform(normalized_str).toarray()
    return train_features


class MlModel(object):
    _parent_folder_path = str(Path(__file__).parent)
    _pickle_model_path = os.path.join(_parent_folder_path, 'pickles', 'lr_model.pkl')
    _pickle_vect_path = os.path.join(_parent_folder_path, 'pickles', 'lr_vectorizer.pkl')
    _model = None
    _vect = None

    def __new__(cls, *args, **kwargs):
        cls._model = super(MlModel, cls).__new__(cls)
        print('Loading model ...')

        if os.path.exists(cls._pickle_model_path):
            with open(cls._pickle_model_path, 'rb') as pkl_file:
                cls._model = pickle.load(pkl_file)
            print(f'Model loaded!: {type(cls._model)}')
        else:
            print('Model file not found!!')

        if os.path.exists(cls._pickle_vect_path):
            with open(cls._pickle_vect_path, 'rb') as pkl_file:
                cls._vect = pickle.load(pkl_file)
            print(f'Vectorizer loaded!: {type(cls._vect)}')

    @classmethod
    async def predict_intent(cls, message: str):
        if isinstance(message, str):
            formatted_intent = await format_intent(cls._vect, message)
            prediction_labels = cls._model.predict(formatted_intent)
            prediction_prob = cls._model.predict_proba(formatted_intent)
            pred_index = np.argmax(prediction_prob)
            return prediction_labels[0], prediction_prob[0, pred_index]
