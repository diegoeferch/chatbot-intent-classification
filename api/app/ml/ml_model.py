import os.path
import pickle


def format_intent(intent: str):
    # TODO: Feature Engineering
    return []


class MlModel(object):
    _parent_folder_path = os.path.dirname(os.path.abspath(__file__))
    _pickle_model_path = os.path.join(_parent_folder_path, 'model.pkl')
    _model = None

    def __new__(cls, *args, **kwargs):
        cls._model = super(MlModel, cls).__new__(cls)
        print('Loading model ...')
        # TODO: Uncomment as soon as pickle is available
        # _model = pickle.load(open(cls._pickle_model_path, 'rb'))
        print('Model loaded!')

    @classmethod
    def predict(cls, intent: str):
        formatted_intent = format_intent(intent)
        prediction = cls._model.predict(formatted_intent)
        return prediction[0]
