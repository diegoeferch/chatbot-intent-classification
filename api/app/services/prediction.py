import asyncio
from typing import Tuple
from ..ml.ml_model import MlModel
from ..models.request_params import RequestParams


async def intent_prediction(params: RequestParams) -> dict:
    label, prob = await MlModel.predict_intent(params.message)
    return {'intent': label, 'probability': prob}
