import asyncio
from typing import Tuple
from ..ml.ml_model import MlModel


async def intent_prediction(intent: str) -> dict:
    # prediction = await MlModel.predict(intent)
    await asyncio.sleep(0.3)
    return {'intent': 'Yay!', 'probability': 0.678543}
