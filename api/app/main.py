from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI

from .models.request_params import RequestParams
from .models.intent_prediction import IntentPrediction
from .services.prediction import intent_prediction
from .ml.ml_model import MlModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loads model
    MlModel()
    yield


app = FastAPI(title="Intent Classification API", lifespan=lifespan)


@app.post('/intent', response_model=IntentPrediction)
async def predict_intent(intent_params: RequestParams) -> Any:
    prediction = await intent_prediction(intent_params)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
