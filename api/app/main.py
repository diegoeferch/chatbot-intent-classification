from typing import Any

import uvicorn
from fastapi import FastAPI

from .models.request_params import RequestParams
from .models.intent_prediction import IntentPrediction
from .services.prediction import intent_prediction

app = FastAPI(title="Intent Classification API")


@app.post('/intent', response_model=IntentPrediction)
async def predict_intent(intent_params: RequestParams) -> Any:
    prediction = await intent_prediction(intent_params.message)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
