from pydantic import BaseModel, field_validator


class IntentPrediction(BaseModel):
    intent: str
    probability: float

    @field_validator('probability')
    @classmethod
    def prob_formatting(cls, v: float) -> float:
        return round(v, 3)

