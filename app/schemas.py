from pydantic import BaseModel

class PredictRequest(BaseModel):
    user: int
    k: int = 10