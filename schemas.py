from pydantic import BaseModel
from typing import Dict

class PredictionSummary(BaseModel):
    activity_summary: Dict[str, int]