from pydantic import BaseModel


class CaptionResponse(BaseModel):
    success: bool
    greedy_caption: str
    beam_caption: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
