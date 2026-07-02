"""Pydantic schemas for violation API payloads."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ViolationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    plate_number: str
    timestamp: datetime
    image_path: str
    status: str


class ViolationCreateResponse(BaseModel):
    id: int
    plate_number: str
    timestamp: datetime
    image_path: str
    status: str
    message: str = "Violation recorded and alert dispatched"
