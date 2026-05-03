from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.training import DetectorTrainingStatus
from app.services.detector_training_service import DetectorTrainingService


router = APIRouter(prefix="/api/training", tags=["training"])

service = DetectorTrainingService()


@router.get("/panel-detector", response_model=DetectorTrainingStatus)
def get_panel_detector_training_status() -> DetectorTrainingStatus:
    return service.get_status()


@router.post("/panel-detector/train", response_model=DetectorTrainingStatus)
def start_panel_detector_training() -> DetectorTrainingStatus:
    try:
        return service.start_training()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/panel-detector/cancel", response_model=DetectorTrainingStatus)
def cancel_panel_detector_training() -> DetectorTrainingStatus:
    try:
        return service.cancel_training()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
