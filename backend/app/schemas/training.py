from __future__ import annotations

from pydantic import BaseModel


class DetectorTrainingStatus(BaseModel):
    training_status: str = "idle"
    is_training: bool = False
    ready_to_train: bool = False
    progress_percent: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    min_new_annotations: int = 0
    panel_annotations_total: int = 0
    new_panel_annotations: int = 0
    ocr_annotations_total: int = 0
    new_ocr_annotations: int = 0
    remaining_annotations_until_ready: int = 0
    last_trained_at: str | None = None
    checkpoint_path: str | None = None
    log_path: str | None = None
    pid: int | None = None
    message: str | None = None
