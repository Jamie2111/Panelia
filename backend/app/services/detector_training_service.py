from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from app.core.config import get_settings
from app.schemas.training import DetectorTrainingStatus
from app.utils.files import ensure_dir, read_json, write_json


BACKEND_ROOT = Path(__file__).resolve().parents[2]


class DetectorTrainingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._status_path = self.settings.panel_detector_models_dir / "panel_detector_training_status.json"
        self._metadata_path = self.settings.panel_detector_models_dir / "panel_detector_v2.metadata.json"
        self._log_path = ensure_dir(self.settings.data_dir / "_service_logs") / "panel_detector_training.log"
        self._panel_annotations_dir = ensure_dir(self.settings.training_data_dir / "annotations")
        self._ocr_annotations_dir = ensure_dir(self.settings.ocr_training_data_dir / "annotations")

    def get_status(self) -> DetectorTrainingStatus:
        state = self._reconcile_runtime_state()
        metadata = self._load_metadata()
        last_trained_at = self._parse_datetime(metadata.get("trained_at"))

        panel_total, panel_new = self._annotation_counts(self._panel_annotations_dir, last_trained_at)
        ocr_total, ocr_new = self._annotation_counts(self._ocr_annotations_dir, last_trained_at)
        min_new = max(1, int(self.settings.panel_detector_training_min_new_annotations))
        ready = panel_new >= min_new and not bool(state.get("is_training"))
        remaining = max(min_new - panel_new, 0)

        message = str(state.get("message") or "").strip() or self._default_message(
            is_training=bool(state.get("is_training")),
            ready=ready,
            panel_new=panel_new,
            min_new=min_new,
        )

        return DetectorTrainingStatus(
            training_status=str(state.get("training_status") or "idle"),
            is_training=bool(state.get("is_training")),
            ready_to_train=ready,
            progress_percent=float(state.get("progress_percent") or 0.0),
            current_epoch=int(state.get("current_epoch") or 0),
            total_epochs=int(state.get("total_epochs") or 0),
            train_loss=float(state["train_loss"]) if isinstance(state.get("train_loss"), (int, float)) else None,
            val_loss=float(state["val_loss"]) if isinstance(state.get("val_loss"), (int, float)) else None,
            min_new_annotations=min_new,
            panel_annotations_total=panel_total,
            new_panel_annotations=panel_new,
            ocr_annotations_total=ocr_total,
            new_ocr_annotations=ocr_new,
            remaining_annotations_until_ready=remaining,
            last_trained_at=metadata.get("trained_at"),
            checkpoint_path=metadata.get("checkpoint_path"),
            log_path=str(self._log_path),
            pid=state.get("pid"),
            message=message,
        )

    def start_training(self) -> DetectorTrainingStatus:
        status = self.get_status()
        if status.is_training:
            raise RuntimeError("Panel detector training is already running.")
        if status.new_panel_annotations <= 0:
            raise RuntimeError("No new corrected panel annotations are available yet.")

        command = [sys.executable, "training/train_panel_detector.py"]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(BACKEND_ROOT)
        environment["PANELIA_TRAINING_STATUS_PATH"] = str(self._status_path)

        with self._log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=BACKEND_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        write_json(
            self._status_path,
            {
                "training_status": "running",
                "is_training": True,
                "pid": int(process.pid),
                "started_at": self._now().isoformat(),
                "progress_percent": 0.0,
                "current_epoch": 0,
                "total_epochs": 0,
                "train_loss": None,
                "val_loss": None,
                "message": "Training the panel detector on human-corrected pages.",
            },
        )
        return self.get_status()

    def cancel_training(self) -> DetectorTrainingStatus:
        state = self._reconcile_runtime_state()
        if not bool(state.get("is_training")):
            raise RuntimeError("Panel detector training is not running.")

        pid = state.get("pid")
        if not isinstance(pid, int):
            raise RuntimeError("Panel detector training does not have a live process to cancel.")

        try:
            process_group = os.getpgid(pid)
            os.killpg(process_group, signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

        deadline = time.time() + 3.0
        while time.time() < deadline and self._pid_is_running(pid):
            time.sleep(0.1)

        cancelled_state = {
            "training_status": "cancelled",
            "is_training": False,
            "pid": None,
            "started_at": state.get("started_at"),
            "progress_percent": float(state.get("progress_percent") or 0.0),
            "current_epoch": int(state.get("current_epoch") or 0),
            "total_epochs": int(state.get("total_epochs") or 0),
            "train_loss": state.get("train_loss"),
            "val_loss": state.get("val_loss"),
            "message": "Detector training cancelled.",
        }
        write_json(self._status_path, cancelled_state)
        return self.get_status()

    def _reconcile_runtime_state(self) -> dict[str, Any]:
        state = read_json(self._status_path, default={}) or {}
        if str(state.get("training_status") or "") != "running":
            return state

        pid = state.get("pid")
        if isinstance(pid, int) and self._pid_is_running(pid):
            return state

        metadata = self._load_metadata()
        started_at = self._parse_datetime(state.get("started_at"))
        trained_at = self._parse_datetime(metadata.get("trained_at"))
        if trained_at is not None and (started_at is None or trained_at >= started_at):
            reconciled = {
                "training_status": "completed",
                "is_training": False,
                "pid": None,
                "started_at": state.get("started_at"),
                "progress_percent": 100.0,
                "current_epoch": int(state.get("total_epochs") or state.get("current_epoch") or 0),
                "total_epochs": int(state.get("total_epochs") or 0),
                "train_loss": state.get("train_loss"),
                "val_loss": state.get("val_loss"),
                "message": "Latest detector checkpoint is ready to use.",
            }
        else:
            reconciled = {
                "training_status": "failed",
                "is_training": False,
                "pid": None,
                "started_at": state.get("started_at"),
                "progress_percent": float(state.get("progress_percent") or 0.0),
                "current_epoch": int(state.get("current_epoch") or 0),
                "total_epochs": int(state.get("total_epochs") or 0),
                "train_loss": state.get("train_loss"),
                "val_loss": state.get("val_loss"),
                "message": "Detector training stopped before a new checkpoint finished.",
            }
        write_json(self._status_path, reconciled)
        return reconciled

    def _load_metadata(self) -> dict[str, Any]:
        return read_json(self._metadata_path, default={}) or {}

    def _annotation_counts(self, directory: Path, last_trained_at: datetime | None) -> tuple[int, int]:
        annotation_paths = sorted(directory.glob("*.json"))
        total = len(annotation_paths)
        if last_trained_at is None:
            return total, total
        new_count = sum(1 for path in annotation_paths if self._mtime(path) > last_trained_at)
        return total, new_count

    def _default_message(self, *, is_training: bool, ready: bool, panel_new: int, min_new: int) -> str:
        if is_training:
            return "Training the panel detector on the newest corrected pages."
        if ready:
            return "New training data available. Train detector now."
        if panel_new <= 0:
            return "Save corrected panel boxes to start building detector training data."
        return f"{panel_new} new corrected page{'s' if panel_new != 1 else ''} saved. Collect {max(min_new - panel_new, 0)} more to train confidently."

    def _mtime(self, path: Path) -> datetime:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None

    def _pid_is_running(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _now(self) -> datetime:
        return datetime.now(tz=UTC)
