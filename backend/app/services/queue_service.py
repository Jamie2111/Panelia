from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from redis import Redis

from app.core.config import get_settings
from app.utils.files import ensure_dir


@dataclass
class QueueMessage:
    project_id: str
    job_id: str
    stage: str


class QueueService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.queue_key = "panelia:jobs"
        self.queue_root = ensure_dir(self.settings.data_dir / "_queue")
        self.message_dir = ensure_dir(self.queue_root / "messages")
        self.cancel_dir = ensure_dir(self.queue_root / "cancel")
        self.pause_dir = ensure_dir(self.queue_root / "pause")
        self.client: Redis | None = None
        self.mode = "filesystem"

        try:
            client = Redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            client.ping()
            self.client = client
            self.mode = "redis"
        except Exception:
            self.client = None

    def enqueue(self, project_id: str, job_id: str, stage: str) -> None:
        payload = json.dumps({"project_id": project_id, "job_id": job_id, "stage": stage})
        if self.client:
            self.client.rpush(self.queue_key, payload)
            return

        queue_file = self.message_dir / f"{time.time_ns()}_{job_id}_{uuid4().hex}.json"
        queue_file.write_text(payload, encoding="utf-8")

    def reserve(self, timeout_seconds: int = 5) -> QueueMessage | None:
        if self.client:
            message = self.client.blpop(self.queue_key, timeout=timeout_seconds)
            if not message:
                return None
            _, payload = message
            data = json.loads(payload)
            return QueueMessage(**data)

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            queued_files = sorted(self.message_dir.glob("*.json"))
            for path in queued_files:
                claim_path = Path(f"{path}.claimed")
                try:
                    path.rename(claim_path)
                except FileNotFoundError:
                    continue

                payload = claim_path.read_text(encoding="utf-8")
                claim_path.unlink(missing_ok=True)
                data = json.loads(payload)
                return QueueMessage(**data)
            time.sleep(0.25)
        return None

    def request_cancel(self, job_id: str) -> None:
        if self.client:
            self.client.setex(f"panelia:cancel:{job_id}", 86400, "1")
            return
        (self.cancel_dir / f"{job_id}.flag").write_text("1", encoding="utf-8")

    def is_cancel_requested(self, job_id: str) -> bool:
        if self.client:
            return self.client.exists(f"panelia:cancel:{job_id}") == 1
        return (self.cancel_dir / f"{job_id}.flag").exists()

    def clear_cancel(self, job_id: str) -> None:
        if self.client:
            self.client.delete(f"panelia:cancel:{job_id}")
            return
        (self.cancel_dir / f"{job_id}.flag").unlink(missing_ok=True)

    def request_pause(self, job_id: str) -> None:
        if self.client:
            self.client.setex(f"panelia:pause:{job_id}", 86400, "1")
            return
        (self.pause_dir / f"{job_id}.flag").write_text("1", encoding="utf-8")

    def is_pause_requested(self, job_id: str) -> bool:
        if self.client:
            return self.client.exists(f"panelia:pause:{job_id}") == 1
        return (self.pause_dir / f"{job_id}.flag").exists()

    def clear_pause(self, job_id: str) -> None:
        if self.client:
            self.client.delete(f"panelia:pause:{job_id}")
            return
        (self.pause_dir / f"{job_id}.flag").unlink(missing_ok=True)
