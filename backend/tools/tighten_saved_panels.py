from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.project import PanelBox, PipelineStage, StageStatus
from app.services.panel_detection_service import MagiPanelDetectionService
from app.services.project_store import ProjectStore


def tighten_project(store: ProjectStore, project_id: str) -> dict[str, object]:
    project = store.get_project(project_id)
    panels = list(project.panels)
    if not panels:
        return {"project_id": project_id, "changed": 0, "skipped": True, "reason": "no-panels"}

    page_paths = store.list_page_paths(project_id)
    detector = MagiPanelDetectionService()
    updated: list[PanelBox] = []
    changed = 0
    total_area_before = 0
    total_area_after = 0

    page_images: dict[int, Image.Image] = {}
    try:
        for panel in panels:
            total_area_before += max(1, panel.width) * max(1, panel.height)
            page_index = max(panel.page - 1, 0)
            if page_index >= len(page_paths):
                updated.append(panel)
                total_area_after += max(1, panel.width) * max(1, panel.height)
                continue
            if page_index not in page_images:
                page_images[page_index] = Image.open(page_paths[page_index]).convert("RGB")
            image = page_images[page_index]
            image_array = np.asarray(image)
            x, y, w, h = detector._refine_box_to_content(
                image_array,
                (panel.x, panel.y, panel.width, panel.height),
                image.width,
                image.height,
            )
            if (x, y, w, h) != (panel.x, panel.y, panel.width, panel.height):
                changed += 1
                panel = panel.model_copy(update={"x": x, "y": y, "width": w, "height": h})
            updated.append(panel)
            total_area_after += max(1, panel.width) * max(1, panel.height)
    finally:
        for image in page_images.values():
            image.close()

    if changed:
        store.save_panels(project_id, updated)
        store.update_stage_state(
            project_id,
            PipelineStage.PANEL_REVIEW,
            StageStatus.COMPLETED,
            progress=100,
            message="Panel boxes tightened to panel content and saved.",
        )
        store.update_stage_state(
            project_id,
            PipelineStage.SCRIPT_GENERATION,
            StageStatus.READY,
            progress=0,
            message="Panel crops tightened. Generate the script when you're ready.",
        )
        store.update_stage_state(
            project_id,
            PipelineStage.NARRATION_GENERATION,
            StageStatus.PENDING,
            progress=0,
            message="Generate a script before creating audio.",
        )
        store.update_stage_state(
            project_id,
            PipelineStage.VIDEO_RENDERING,
            StageStatus.PENDING,
            progress=0,
            message="Generate audio before rendering video.",
        )

    area_delta = 0.0
    if total_area_before > 0:
        area_delta = (total_area_before - total_area_after) / total_area_before

    return {
        "project_id": project_id,
        "changed": changed,
        "panel_count": len(panels),
        "area_reduction_ratio": round(area_delta, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tighten existing saved panel boxes to visible content.")
    parser.add_argument("project_ids", nargs="+", help="Project ids to update.")
    args = parser.parse_args()

    store = ProjectStore()
    results = [tighten_project(store, project_id) for project_id in args.project_ids]
    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
