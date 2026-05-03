from __future__ import annotations

from app.schemas.project import PanelBox


def detector_training_panels_for_page(
    panels: list[PanelBox],
    page_number: int,
) -> list[PanelBox]:
    """Return page panels that should supervise detector training.

    Detector labels should reflect page structure, not recap inclusion.
    We therefore keep every surviving panel box on the page except boxes
    that are still machine-marked as auto-skipped junk.
    """
    target_page = int(page_number)
    return [
        panel
        for panel in panels
        if int(panel.page) == target_page and not bool(panel.auto_skipped)
    ]


def page_detector_training_signature(
    panels: list[PanelBox],
    page_number: int,
) -> list[tuple[int, int, int, int]]:
    relevant = detector_training_panels_for_page(panels, page_number)
    return sorted(
        [
            (
                int(panel.x),
                int(panel.y),
                int(panel.width),
                int(panel.height),
            )
            for panel in relevant
        ],
        key=lambda item: (item[1], item[0], item[3], item[2]),
    )


def changed_annotation_pages_for_detector_training(
    before_panels: list[PanelBox],
    after_panels: list[PanelBox],
) -> dict[int, list[PanelBox]]:
    changed: dict[int, list[PanelBox]] = {}
    page_numbers = sorted(
        {int(panel.page) for panel in before_panels}
        | {int(panel.page) for panel in after_panels}
    )
    for page_number in page_numbers:
        before_signature = page_detector_training_signature(before_panels, page_number)
        after_signature = page_detector_training_signature(after_panels, page_number)
        if before_signature == after_signature:
            continue
        changed[page_number] = detector_training_panels_for_page(after_panels, page_number)
    return changed
