# Deprecated Script Services

As of the vision-grounded narration refactor, the following services are
**deprecated** and should not be used for new code paths.

The new single source of truth is:
- `app/services/panel_vision_narrator.py` — vision-grounded panel narration
- Triggered when `pipeline_config.script_pipeline_version == "vision"`

## Why deprecated

The pre-vision pipeline ran 8+ services in a cascade that compounded errors:

```
script_generator → script_polisher → script_cleaner_service →
story_beats → story_grounding → story_script_service →
script_quality_service → story_segment_repair_service →
script_narrative_polish_service → script_generation_vnext
```

This produced:
- Duplicate narrations from polish passes rewriting good output
- Generic templates that don't match panel content (no vision input)
- Quality scores of ~17/100 with no path to fix in-place
- ~50,000 lines of overlapping concerns across services

## Deprecated services

| File | Lines | Status |
|------|------|--------|
| `script_generator.py` | — | Deprecated. Use `PanelVisionNarrator`. |
| `script_polisher.py` | 1396 | Deprecated. Polish passes corrupt good output. |
| `script_quality_service.py` | 2263 | Deprecated. Quality gates that don't gate. |
| `script_cleaner_service.py` | — | Deprecated. The new path doesn't need cleanup. |
| `script_narrative_polish_service.py` | — | Deprecated. |
| `script_generation_vnext.py` | — | Superseded by the vision pipeline. |
| `story_script_service.py` | — | Deprecated. |
| `story_segment_repair_service.py` | — | Deprecated. Bad panels get regenerated in-place. |
| `story_grounding.py` | — | Helpers may be salvageable (name extraction). |
| `story_beats.py` | — | Deprecated. |

## How to migrate a project

1. Open the project's `metadata.json`
2. Set `pipeline_config.script_pipeline_version = "vision"`
3. Re-run `SCRIPT_GENERATION` — the new vision narrator takes over

## When to actually delete

These files are kept on disk so existing projects that haven't been migrated
to `"vision"` mode continue to function. Once all projects are migrated and
the legacy paths have not been exercised for a release cycle, the files can
be deleted alongside their imports in `pipeline/stages.py`.

Tracking task: see roadmap day 1 — "rip out 8 of 9 script services".
