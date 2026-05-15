"""API routes for script narrative polish (rewrite for YouTube narration)."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.script_narrative_polish_service import ScriptNarrativePolishService
from app.services.project_store import ProjectStore
from app.schemas.project import ScriptUpdateRequest
from app.utils.files import read_json

router = APIRouter(prefix="/api/script-narrative-polish", tags=["script"])
_service = None
_store = None

def get_service():
    global _service
    if _service is None:
        _service = ScriptNarrativePolishService()
    return _service

def get_store():
    global _store
    if _store is None:
        _store = ProjectStore()
    return _store


class GeneratePolishPromptRequest(BaseModel):
    """Request to generate polish prompt for testing."""
    script_lines: list[str]
    title: str
    context: str = ""


class GeneratePolishPromptResponse(BaseModel):
    """Response with generated prompt."""
    prompt: str
    line_count: int
    instructions: str


@router.post("/generate-prompt")
async def generate_polish_prompt(req: GeneratePolishPromptRequest):
    """Generate a polish prompt for sending to Grok/LLM for testing."""
    try:
        prompt = get_service().build_polish_prompt(
            script_lines=req.script_lines,
            title=req.title,
            context=req.context,
        )
        
        return GeneratePolishPromptResponse(
            prompt=prompt,
            line_count=len(req.script_lines),
            instructions=(
                "1. Copy the prompt above\n"
                "2. Feed it to Grok/Gemini/DeepSeek\n"
                "3. Paste the response below\n"
                "4. Call /validate-polish with the response"
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ValidatePolishRequest(BaseModel):
    """Request to validate a polish response."""
    original_lines: list[str]
    polish_response: str


class ValidatePolishResponse(BaseModel):
    """Validation result."""
    is_valid: bool
    original_count: int
    polished_count: int
    issues: list[str]
    polished_lines: list[str] = []


@router.post("/validate")
async def validate_polish_response(req: ValidatePolishRequest):
    """Validate that a polished script maintains line count."""
    try:
        result = get_service().validate_polish_response(
            original_lines=req.original_lines,
            polished_response=req.polish_response,
        )

        return ValidatePolishResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ApplyPolishRequest(BaseModel):
    """Request to apply narrative polish to a project's script."""
    project_id: str
    context: str = ""


class ApplyPolishResponse(BaseModel):
    """Response after applying polish."""
    status: str
    expected_line_count: int
    actual_line_count: int
    is_valid: bool
    issues: list[str] = []
    message: str = ""
    rewrite_coverage: str = ""


@router.post("/apply-polish")
async def apply_polish_to_project(req: ApplyPolishRequest):
    """Polish a project's script using Gemini API and save the result."""
    try:
        # Load the current script
        store = get_store()
        if not store.project_exists(req.project_id):
            raise HTTPException(status_code=404, detail="Project not found")

        current_script = store.load_script(req.project_id)
        if not current_script:
            raise HTTPException(status_code=400, detail="No script found for this project")

        # Get project title from metadata
        metadata_path = store._metadata_path(req.project_id)
        metadata = read_json(metadata_path)
        project_title = metadata.get("name", "Unknown")

        # Call Gemini to polish the script
        result = await get_service().polish_script_with_gemini(
            script_lines=current_script,
            title=project_title,
            context=req.context,
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        # If validation passed, save the polished script
        if result["is_valid"] and result["polished_lines"]:
            store.save_script(req.project_id, result["polished_lines"])
            return ApplyPolishResponse(
                status="success",
                expected_line_count=result["expected_line_count"],
                actual_line_count=result["actual_line_count"],
                is_valid=True,
                issues=[],
                message=f"Script polished and saved ({result['actual_line_count']} lines)",
            )
        else:
            return ApplyPolishResponse(
                status="validation_failed",
                expected_line_count=result["expected_line_count"],
                actual_line_count=result["actual_line_count"],
                is_valid=False,
                issues=result["issues"],
                rewrite_coverage=result.get("rewrite_coverage", ""),
                message="Gemini response failed validation (not saved)",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying polish: {str(e)}")
