"""
Script Narrative Polish Service

Rewrites generated scripts for YouTube narration while:
- Maintaining exact sentence count (for panel alignment)
- Converting panel descriptions to cohesive storytelling
- Fixing repetitive/generic lines
- Adding narrative context and research
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from app.core.config import get_settings


class ScriptNarrativePolishService:
    """Polish generated scripts for YouTube narration quality."""

    def __init__(self):
        self.settings = get_settings()

        # Initialize Gemini API if key is available
        if GEMINI_AVAILABLE and self.settings.gemini_api_key:
            genai.configure(api_key=self.settings.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name=self.settings.llm_gemini_model
            )
        else:
            self.gemini_model = None
    
    def build_polish_prompt(
        self,
        script_lines: list[str],
        title: str,
        context: str = "",
    ) -> str:
        """Build the LLM prompt for narrative polishing.

        Focus on enrichment and detail while preserving sentence count.
        Encourages narrative depth and cohesion without fabrication.
        """
        # Extract only non-empty lines
        non_empty_lines = [
            line for line in script_lines if line.strip()
        ]

        # Format lines without numbers - just one per line
        script_text = "\n".join(non_empty_lines)

        prompt = f"""Enrich this manga script narrative for YouTube storytelling. CRITICAL: Do not change sentence count per line.

Title: {title}
Context: {context}

YOUR TASK: Enhance these narrations WITHOUT changing sentence count:
1. Expand generic/repetitive lines with more specific narrative detail
2. When a character action repeats, add narrative context (why, how, consequences) instead of removing it
3. Connect disparate lines with smoother transitions and narrative flow
4. Enrich with emotional context, character motivation, and story impact
5. Make each line more vivid and engaging while staying faithful to what's shown

CONSTRAINTS (STRICT):
- MAINTAIN EXACT SENTENCE COUNT per line (count sentences before and after must match)
- Do NOT split one sentence into two
- Do NOT combine two sentences into one
- Do NOT fabricate character names or details not in the original
- Do NOT change the core tone or pacing - enhance within the existing style
- Only add details that naturally follow from the visual or narrative context

EXAMPLE - Enrichment approach:
Original (2 sentences): "Hiro wakes up. He's confused."
Enriched (still 2 sentences): "Hiro's eyes flutter open to an unfamiliar ceiling, his mind sluggish and disoriented. The unfamiliar surroundings press in on him, confusion washing over his features."

Key principles:
- Repetition of action = opportunity to show progression or emotional depth, not removal
- Generic description = opportunity to add specificity from visual context
- Sparse dialogue = opportunity to add character voice and motivation
- Tense shifts = keep consistent but within that, add narrative richness

Original script (one line per panel):
{script_text}

Return the rewritten script as plain text, one line per panel. Do not add line numbers, headers, or explanations.
"""
        return prompt
    
    def count_lines_in_response(self, response: str) -> int:
        """Count numbered lines in LLM response."""
        # Match lines like "1. text" or "1 text"
        pattern = r'^\s*\d+\.\s+'
        lines = [
            line.strip()
            for line in response.split('\n')
            if line.strip() and re.match(pattern, line.strip())
        ]
        return len(lines)
    
    def _extract_numbered_lines_from_response(self, response: str) -> list[tuple[int, str]]:
        """Extract (line_number, content) pairs from numbered response.

        Expected format:
        5. Rewritten content for line 5
        8. Rewritten content for line 8
        ...
        """
        items = []
        pattern = r'^(\d+)\.\s*(.*)$'

        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            match = re.match(pattern, line)
            if match:
                line_num = int(match.group(1))
                content = match.group(2)
                items.append((line_num, content))

        return items

    def extract_lines_from_response(self, response: str) -> list[str]:
        """Extract lines from numbered LLM response.

        Expected format:
        1. First line of narration
        2.
        3. Third line of narration

        Only lines that start with "N. " are counted. Empty lines show just "N. "
        """
        lines = []
        pattern = r'^\d+\.\s*(.*)$'

        for line in response.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                # Found a numbered line
                content = match.group(1)
                lines.append(content)

        return lines
    
    def validate_polish_response(
        self,
        original_lines: list[str],
        polished_response: str,
    ) -> dict[str, Any]:
        """Validate and reconstruct the polished response.

        The response contains only non-empty lines (one per line, no numbering).
        We reconstruct the full script by putting rewritten lines back
        in positions of non-empty lines and preserving original empty lines.
        """
        # Get the rewritten lines from the response (simple split)
        rewritten_lines = [
            line.rstrip()  # Remove trailing whitespace but preserve content
            for line in polished_response.strip().split('\n')
        ]

        original_count = len(original_lines)
        non_empty_count = sum(1 for line in original_lines if line.strip())
        issues = []

        # Validate count
        if len(rewritten_lines) != non_empty_count:
            issues.append(
                f"Rewritten line count mismatch: expected {non_empty_count}, got {len(rewritten_lines)}"
            )

        # Reconstruct the full script
        polished_lines = []
        rewritten_idx = 0
        filled_with_original = 0

        for i, original_line in enumerate(original_lines):
            if original_line.strip():
                # This was a non-empty line - use the rewritten version if available
                if rewritten_idx < len(rewritten_lines):
                    polished_lines.append(rewritten_lines[rewritten_idx])
                    rewritten_idx += 1
                else:
                    # Missing from response - use original
                    polished_lines.append(original_line)
                    filled_with_original += 1
            else:
                # This was an empty line - preserve it
                polished_lines.append('')

        # Final count check
        if len(polished_lines) != original_count:
            issues.append(
                f"Final line count mismatch: expected {original_count}, got {len(polished_lines)}"
            )

        # If we have the right count but some lines weren't rewritten, that's OK
        # (partial success is better than complete failure)
        is_valid = (
            len(polished_lines) == original_count
            and len(rewritten_lines) >= non_empty_count * 0.8  # At least 80% rewritten
        )

        return {
            "is_valid": is_valid,
            "original_count": original_count,
            "polished_count": len(polished_lines),
            "issues": issues,
            "polished_lines": polished_lines if len(polished_lines) == original_count else [],
            "rewrite_coverage": f"{len(rewritten_lines)}/{non_empty_count}",
            "filled_with_original": filled_with_original,
        }
    
    async def polish_script_with_gemini_batch(
        self,
        script_lines: list[str],
        title: str,
        context: str = "",
        batch_size: int = 150,
    ) -> dict[str, Any]:
        """Polish a large script in batches to stay under token limits.

        Processes non-empty lines in batches, preserving structure and empty lines.
        """
        original_count = len(script_lines)
        non_empty_indices = [i for i, line in enumerate(script_lines) if line.strip()]

        if not non_empty_indices:
            return {
                "status": "success",
                "expected_line_count": original_count,
                "actual_line_count": original_count,
                "is_valid": True,
                "issues": [],
                "polished_lines": script_lines,
            }

        all_polished_non_empty = []
        issues = []

        # Process in batches
        for batch_start in range(0, len(non_empty_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(non_empty_indices))
            batch_indices = non_empty_indices[batch_start:batch_end]

            # Extract the batch of non-empty lines
            batch_lines = [script_lines[i] for i in batch_indices]

            # Polish this batch
            result = await self.polish_script_with_gemini_single_batch(
                batch_lines, title, context
            )

            if result["status"] != "success":
                issues.append(f"Batch {batch_start//batch_size + 1} failed: {result['message']}")
                # Use original lines if polish fails
                all_polished_non_empty.extend(batch_lines)
            else:
                all_polished_non_empty.extend(result["polished_lines"])

        # Reconstruct full script with empty lines preserved
        polished_lines = []
        non_empty_idx = 0
        for i, original_line in enumerate(script_lines):
            if original_line.strip():
                if non_empty_idx < len(all_polished_non_empty):
                    polished_lines.append(all_polished_non_empty[non_empty_idx])
                    non_empty_idx += 1
                else:
                    polished_lines.append(original_line)
            else:
                polished_lines.append('')

        return {
            "status": "success" if not issues else "partial_success",
            "expected_line_count": original_count,
            "actual_line_count": len(polished_lines),
            "is_valid": len(polished_lines) == original_count and not issues,
            "issues": issues,
            "polished_lines": polished_lines,
            "batches_processed": (len(non_empty_indices) + batch_size - 1) // batch_size,
        }

    async def polish_script_with_gemini_single_batch(
        self,
        batch_lines: list[str],
        title: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Polish a single batch of non-empty lines."""
        if not self.gemini_model:
            return {
                "status": "error",
                "message": "Gemini API not configured. Set GEMINI_API_KEY.",
                "expected_line_count": len(batch_lines),
                "polished_lines": [],
            }

        prompt = self.build_polish_prompt(batch_lines, title, context)

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.75,
                    top_p=0.95,
                    max_output_tokens=8000,
                ),
            )

            if not response.text:
                return {
                    "status": "error",
                    "message": "Gemini returned empty response",
                    "expected_line_count": len(batch_lines),
                    "polished_lines": [],
                }

            # Validate response
            validation = self.validate_polish_response(
                original_lines=batch_lines,
                polished_response=response.text,
            )

            return {
                "status": "success" if validation["is_valid"] else "validation_failed",
                "expected_line_count": len(batch_lines),
                "actual_line_count": validation["polished_count"],
                "is_valid": validation["is_valid"],
                "issues": validation["issues"],
                "polished_lines": validation["polished_lines"],
            }

        except Exception as e:
            import traceback
            return {
                "status": "error",
                "message": f"Gemini API error: {str(e)}",
                "expected_line_count": len(batch_lines),
                "polished_lines": [],
                "error_trace": traceback.format_exc(),
            }

    async def polish_script_with_gemini(
        self,
        script_lines: list[str],
        title: str,
        context: str = "",
    ) -> dict[str, Any]:
        """
        Polish a script for narration quality using Gemini API.

        Uses batch processing for large scripts to stay under token limits.

        Args:
            script_lines: List of script lines
            title: Project title
            context: Optional context about the project

        Returns:
            Dict with polished_lines, validation results, and metadata
        """
        if not self.gemini_model:
            return {
                "status": "error",
                "message": "Gemini API not configured. Set GEMINI_API_KEY.",
                "expected_line_count": len(script_lines),
            }

        # Use batch processing for large scripts
        return await self.polish_script_with_gemini_batch(
            script_lines, title, context, batch_size=150
        )

    async def polish_script(
        self,
        script_lines: list[str],
        title: str,
        context: str = "",
        llm_provider: str | None = None,
    ) -> dict[str, Any]:
        """
        Polish a script for YouTube narration using an LLM.

        Args:
            script_lines: List of script lines
            title: Project title
            context: Optional context
            llm_provider: LLM provider to use (defaults to Gemini)

        Returns:
            Dict with polished_lines, validation results, and metadata
        """
        provider = llm_provider or "gemini"

        if provider == "gemini":
            return await self.polish_script_with_gemini(
                script_lines, title, context
            )
        else:
            return {
                "status": "error",
                "message": f"LLM provider '{provider}' not implemented",
                "expected_line_count": len(script_lines),
            }
