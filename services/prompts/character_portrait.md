You are enumerating the recurring cast in a manga/comic chapter by looking directly at page images.

Return strict JSON only in this format:
{
  "characters": [
    {
      "name": "Canonical Name or best-supported descriptive placeholder",
      "role": "protagonist|antagonist|supporting|cameo",
      "visual_description": "Short, concrete visual description",
      "portrait_pages": [1, 2],
      "aliases": ["optional alternate spellings or labels"]
    }
  ]
}

Rules:
- Read the page images directly. Do not infer names from OCR unless the name is visibly present in the artwork and clearly legible.
- Prefer real names when the page art or dialogue clearly supports them.
- If the real name is not readable yet, use a stable descriptive placeholder like "Unknown white-haired girl" rather than hallucinating.
- Return one entry per recurring person, not one entry per outfit, pose, or page.
- If the same person appears with multiple labels like "Mr. Zhang", "Zhang", "Protagonist", or a close spelling variant, pick the best-supported canonical name for `name` and put the alternates in `aliases`.
- Do not use generic labels like "Protagonist", "Unknown man", or "Main character" as `name` if a more specific surname or full name is plausibly supported elsewhere in the same batch.
- If a character is visibly the same person you already enumerated in this batch (same hair color, same gender, same outfit or hairstyle), emit exactly one entry and list every label you saw them under in `aliases` — never split one real person across multiple canonical entries.
- Do not output generic phrases, dialogue fragments, or OCR junk as names.
- Keep `visual_description` brief and visually grounded.
- Only include characters who recur or clearly matter to the chapter.
- If no meaningful recurring character is visible, return an empty `characters` array.

Chapter context:
{chapter_context}

Known/project hints:
{project_context}

Page count in this request: {page_count}
