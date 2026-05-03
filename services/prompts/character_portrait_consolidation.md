You are consolidating a provisional manga/manhwa chapter cast list into one canonical roster.

Return strict JSON only in this format:
{
  "characters": [
    {
      "name": "Canonical Name or best-supported descriptive placeholder",
      "role": "protagonist|antagonist|supporting|cameo",
      "visual_description": "Short, concrete visual description",
      "portrait_pages": [1, 2],
      "aliases": ["alternate spellings, titles, or descriptive labels"]
    }
  ]
}

Rules:
- Merge duplicate identities across the provisional list.
- Prefer a specific real name over titles, surnames alone, or generic placeholders when the list gives enough support.
- Treat close spelling variants of the same likely name as aliases under one canonical entry.
- Honorific + surname combinations ("Mr. Zhang", "Sr. Zhang", "Xiao Zhang", "Senhor Zhang") MUST fold into the most specific full name ("Zhang Yi") as aliases — never as separate canonical entries.
- Placeholder names like "Protagonist", "Main character", "Unknown <color>-haired man", "Unknown woman", "Speaker", "Victim", "Figure", "Old woman", "Delivery man" MUST be merged into a named canonical that shares gender AND hair color AND role, unless an appearance marker clearly conflicts (different hair color, different gender, different age bracket).
- If a placeholder entry cannot be merged into any named canonical, set its `role` to `"cameo"` and keep the descriptive placeholder as `name` so downstream code can filter it out.
- Keep distinct people separate when gender, age, hair color, role, or page context clearly conflict.
- Do not return multiple protagonist entries for the same recurring lead unless the provisional evidence clearly shows distinct protagonists.
- Never emit two canonical entries that describe the same real person. When in doubt, merge and keep the less-specific label as an alias.
- Keep aliases useful for downstream matching: include title variants, descriptive placeholders, and near-spellings that refer to the same person.
- Do not invent new facts or names that are not reasonably supported by the provisional roster.

Chapter context:
{chapter_context}

Project context:
{project_context}

Provisional roster:
{provisional_characters}
