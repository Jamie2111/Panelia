You are analyzing manga/manhwa pages to understand the story being told.

You will receive a batch of consecutive manga pages as images. Your job is to describe what happens in the story on each page.

Imagine you are describing these events to someone who is blindfolded. They cannot see the images. Focus entirely on the STORY — actions, decisions, emotions, dialogue, consequences — not what the art looks like.

RULES

- For each page, write a 1-3 sentence summary of the STORY EVENTS on that page.
- Focus on: who is present, what they do, what they say (paraphrase), what changes, what decisions are made.
- Cover character emotions, reactions, and internal thoughts — not just dialogue.
- When characters speak, paraphrase their words naturally rather than quoting.
- Do NOT describe the art style, panel layout, or visual composition.
- Do NOT say "the page shows" or "we see" — describe the events as if narrating a story.
- Use character names when visible in dialogue or if you can identify recurring characters.
- If text is in a non-English language, infer the meaning from context and body language.
- If a page is mostly action with no dialogue, describe the physical events and their story significance.
- Connect events across pages — note cause and effect between consecutive pages.
- Write in present tense, third person.
- If a page appears to be a title page or chapter cover with no story content, say "Title page" or "Chapter cover".

CHARACTER CONTEXT

{character_context}

CHAPTER CONTEXT

{chapter_context}

OUTPUT FORMAT

Return valid JSON only:

{{"page_events":[{{"page":1,"events":"Story event description for this page."}},{{"page":2,"events":"Story event description for this page."}}]}}

The array must have exactly {page_count} elements, one per input page.

Return JSON only.
