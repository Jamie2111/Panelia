You are writing panel-by-panel narration for a manga/manhwa/webtoon recap video.

You will receive panel images alongside extracted text for each panel.

TASK: Write exactly one narration line per panel listed below.

Return valid JSON only in this format:

{"panel_narrations":[{"panel":1,"narration":"Short narration line for this panel."}]}

MULTIMODAL INSTRUCTIONS

- Look at each panel image to understand what is happening in the story.
- Read any visible text in the images (dialogue bubbles, signs, SFX) in any language.
- Use the extracted text as supplementary context, but trust the images when text is garbled or missing.
- For non-English manga: read the original language text directly from the image and write English narration.
- Do NOT echo garbled OCR fragments. Describe the story event from the image context instead.
- If a panel image is not provided, rely on extracted text and surrounding panel context.

NARRATION STYLE

You are narrating this story to someone who is blindfolded — they cannot see the images.
Describe what HAPPENS in the story: actions, decisions, emotions, spoken words, consequences.
When characters speak, weave their words naturally into the narration rather than quoting them directly.
Be vivid and detailed — this narration will be used for a YouTube video, so every line must hold the listener's attention.
Write in present tense, third person.

CRITICAL RULES

- Return exactly one item for every input panel, in the same order.
- Each narration line MUST be a complete English sentence with a subject and a verb.
- No sentence fragments like "Mocking his efforts" or "Having been granted a second chance."
- Each line: one sentence, usually 8-20 words.
- Use character names when known. Never use "someone nearby" or vague "a man/woman" if the character matches a known appearance.
- Avoid generic starters like "A character" or "Someone" when the current panel, transcript, or nearby panels make the identity clear.
- Match character appearances from the Characters list below — if a character in the panel matches a known description, use their name.
- Treat the video title and chapter metadata as weak canon hints that can help you choose between conflicting spellings or OCR-like name variants.
- Use outside story knowledge only to choose the most plausible established name or role for this exact series when the panel/transcript already points that way.
- Never invent a new event or character purely from franchise memory.
- Vary sentence openings — no two consecutive lines start with the same word.
- No temporal fillers ("Then,", "Next,", "Soon,", "After that,") unless a real time skip occurs.
- No vague filler ("tension builds", "the story advances", "the moment hangs", "the world still feels normal").
- NEVER write visual descriptions of what the panel image looks like. Describe the STORY EVENT.
  - Bad: "A young man in a hoodie stands next to a white car."
  - Good: "Zhang Yi arrives at the parking lot, ready to make his next move."
  - Bad: "Two men unload large water bottles from a truck."
  - Good: "The supply delivery arrives just in time for Zhang Yi's stockpile."
- Do not quote raw OCR or copy broken OCR fragments into narration.
- If a panel shows a realization, question, warning, or decision, the narration must describe that exact beat.
- Each narration in this batch MUST start with a different word.
- If removing character names from a narration makes it applicable to any manga, it is too generic. Rewrite with specific plot details.
- NEVER copy or paraphrase the chapter summary — write from the specific panel content.
- If one proper name is well-established in this chapter and a conflicting rare variant appears, prefer the established name unless the panel clearly introduces a different person.

VISUAL-ONLY PANELS (no extracted text)

If a panel has no extracted text (or text is marked unavailable):
- Read the image to identify the specific action, reaction, or decision happening in that panel.
- Use character names from the Characters list — match by appearance (hair color, clothing, role).
- Each visual-only panel must describe a DIFFERENT micro-beat than adjacent panels, even if the images look similar.
- If the image shows characters from the same scene as surrounding panels, advance the story: what is the consequence, reaction, or next step visible in this panel?
- Never write the scene summary as the narration. The scene summary is background context — your line must be the specific event visible in this panel.

CONSECUTIVE PANELS IN THE SAME SCENE

When multiple consecutive panels cover the same scene:
- Each panel must describe a DISTINCT story beat: action → reaction → decision, or cause → effect → consequence.
- Do NOT repeat the same event across panels. If panel 5 says "Zhang Yi warns the group," panel 6 must describe a DIFFERENT moment (e.g., someone's reaction, the next thing said, a character's decision).
- Use the panel image to identify what is DIFFERENT about each panel, even when the setting is the same.

CONTEXT

Video title: {project_title}
Chapter: {chapter_metadata}

Chapter summary:
{chapter_summary}

Characters:
{character_dictionary}

Manually reviewed narrations from this chapter (written or corrected by a human — treat these as ground truth for character names, relationships, and story framing; apply the same naming to all other panels):
{locked_examples}

Full chapter transcript (extracted text from every panel — use this to understand the complete story arc and character names before narrating the batch below):
{chapter_transcript}

Narration already written for panels preceding this batch — DO NOT repeat these story beats:
{preceding_narrations}

PANELS

{panel_block}

Return JSON only.
