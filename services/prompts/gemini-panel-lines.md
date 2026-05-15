You are writing short panel-by-panel narration lines for a manga/manhwa recap video.

You will receive a small chunk of scenes from one chapter.

Your job is to write exactly one narration line for every listed panel number.

Return valid JSON only in this format:

{
  "panel_narrations": [
    {
      "panel": 1,
      "narration": "Short narration line for this panel."
    }
  ]
}

NARRATION STYLE

You are narrating a story to someone who is blindfolded. They cannot see the images.
Describe what HAPPENS — actions, decisions, emotions, dialogue — not what the image looks like.

THE CORE RULE: Every line MUST describe a concrete STORY EVENT.
- "She attacks him" ← story event ✓
- "He realizes the truth" ← story event ✓
- "They argue about the mission" ← story event ✓
- "The red-haired student cries" ← NOT a story event, just appearance ✗
- "A character stands there confused" ← NOT a story event, just positioning ✗
- "The dialogue continues" ← NOT a story event, too vague ✗

Cover character reactions, internal thoughts, and atmosphere, not just dialogue.
When characters speak, embed their words naturally into the narration rather than quoting them directly.
Be detailed and vivid — this narration will be used for a YouTube video, so it must hold the listener's attention.

MULTIMODAL INPUT

You may receive panel images alongside the extracted text. If the extracted text is garbled, broken, empty, or in a non-English language (e.g. Portuguese, Chinese, Korean), use the panel images to understand what happens and write accurate English narration. Read any visible text in the images directly. Do NOT attempt to translate garbled OCR yourself — describe the story event from the image context instead.

HANDLING PANELS WITH WEAK/MISSING TEXT

When a panel has no OCR text, broken text, or only image evidence:
- Focus on the ACTION: movement, combat, emotional reactions, physical states
- Describe what characters are DOING, not what they look like
- Look for body language, positioning, motion, changes in scene
- Generate CONCRETE story events, not generic dialogue templates
- NEVER fall back to vague filler like "the dialogue continues" or "the moment unfolds" when there's no dialogue
- Instead, narrate the VISUAL STORY: "She strikes first, forcing him back into the corner."

CRITICAL RULES

- Return exactly one item for every input panel.
- Keep the same panel numbers and the same order.
- Every narration line MUST be a complete English sentence with a subject and a verb. No fragments like "Mocking his efforts" or "Having been granted a second chance."
- Each narration must match the meaning of that panel's extracted text (or visible image content if text is broken).
- If a panel shows a realization, question, warning, or decision, the narration must describe that exact beat.
- ABSOLUTELY FORBIDDEN FILLER LINES (never write these):
  - "the dialogue presses forward"
  - "the conversation continues"
  - "the exchange continues"
  - "the story advances"
  - "another tense beat"
  - "the chapter continues"
  - "the next moment unfolds"
  - "the world still feels normal"
  - "by the end of the chapter"
  - "a sharp question cuts through"
  - "one pointed question makes it clear"
  - "the panel holds for a beat"
  - "the moment catches on"
  - "tension builds"
  - "the pressure keeps rising"
  - "the moment unfolds"
  - "time passes"
  - "the scene shifts"
  - "attention turns"
  - "focus changes"
  - "uncertainty instead of"
  - "through uncertainty without"
  - Any sentence that doesn't describe a concrete action, decision, or spoken content
- NEVER use "someone nearby" as a character reference. Use the character's name, or describe by role (the guard, the merchant, the stranger, another man, the other person). This phrase corrupts output.
- NEVER write visual descriptions of what the panel image looks like. Describe the STORY EVENT, not the image.
  - Bad: "A young man in a hoodie stands next to a white car."
  - Good: "Zhang Yi arrives at the parking lot, ready to make his next move."
  - Bad: "Two men unload large water bottles from a truck."
  - Good: "The supply delivery arrives just in time for Zhang Yi's stockpile."
- NEVER use appearance descriptors (hair color, clothing, physical marks, facial expressions) as primary subject identifiers. Use character names or functional roles only.
  - Bad: "The green-haired student cries with a bloody nose and a cut cheek."
  - Good: "The student's pain narrows his choices as he struggles to recover."
  - Bad: "A red-haired girl stands in the hallway."
  - Good: "She decides to confront him about the truth she just learned."
- Do not describe physical states without narrative purpose. A character's appearance only matters if it drives the plot.
  - Bad: "His face is marked by wounds and exhaustion."
  - Good: "His injuries make him reconsider whether he can still fight."
- Do not quote raw OCR unless absolutely necessary.
- Do not copy broken OCR fragments directly into narration.
- Infer the event behind the dialogue and describe that event instead.
- Do not reuse a scene-level summary for a panel if the panel text points to a more specific event.
- Use character names when they are present in the panel text or scene summary.
- Prefer specific subjects over pronouns when possible.
- Keep each narration to one sentence, usually 8-18 words.
- Vary the wording from panel to panel.
- Do NOT start consecutive narrations with the same word or phrase.
- Do NOT prepend temporal fillers like "Then,", "Next,", "Soon,", "After that," unless a real time skip occurs.

HANDLING POOR OR NON-ENGLISH OCR

- The extracted text may come from OCR of non-English comics (Portuguese, Chinese, Korean, etc.) and can be garbled, incomplete, or partially translated.
- When the extracted text is broken or nonsensical, use the chapter recap anchor and surrounding panel context to infer what the panel is showing.
- Never echo garbled OCR fragments. Write a clean English narration that describes the story event.
- If a panel's text is entirely unreadable, write a narration based on what logically happens between the previous and next panels in the scene.

DIVERSITY AND CONTINUITY

- Each narration in this batch MUST start with a different word. Do not begin two consecutive lines with the same character name.
- If removing character names from a narration makes it applicable to any manga panel, it is too generic. Rewrite it with specific plot details from the extracted text.
- These panels form a continuous scene. Each narration should logically follow the previous one, as if telling one story.
- Never use "someone nearby" as a character reference. Use the character's name if known, or describe them by role (the guard, the merchant, the stranger).

QUALITY EXAMPLES

- Bad panel narration:
  "The world still feels normal."
- Good panel narration:
  "Zhang Yi warns the contractor that dangerous enemies may come after him."

- Bad panel narration:
  "Zhang Yi regrets trusting others."
- Good panel narration:
  "Zhang Yi realizes the impossible has happened and that he is back before the disaster."

- Bad panel narration:
  "By the end of the chapter."
- Good panel narration:
  "He hurries to clear the warehouse before anyone notices what he is doing."

- Bad panel narration (visual appearance as subject):
  "The green-haired student cries with a bloody nose."
- Good panel narration (actual story event):
  "The student reels from the impact, his pain now pushing him toward a breaking point."

- Bad panel narration (generic/filler):
  "The dialogue presses forward through uncertainty."
- Good panel narration (specific story beat):
  "She demands to know if he was lying the whole time."

- Bad panel narration (caption-like):
  "A student with red hair is thrown across the hallway."
- Good panel narration (narrative action):
  "He crashes against the lockers, his body slamming hard before he drops to the ground."

- Bad panel narration (template when text is weak):
  "The dialogue presses forward through uncertainty."
- Good panel narration (action focus when text is weak):
  "He lashes out with a combination attack, pressing his advantage as the other fighter retreats."

- Bad panel narration (generic filler for weak evidence):
  "The moment continues as the conversation unfolds."
- Good panel narration (concrete action for weak evidence):
  "She pivots on her heel and counters his next move, seizing the opening he leaves."

CONTEXT

Video project title:
{project_title_context}

Chapter metadata:
{chapter_metadata}

Chapter recap anchor:
{chapter_summary}

Scene chunk:
{scene_panel_block}

Return JSON only.
