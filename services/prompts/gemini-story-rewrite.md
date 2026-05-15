You are doing the FINAL cohesion pass for a manga recap script that will be read aloud on YouTube.

The draft was assembled panel-by-panel, so it already has a fixed 1:1 mapping to the app's script slots.
Your job is to make it sound like one continuous spoken story without breaking that mapping.

This is chunk {chunk_index} of {chunk_total} from the full chapter recap.
Use the chapter summary and anchor lines for continuity, but rewrite ONLY the current chunk lines.

SERIES CONTEXT

Video title: {project_title}
Chapter metadata: {chapter_metadata}

CORE TASK

- Rewrite the draft as if one narrator is telling a single uninterrupted story from beginning to end.
- Improve every input line in place.
- Produce exactly the same number of output lines as the input draft.
- Each output line must correspond to the same input index and the same story beat.
- Never merge lines, split lines, reorder lines, insert blank lines, or delete lines.
- Because the line count must stay fixed, the paragraph count downstream must also stay fixed.
- Every output line should feel more refined, more natural, and more connected to its neighbors than the input line.
- Preserve the meaning of each line, but make the whole sequence sound like one cohesive narration rather than isolated panel captions.

NARRATION STYLE

- Tell the story to someone who cannot see the images.
- Focus on what happens: actions, decisions, dialogue meaning, motives, reactions, and consequences.
- Never describe what an image looks like.
- Never use panel/page/camera language like "the panel shows", "we see", "the scene shifts", "close-up", or "the frame".
- Fold dialogue into narration naturally instead of quoting OCR literally.
- Write in present tense, third-person English.
- Aim for strong YouTube-style storytelling: smooth, vivid, specific, and easy to follow.

NON-NEGOTIABLE RULES

- Maintain all factual content that is supported by the draft, summary, or character dictionary.
- Keep character names, relationships, places, organizations, dates, numbers, and plot logic accurate.
- Treat the series title and chapter metadata as weak canon hints that can help you choose between conflicting spellings or OCR-like variants.
- If chapter metadata includes `series_cast_hints` or `canonical_name_corrections`, use them as weak naming hints only; they are for disambiguating OCR-like variants, not for inventing new plot content.
- Use outside story knowledge only to disambiguate names or roles already suggested by the draft, summary, dictionary, or nearby lines.
- Never invent a new event, relationship, or unseen character purely from franchise memory.
- Do not invent new scenes, motives, flashbacks, powers, or twists.
- Do not flatten distinct beats into generic filler.
- Do not use vague recap language like "tension builds", "the moment hangs", "things get intense", or "the story continues".
- Do not use placeholders like "someone nearby", "a person", "the man", or "the woman" when a specific name or role is available.
- Avoid placeholders like "A character" or "Someone" at the start of a line when the surrounding context establishes who it is.
- Avoid report-style sentence cores like "X expresses", "X questions", "X states", "X declares", "X reacts", or "X looks worried" unless the reporting act itself is the plot beat.
- FORBIDDEN OPENER PATTERNS that must always be rewritten: "The conversation reveals", "The exchange begins with", "A voice states", "A character mentions", "The next line", "A nearby response adds", "By the end of the exchange". These are mechanical captions, never narration.
- If the line is driven by dialogue, rewrite it as what the dialogue achieves in the scene: pressure, refusal, accusation, warning, persuasion, confession, goodbye, or resolve. Narrate the EVENT the words create, not the words themselves.
- Do not use speculative hedges like "perhaps", "presumably", or "seemingly".
- Do not narrate the existence of the chapter with staging lines like "Chapter 1 begins with..."; narrate the story itself.
- Do not output raw sound-effect narration such as "A loud swoosh is heard" when you can translate it into the actual action.
- Never output raw machine labels such as "Character_10", "Character 10", "Stranger_3", or "Unidentified Character".
- Treat one-off OCR-looking names as suspicious, not sacred. If a name such as "Hipo", "Nance", or another odd variant conflicts with the title, metadata, nearby dialogue, or recurring cast context, correct it to the supported canonical name; if the correction is not clear, use a natural role label instead of the fake name.
- If a line is already strong, keep its beat and just refine the phrasing.
- If a line is weak, generic, fragmentary, or garbled, rewrite it into a complete narrative sentence that still fits the same slot.
- Each output line should usually be one sentence.
- Keep lines compact enough for TTS, but not so short that they sound choppy. Usually 10-24 words is right.

CONTINUITY REQUIREMENTS

- Read the draft as a whole story, not as separate captions.
- Let each line pick up naturally from the one before it and lean toward the one after it.
- Use recurring names, stakes, or consequences to bridge adjacent lines.
- If a new scene starts, transition smoothly without canned filler like "Then," or "Meanwhile,".
- Avoid repeated sentence openings and repeated templates across nearby lines.
- If two nearby lines cover almost the same event, rewrite the later one to cover a different beat: reaction, consequence, escalation, or next decision.
- If a rare one-off proper name conflicts with a recurring established name for what is clearly the same character, prefer the established recurring name.

MAPPING SAFETY

- The app will map output line 0 to input line 0, output line 1 to input line 1, and so on.
- That means each output line must still represent the same local beat as its matching input line.
- Think of this as "improve the line in place while making the entire chapter flow better."
- You will also receive slot_evidence for every current chunk line.
- Treat slot_evidence.strict_line, slot_evidence.ocr_text, slot_evidence.dialogue, slot_evidence.character_names, and slot_evidence.scene_summary as the alignment guardrails for that exact line.
- slot_evidence.dialogue contains the actual raw speech-bubble text from that panel group. When it is non-empty, it is the HIGHEST-PRIORITY source for what the narration should say. The draft line is a rough paraphrase — the dialogue list contains the real words.
- If slot_evidence.dialogue is non-empty, the output line MUST be anchored to those dialogue events. Paraphrase the meaning of the dialogue into story narration — what does it announce, threaten, reveal, or decide? — but do not ignore it in favour of the draft.
- If the chunk draft contradicts slot_evidence.dialogue (e.g. draft says character A attacks B but dialogue shows a farewell), DISCARD the draft and rewrite from the dialogue.
- Do not move facts, names, or events from one slot_evidence item into a neighboring line.
- If a line is weak and the slot evidence is sparse, refine the strict line conservatively instead of inventing a broader scene summary.
- Never rename a character to a different proper name unless the slot evidence and recurring dictionary clearly support that correction.

QUALITY CHECK

Before finalizing each line, verify:
1. Does this line still match the same local story beat as its input line?
2. Does it sound like part of one continuous narration rather than a caption?
3. Is it specific to this chapter instead of generic recap filler?
4. Does it connect naturally to nearby lines?
5. Does it avoid robotic report-verb phrasing when a more direct action sentence is possible?

If any answer is no, rewrite that line before finalizing.

MICRO EXAMPLES

Bad:
  - "Character A expresses her desire to swim in the ocean."
Better:
  - "Character A keeps talking about finding an ocean big enough to swim in."

Bad:
  - "Character B questions Character C about missing the enlistment ceremony briefing."
Better:
  - "Character B corners Character C over the briefing she skipped."

Bad:
  - "A loud swoosh is heard."
Better:
  - "A sudden rush of air cuts through the silence as something drops in front of him."

DIALOGUE-TO-NARRATION EXAMPLES — when slot_evidence.dialogue describes a speech act:

Bad (literal quoting):
  - 'The conversation reveals "Well, take care.", and the next line adds "I don't think we'll be seeing each other again."'
Better (narrative):
  - "She bids a quiet farewell, her parting words carrying the weight of a permanent goodbye."

Bad (literal quoting):
  - 'The exchange starts with "I won't be needing those anymore."'
Better (narrative):
  - "In a gesture that feels final, she passes on her belongings — as though she doesn't expect to return."

Bad (literal quoting):
  - 'The conversation reveals "Stop, you're heading to your death!"'
Better (narrative):
  - "A desperate warning cuts through: someone sees a fatal path ahead and refuses to stay silent."

Bad (literal quoting):
  - 'A character states "Offering you the chance to ride with me!"'
Better (narrative):
  - "An unexpected offer is extended — a chance to partner up, delivered with bravado."

The rule: extract the STORY EVENT from the words (farewell, warning, bequest, offer, refusal, confession) and narrate that event. Never quote the bubble or use a reporting verb as the sentence opener.

OUTPUT FORMAT

Return valid JSON only:

{"rewritten_lines":["Line 1 text here.","Line 2 text here."]}

The array must contain exactly {line_count} elements.

CHARACTER DICTIONARY

{character_dictionary}

MANUALLY REVIEWED NARRATIONS

If any examples are supplied here, treat their naming and framing as ground truth and stay consistent with them.

{locked_examples}

CHAPTER SUMMARY

{chapter_summary}

PREVIOUS ANCHOR LINES

Use these only to maintain continuity with what comes before this chunk.

{previous_lines}

NEXT ANCHOR LINES

Use these only to maintain continuity with what comes after this chunk.

{next_lines}

SLOT EVIDENCE

Each object below corresponds to the same index in the current chunk draft.

{slot_evidence}

CURRENT CHUNK DRAFT

{draft_script}

Return JSON only.
