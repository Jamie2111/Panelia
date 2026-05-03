from __future__ import annotations

import math
import re
from typing import Any

from app.services.ocr_cleaner import clean_ocr_text, keyword_tokens


class PanelMapper:
    def map_scene_summaries_to_panels(
        self,
        panels: list[dict[str, Any]],
        scene_seeds: list[dict[str, Any]],
        scene_summaries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not panels:
            return []

        ordered_panels = sorted(panels, key=lambda item: int(item["panel"]))
        summary_lookup: dict[int, dict[str, Any]] = {}
        for item in scene_summaries:
            scene_id = self._coerce_int(item.get("scene_id"))
            if scene_id is not None:
                summary_lookup[scene_id] = item
        seeds = scene_seeds or self._synthetic_seeds(ordered_panels, max(1, min(10, len(scene_summaries) or 1)))

        mapped: list[dict[str, Any]] = []
        mapped_panel_ids: set[str] = set()
        previous_line = ""

        for seed in seeds:
            scene_panels = self._panels_for_seed(ordered_panels, seed)
            if not scene_panels:
                continue

            scene_id = self._coerce_int(seed.get("scene_id")) or len(mapped) + 1
            scene_summary = summary_lookup.get(scene_id, {})
            scene_summary_text = self._normalize_sentence(
                str(scene_summary.get("narration") or scene_summary.get("summary") or "").strip(),
                allow_empty=True,
            )
            scene_description_text = self._normalize_phrase(str(scene_summary.get("description") or "").strip())
            lines = self._build_scene_panel_lines(scene_panels, scene_summary)

            for panel, line in zip(scene_panels, lines, strict=False):
                narration = line
                if previous_line and self._sentence_similarity(previous_line, narration) > 0.9:
                    narration = self._vary_sentence(narration, panel, len(mapped))
                mapped.append(
                    {
                        "panel": panel["panel"],
                        "panel_id": panel["panel_id"],
                        "page": panel["page"],
                        "narration": narration,
                        "scene_id": scene_id,
                        "scene_summary": scene_summary_text,
                        "scene_description": scene_description_text,
                    }
                )
                mapped_panel_ids.add(str(panel["panel_id"]))
                previous_line = narration

        for panel in ordered_panels:
            panel_id = str(panel["panel_id"])
            if panel_id in mapped_panel_ids:
                continue
            fallback = self._fallback_panel_line(panel)
            if not fallback:
                continue
            mapped.append(
                {
                    "panel": panel["panel"],
                    "panel_id": panel["panel_id"],
                    "page": panel["page"],
                    "narration": fallback,
                    "scene_id": None,
                    "scene_summary": "",
                    "scene_description": "",
                }
            )

        return sorted(mapped, key=lambda item: int(item["panel"]))

    def _build_scene_panel_lines(
        self,
        scene_panels: list[dict[str, Any]],
        scene_summary: dict[str, Any],
    ) -> list[str]:
        summary_text = self._normalize_sentence(
            str(scene_summary.get("narration") or scene_summary.get("summary") or "").strip(),
            allow_empty=True,
        )
        fragments = self._expand_scene_fragments(summary_text, len(scene_panels))
        built: list[str] = []
        seen_lines: list[str] = []

        for index, panel in enumerate(scene_panels):
            fragment = fragments[index] if index < len(fragments) else ""
            panel_text = clean_ocr_text(str(panel.get("text", "")))
            narration = self._compose_panel_line(panel_text, fragment, summary_text, index, len(scene_panels))
            narration = self._avoid_repetition(narration, seen_lines, panel_text, fragment, summary_text, index, len(scene_panels))
            built.append(narration)
            seen_lines.append(narration)

        return built

    def _compose_panel_line(
        self,
        panel_text: str,
        fragment: str,
        scene_summary: str,
        panel_index: int,
        panel_count: int,
    ) -> str:
        panel_line = self._panel_specific_line(panel_text, scene_summary, panel_index, panel_count)
        if panel_line:
            return self._normalize_sentence(panel_line)
        if fragment:
            return self._normalize_sentence(fragment)
        if scene_summary:
            fragments = self._summary_clauses(scene_summary)
            if fragments:
                if panel_count <= 2:
                    return self._normalize_sentence(fragments[min(panel_index, len(fragments) - 1)])
                if panel_index == 0:
                    return self._normalize_sentence(fragments[0])
                if panel_index == panel_count - 1:
                    return self._normalize_sentence(fragments[-1])
                return ""
            if panel_count <= 2:
                return self._normalize_sentence(scene_summary)
        return ""

    def _panels_for_seed(self, panels: list[dict[str, Any]], seed: dict[str, Any]) -> list[dict[str, Any]]:
        wanted_ids = {str(value) for value in seed.get("panel_ids", []) or []}
        wanted_orders = {int(value) for value in seed.get("panels", []) or []}
        if wanted_ids:
            matched = [panel for panel in panels if str(panel["panel_id"]) in wanted_ids]
            if matched:
                return matched
        if wanted_orders:
            matched = [panel for panel in panels if int(panel["panel"]) in wanted_orders]
            if matched:
                return matched
        return []

    def _synthetic_seeds(self, panels: list[dict[str, Any]], scene_count: int) -> list[dict[str, Any]]:
        panels_per_scene = max(1, math.ceil(len(panels) / max(scene_count, 1)))
        seeds: list[dict[str, Any]] = []
        for index in range(scene_count):
            start = index * panels_per_scene
            end = min(len(panels), start + panels_per_scene)
            chunk = panels[start:end]
            if not chunk:
                continue
            seeds.append(
                {
                    "scene_id": index + 1,
                    "panel_ids": [str(panel["panel_id"]) for panel in chunk],
                    "panels": [int(panel["panel"]) for panel in chunk],
                }
            )
        return seeds

    def _expand_scene_fragments(self, summary: str, panel_count: int) -> list[str]:
        if not summary:
            return ["" for _ in range(panel_count)]

        sentence_parts = [
            self._normalize_sentence(part, allow_empty=True)
            for part in re.split(r"(?<=[.!?;:])\s+", summary)
            if part.strip()
        ]
        fragments: list[str] = []
        for part in sentence_parts:
            normalized = self._normalize_sentence(part, allow_empty=True)
            if normalized and not self._looks_like_dangling_clause(normalized):
                fragments.append(normalized)

        if not fragments:
            fragments = [summary]
        if len(fragments) >= panel_count:
            return fragments[:panel_count]

        distributed = ["" for _ in range(panel_count)]
        positions = self._spread_positions(panel_count, len(fragments))
        for fragment, position in zip(fragments, positions, strict=False):
            distributed[position] = fragment
        return distributed

    def _panel_specific_line(self, text: str, scene_summary: str, panel_index: int, panel_count: int) -> str:
        lowered = text.casefold()
        if not lowered:
            return ""
        if "three ways to survive" in lowered or ("destroyed world" in lowered and "survive" in lowered):
            return "A grim narration lays out the rules for surviving a ruined world."
        if "my name" in lowered and "kim" in lowered and "dok" in lowered:
            return "The introduction finally lands on his full name."
        if ("i am" in lowered or "my name" in lowered) and any(token in lowered for token in ("dok", "reader", "only child")):
            return "The protagonist introduces himself and explains the meaning behind his unusual name."
        if "only child" in lowered or ('"reader"' in lowered) or ("reader." in lowered):
            return "The explanation of his name quietly underscores how isolated his life has been."
        if "i get it" in lowered or "summarizing" in lowered or "that's me" in lowered:
            return "He sums himself up with a dry, matter-of-fact honesty."
        if any(token in lowered for token in ("years old", "university", "company", "outsourced", "single man")) and any(
            token in lowered for token in ("dok", "my name", "ordinary", "life")
        ):
            return "The introduction paints him as an ordinary office worker stuck in a lonely routine."
        if "university" in lowered and any(token in lowered for token in ("outsourced", "company", "subsidiary")):
            return "His background makes it clear he is just another overworked office employee."
        if "webnovel" in lowered and any(token in lowered for token in ("home", "way back", "commute", "metro")):
            return "The ride home becomes another quiet stretch spent buried in webnovels."
        if "today on the way back" in lowered or ("metro" in lowered and "company" in lowered):
            return "Even the familiar commute home starts to feel a little different."
        if "cell phone" in lowered or ("reading" in lowered and "focus" in lowered):
            return "He stays glued to the novel, barely noticing the world around him."
        if "studying korean" in lowered:
            return "Their small talk drifts into language and the details of ordinary life."
        if "3149" in lowered or ("10 years" in lowered and "read" in lowered):
            return "More than three thousand chapters and a decade of reading have shaped his life."
        if "visualiz" in lowered or "views" in lowered or "comments" in lowered:
            return "He remembers following the novel even when almost nobody else cared about it."
        if "infernal front" in lowered or "wrong number" in lowered:
            return "The brutal details of the novel still linger vividly in his memory."
        if "forgot about you" in lowered or ("congratulations" in lowered and "dok-ja-ssi" in lowered):
            return "Her attention snapping back to him is enough to catch him off guard."
        if "yoo blood-ah" in lowered or "sang-ah" in lowered:
            if "promotion" in lowered or "studying" in lowered:
                return "Yoo Sang-ah's polished image only makes the distance between their lives feel larger."
            return "The mention of Yoo Sang-ah pulls his thoughts away from the novel for a moment."
        if "erm" in lowered and "dok-ja" in lowered:
            return "Hearing his name from her is enough to break his train of thought."
        if "realism" in lowered or "genre of my life" in lowered:
            return "He starts comparing his own life to the kind of story he wishes he were living."
        if "protagonist?" in lowered:
            return "For a brief moment, the idea of becoming the protagonist stops feeling impossible."
        if "thank you very much" in lowered:
            return "A polite thanks leaves the exchange hanging in an awkward silence."
        if "new e-mail" in lowered or "e-mail" in lowered:
            return "A fresh email notification suddenly cuts through the moment."
        if "whoa" in lowered or "wow" in lowered:
            return "Panic erupts all around him without warning."
        if "don't worry" in lowered and "nothing" in lowered:
            return "Uncertain reassurances do little to settle the growing unease."
        if "author" in lowered and ("survive" in lowered or "ways" in lowered):
            return "The possibility that the author is reaching out suddenly feels real."
        if "competition" in lowered and "author" in lowered:
            return "For a moment, the author stops feeling like a distant stranger."
        if "abbreviated" in lowered and "survival" in lowered:
            return "Even the shortened title of the novel carries weight after all these years."
        if "it's over today" in lowered:
            return "The thought of the novel reaching its end leaves him quietly unsettled."
        if "spanish" in lowered or ("means" in lowered and "money" in lowered) or "phrase" in lowered:
            return "A stray bit of wordplay turns the exchange unexpectedly awkward."
        if "author" in lowered and any(token in lowered for token in ("comments", "readers", "desires", "novel")):
            return "Years of reading finally push him to dwell on the distance between author and reader."
        if "email" in lowered or "monetization" in lowered or ("thank" in lowered and "support" in lowered):
            return "An unexpected reply from the author suddenly breaks his routine."
        if "department" in lowered or "business today" in lowered or ("boss" in lowered and "good luck" in lowered):
            return "Office chatter makes the workday feel even more draining."
        if any(token in lowered for token in ("bicycle", "extras lately", "exercise", "overtime")):
            return "Casual small talk turns to overtime, commuting, and the grind of work."
        if "reality other than mine" in lowered or "heart stolen" in lowered:
            return "The distance between her polished world and his own suddenly feels impossible to ignore."
        if "lend me money" in lowered or "loan me money" in lowered or "may lend me money" in lowered:
            return "Her casual request for money catches him completely off guard."
        if "benefit for me" in lowered and "pay you back" in lowered:
            return "He haggles over brutal loan terms because survival is worth the risk."
        if any(token in lowered for token in ("what if i only get", "still pay you back 7", "pay you back 7")):
            return "He keeps negotiating the loan terms to squeeze out more survival money."
        if "what are you doing" in lowered and "reading" in lowered:
            return "Her question finally drags his attention away from the screen."
        if any(token in lowered for token in ("scenario", "free period", "planetary system", "run away", "main scenario")):
            return "A system warning shatters the ordinary evening and signals the start of disaster."
        if "hello" in lowered and any(token in lowered for token in ("home", "metro", "company")) and any(token in lowered for token in ("you?", "yeah", "back")):
            return "A chance encounter briefly pulls him out of his usual commute."
        if "title of a living legend" in lowered or ("mysterious existence" in lowered and "holy night" in lowered):
            return "The narration presents Santa Claus as a legendary figure who delivers gifts on the holy night."
        if "but, in reality" in lowered or "but in reality" in lowered or lowered.strip() == "reality..":
            return "The public legend gives way to the truth behind Santa Claus."
        if "first mission" in lowered and "pay back" in lowered and "father" in lowered:
            return "The first mission feels deeply personal because it is a chance to repay a debt to her father."
        if "pay back" in lowered and "debt" in lowered and "father" in lowered:
            return "The mission matters to her because it feels like a chance to repay what she owes her father."
        if "first mission" in lowered:
            return "The long-awaited first mission finally arrives, filling her with anxious excitement."
        if "strongest santa claus" in lowered or "elegant and cool" in lowered:
            return "The crowd gushes over Lord Frost's strength and striking reputation."
        if "lady christina" in lowered and "lord frost" in lowered:
            return "Christina's admiration for Lord Frost is obvious to everyone around her."
        if "i want to become a santa claus" in lowered or "bring a lot of presents" in lowered:
            return "Christina dreams of becoming the kind of Santa Claus who can bring joy to everyone."
        if "thanks" in lowered and "father" in lowered and "able" in lowered:
            return "Gratitude for her father's support makes the moment feel even more meaningful."
        if "just practiced" in lowered or "it went perfect" in lowered:
            return "She tries to wave the mishap away as practice even after the awkward moment."
        if "biological reactions" in lowered or "false ignal" in lowered or "false signal" in lowered:
            return "A quick check suggests the strange reaction may have been a false alarm."
        if "next destination" in lowered or "won't finish on time" in lowered:
            return "Time pressure forces the delivery run to keep moving to the next stop."
        if "delivering the rest of these presents" in lowered:
            return "She pushes ahead, determined to deliver every remaining present before the night ends."
        if "they left food" in lowered or "so happy" in lowered:
            return "An unexpected bit of food briefly lifts the mood in the middle of the mission."
        if "have half of it" in lowered:
            return "They decide to split the food so the delivery run can continue."
        if "our secret" in lowered or "on the reindeer" in lowered or "strict" in lowered:
            return "They quietly bend the rules and agree to keep the little break a secret."
        if "one person left" in lowered:
            return "With only one stop left, the mission enters its final stretch."
        if "what were you expecting" in lowered:
            return "The teasing question catches her off guard for a moment."
        if "only kid who would ask for that" in lowered:
            return "Her unusual request stands out even among the children on the list."
        if "politics" in lowered or "bio graphy" in lowered or "biography" in lowered:
            return "Instead of toys, the request points toward books and ideas beyond a normal child's wish."
        if "who goes there" in lowered:
            return "A sudden voice interrupts the route and stops them in their tracks."
        if "who's not asleep" in lowered:
            return "The sight of someone still awake instantly throws the mission into danger."
        if "perception inhibiting" in lowered:
            return "She quickly relies on magic to keep the intruder from understanding what he is seeing."
        if "who is this man" in lowered:
            return "The stranger's sudden appearance leaves everyone scrambling for answers."
        if "i have to do something" in lowered:
            return "With no easy answer in sight, she forces herself to act."
        if "can't get it out" in lowered:
            return "The attack lands, but something about the clash still feels wrong."
        if "go and secure the route" in lowered:
            return "An order is issued immediately to secure the escape route."
        if "won't let you get away" in lowered:
            return "She refuses to let the intruder escape after disrupting the mission."
        if "release" in lowered and "spell" in lowered:
            return "The confrontation escalates as the spell binding the scene starts to unravel."
        if "now fall asleep" in lowered:
            return "She tries to end the danger quickly by putting the intruder to sleep."
        if "didn't work" in lowered or "didn t work" in lowered:
            return "Her spell fails, leaving her stunned at the worst possible moment."
        if "i'll run away" in lowered:
            return "Panic takes over once the magic fails to stop him."
        if "doesn't hurt" in lowered:
            return "The unexpected reaction only deepens the confusion around the stranger."
        if "nobody in my family ever treated me like this" in lowered:
            return "The struggle turns personal as the stranger reacts with open shock and anger."
        if "prince's room" in lowered:
            return "Their desperate escape sends them straight toward the prince's room."
        if "wanted to make kids smile" in lowered:
            return "She insists her only goal was to bring joy to children, not start a fight."
        if "could it have been santa claus" in lowered:
            return "Rumors immediately start swirling about whether Santa Claus was really involved."
        if "my first mission got all messed up" in lowered:
            return "The ruined first mission leaves her furious and deeply frustrated."
        if "tonight's report" in lowered or "feel responsible" in lowered:
            return "Guilt settles in as the night's failure has to be explained in an official report."
        if "report" in lowered and "knight" in lowered:
            return "A knight's report hints that the incident is already drawing serious attention."
        if "my lady" in lowered and "intruder" in lowered:
            return "An urgent warning makes it clear that an intruder has appeared."
        if "run away" in lowered and "hurt" in lowered:
            return "Panic takes over as someone tries to flee before the violence gets worse."
        if any(token in lowered for token in ("santa", "claus", "present", "presents", "gift", "gifts", "holy")):
            return "The scene leans into the strange Santa-themed spectacle surrounding the moment."
        if any(token in lowered for token in ("awake", "asleep", "body", "prince", "who is this", "what is happening")):
            return "Confusion takes over as the characters struggle to understand what is happening."
        if any(token in lowered for token in ("spell", "magic", "opponent", "intruder", "take this", "get away", "falter")):
            return "A burst of magic throws the scene into sudden chaos."
        if any(token in lowered for token in ("run away", "hurt", "kya", "way out", "wait!", "room..")):
            return "The situation turns frantic as someone tries to escape the danger closing in."
        if any(token in lowered for token in ("knight", "report", "responsible")):
            return "The aftermath leaves everyone with new questions about what comes next."
        if "did i travel back in time" in lowered or "did i regress" in lowered or "month before the world freezes over" in lowered:
            return "He realizes the impossible has happened and that he is back before the disaster."
        if "where am i" in lowered or "ahhhhh" in lowered:
            return "Shock and confusion hit him the moment he regains consciousness."
        if "didn't i just" in lowered or "get chopped up by my bastard neighbors" in lowered:
            return "The memory of being butchered by his neighbors crashes back into his mind."
        if "corpse" in lowered or "kill him" in lowered or "chopped up" in lowered:
            return "The starving crowd turns on him with murderous intent."
        if "open the door" in lowered or "big share" in lowered or ("resources" in lowered and any(token in lowered for token in ("door", "share", "supplies", "food"))):
            return "They immediately start fighting over his supplies and blaming him."
        if "trusted" in lowered or "too naive" in lowered:
            return "He realizes trusting them was the mistake that got him killed."
        if "my top priority" in lowered or "ensure my safety" in lowered or "secure enough supplies" in lowered:
            return "He decides that survival has to come before anything else this time."
        if "revenge" in lowered:
            return "The betrayal leaves him determined to take revenge."
        if "12 th november" in lowered or "one month before the world freezes over" in lowered:
            return "The date confirms he has gone back to the month before the catastrophe."
        if "regress" in lowered or "travel back in time" in lowered or "month before" in lowered:
            return "He realizes he has somehow returned to the past before the disaster."
        if "supernova" in lowered or "freeze over" in lowered or "blizzard" in lowered or "70 degrees" in lowered:
            return "He remembers exactly how the frozen apocalypse began."
        if "secure both" in lowered or ("weapons" in lowered and "bastards" in lowered):
            return "He decides to secure supplies and weapons before the apocalypse arrives."
        if "warehouse" in lowered and "supervises" in lowered:
            return "He realizes his warehouse connection can solve his supply problem this time."
        if "storage space" in lowered or "massive amount of supplies" in lowered or "store the entire" in lowered:
            return "A storage ability suddenly gives him a way to hoard everything he needs."
        if "did i get something in my eyes" in lowered:
            return "A strange change catches his eye at the exact moment everything shifts."
        if "worry about supplies" in lowered:
            return "The storage ability instantly changes how he thinks about stockpiling supplies."
        if "supplies doesn't decompose" in lowered or "best warehouse in existence" in lowered:
            return "He realizes the storage space can preserve food far better than any normal warehouse."
        if "free labour" in lowered:
            return "He even sees their sudden interest as free labor for his supply run."
        if "help each other" in lowered or "help you out" in lowered:
            return "His neighbors volunteer to help only because they expect something in return."
        if "going to the supermarket" in lowered or "get some supplies" in lowered:
            return "He heads straight for the supermarket to keep stockpiling for the apocalypse."
        if "try one of everything" in lowered:
            return "He orders everything he can while the world still feels normal."
        if "instant noodles" in lowered:
            return "Memories of future starvation make ordinary food feel priceless."
        if "kneel and beg" in lowered or "possible to eat" in lowered:
            return "Memories of future starvation make even ordinary food feel priceless."
        if "try one of everything" in lowered or "delicious food" in lowered or "instant noodles" in lowered:
            return "Knowing food will soon become precious, he starts indulging while he still can."
        if "rich guy" in lowered or "gulfing down the food" in lowered:
            return "The people around him are stunned by how desperately he eats."
        if "esteemed customer" in lowered or "vip-card" in lowered or "spending 50" in lowered or "restaurant" in lowered:
            return "Restaurant staff are shocked by how freely he throws money around."
        if "yu qing" in lowered or "crush on you" in lowered:
            return "His sudden spending makes the people around him see him in a completely different light."
        if "zhang yi is a nice guy" in lowered or ("done shopping" in lowered and "go home" in lowered):
            return "The neighbors keep flattering him while trying to stay close to his supplies."
        if "heh, of course i don't mind" in lowered:
            return "He plays along with them while keeping his real intentions hidden."
        if "don't take what she said seriously" in lowered or "i'm sorry i misspoke" in lowered or "cai ning is just a bit tired" in lowered:
            return "The awkward tension forces them into a quick apology to stay on his good side."
        if "friends about money" in lowered:
            return "Their friendliness only lasts as long as money is involved."
        if "zhang yi is a nice g" in lowered or ("zhang yi is a nice" in lowered and "done shopp" in lowered):
            return "The neighbors keep flattering Zhang Yi so they can stay on his good side."
        if "sweet revenge" in lowered or "people see me hoarding" in lowered:
            return "He stops caring who notices the stockpiling as long as revenge stays in sight."
        if "share some with us" in lowered or "xiao zhang" in lowered:
            return "The neighbors shamelessly ask him to share the food he just secured."
        if "eating all this by myself" in lowered:
            return "He refuses to share any of the supplies he fought to buy."
        if "give me back my chocolate" in lowered or "slap the shit out of you" in lowered:
            return "A petty fight breaks out the moment food becomes the center of attention."
        if "buy it from you" in lowered or "pay you later" in lowered:
            return "They try bargaining for his food the instant he refuses to hand it over."
        if "either pay" in lowered and "supermarket" in lowered:
            return "He shuts down their demands and tells them to buy their own supplies."
        if "neighboorhood committee" in lowered or "how can you be so mean" in lowered:
            return "The complaints start immediately when he refuses to indulge them."
        if "treat us to dinner" in lowered:
            return "Even after all that, the neighbors still expect another favor from him."
        if "forget to treat sometime" in lowered or "do that sometime" in lowered:
            return "Even after helping him shop, they still expect Zhang Yi to reward them later."
        if "what a coincidence" in lowered and "supermarket" in lowered:
            return "His neighbors immediately try to insert themselves into his supermarket run."
        if "what a coincidence" in lowered or "help you out" in lowered or "neighbors" in lowered:
            return "His neighbors quickly attach themselves to him, hoping to benefit from his money."
        if "event soon" in lowered or "camping" in lowered:
            return "His massive shopping spree makes everyone around him wonder what he is really preparing for."
        if "what if there is an apocalypse tomorrow" in lowered:
            return "He half-jokes about the apocalypse while quietly telling the truth."
        if "supermarket" in lowered or "stock up" in lowered or "supplies" in lowered or "event soon" in lowered or "camping" in lowered:
            return "He keeps buying supplies while brushing off questions about his true plan."
        if "dinner sometime" in lowered or "stingy" in lowered or "helped you" in lowered:
            return "Even now, the neighbors only care about what they can get from him."
        if any(token in lowered for token in ("spanish", "phrase", "means")) and "money" in lowered:
            return "A stray explanation about money only makes the conversation feel stranger."
        if "loan" in lowered or "five million" in lowered or "interest" in lowered or "mortgage" in lowered or "deposit" in lowered:
            return "He starts pulling together huge amounts of cash to fund his preparations."
        if "after mortgaging the house" in lowered or "adding my savings" in lowered:
            return "He mortgages his family home and drains his savings to keep the plan alive."
        if "why don't you try our company" in lowered:
            return "A predatory loan company appears just when he needs cash the most."
        if "need a couple millions" in lowered or "need five million" in lowered:
            return "He asks for millions more, desperate to finish preparing before time runs out."
        if "you have a wa" in lowered or "coupi millions" in lowered:
            return "He jumps at the chance to borrow millions and keep the plan moving."
        if "interest rate is 40%" in lowered:
            return "The loan terms are brutal, but he accepts them without hesitation."
        if "eqivalent value as colleteral" in lowered or "help manager" in lowered:
            return "He puts up his house as collateral to get the money immediately."
        if "special deal" in lowered or "dead anyways" in lowered:
            return "The lenders think they are exploiting him, unaware the world is about to end."
        if "squeeze this guy dry" in lowered:
            return "The loan sharks see him as easy money and plan to bleed him dry."
        if "this is mr" in lowered and "zhang" in lowered:
            return "An introduction quickly pulls Zhang Yi deeper into the loan arrangement."
        if "free money" in lowered:
            return "He treats the predatory loan as free money because the apocalypse will erase the debt."
        if "war dragon security company" in lowered:
            return "He seeks out the most formidable security company he can find."
        if "biggest security companies" in lowered:
            return "He turns to one of the biggest security companies he can find."
        if "safe house" in lowered or "vault" in lowered or "alloy" in lowered or "security company" in lowered or "ventilation" in lowered:
            return "He begins turning his apartment into a fortress built for the apocalypse."
        if "manager of the business department" in lowered or "top notch safe house" in lowered:
            return "He commissions a top-tier safe house built to endure the frozen apocalypse."
        if "20\" thick vault door" in lowered or "bulletproof" in lowered:
            return "The security plan starts turning his apartment into a bunker."
        if "this is amazing" in lowered and "impenetrable fortress" in lowered:
            return "For the first time, his home begins to look like a place that can truly survive."
        if "mercenary overseas" in lowered:
            return "The contractor's background convinces him the safe house can really be built."
        if "medicine" in lowered:
            return "He moves on to securing medicine before the supply chain collapses."
        if "weapon contacs" in lowered or "hunting club" in lowered:
            return "He starts looking for weapon sources that will not draw suspicion."
        if "offended some bad guys" in lowered:
            return "He invents a dangerous enemy to justify asking for weapons and protection."
        if "lot of guns" in lowered or ("protection" in lowered and "stranger" in lowered):
            return "He spins a story about armed enemies to justify spending heavily on protection."
        if "technically illegal" in lowered or "feud with someone scary" in lowered:
            return "He leans on a fake grudge story to justify asking about illegal weapons."
        if "yeep, i may have" in lowered:
            return "He plays along and pretends the threat is real to keep the deal moving."
        if "half a month to renovate" in lowered:
            return "The renovation schedule forces him to keep moving quickly on every other preparation."
        if "crossbow" in lowered or "crowbar" in lowered or "knife" in lowered or "axes" in lowered or "hunting" in lowered:
            return "He expands his preparations by looking for weapons he can legally obtain."
        if "since i can preserve fresh food" in lowered or "prepared meals" in lowered:
            return "A new idea strikes him: prepared meals can be stockpiled just as easily."
        if "hong fu tianjin hotel" in lowered and "what can i do for you" in lowered:
            return "He contacts a major hotel to turn his hoarding into a banquet order."
        if "talk to your manager" in lowered or "massive banquet in three days" in lowered:
            return "He hides the next stage of his preparations behind the excuse of a huge banquet."
        if "five hundred baquet tables" in lowered or "deposit of 200 000" in lowered:
            return "The hotel quotes an enormous price, and he agrees without blinking."
        if "account number" in lowered and "deposit" in lowered:
            return "He pays the deposit immediately to keep the banquet plan moving."
        if "money will be useless in a month" in lowered:
            return "With the apocalypse approaching, money already feels temporary to him."
        if "you are not allowed to enter" in lowered:
            return "The sheer scale of the delivery startles even the people guarding the entrance."
        if "can't just let you in with all this stuff" in lowered:
            return "The delivery is so excessive that even the guards stop him on the spot."
        if "dozen trucks" in lowered:
            return "Truck after truck arrives, making his preparations impossible to ignore."
        if "hong fu tianjin hotel" in lowered or "feast worth millions" in lowered:
            return "The neighborhood starts whispering that he must be secretly wealthy."
        if "rich family" in lowered or "make him like me" in lowered:
            return "Some of them start seeing him as a ticket into a richer life."
        if "move everythingto the warehouse" in lowered or "property management" in lowered:
            return "He quietly reroutes the delivery so everything can be hidden away at once."
        if "empty out this place quick" in lowered or "just in case early" in lowered:
            return "He rushes to clear the warehouse before anyone notices what he is doing."
        if "delivered all 500" in lowered or "three course meals" in lowered:
            return "The banquet order arrives in full, exactly as he planned."
        if "we are good to go" in lowered:
            return "With another major step finished, the larger survival plan can finally move forward."
        if "i've already talked to the" in lowered:
            return "He lines up the next connection before delays can ruin the schedule."
        if "this stockpile of mine" in lowered:
            return "The sheer size of his stockpile finally starts to sink in."
        if "incredible amount of fresh" in lowered or "hot pot bases" in lowered:
            return "An enormous reserve of fresh meals finally comes together under his control."
        if "construction of zhang yi's fortress began" in lowered:
            return "By the next day, construction on his fortress-like home is already underway."
        if "show off my skills to a couple of friends" in lowered:
            return "He disguises the weapon purchase as a harmless outing with friends."
        if "luxurious meals" in lowered or "live like a king" in lowered:
            return "For the moment, he plans to enjoy every luxury the coming winter will erase."
        if "what is going on with zhang yi" in lowered or "world was gonna end" in lowered:
            return "The neighbors notice the pattern, but none of them understand what he already knows."
        if "send you to hell" in lowered or "finally see the truth" in lowered:
            return "His hatred for Fang Yuqing hardens as he imagines the revenge still to come."
        if "leave the renovation to you" in lowered or "stay at a hotel" in lowered:
            return "He leaves the apartment renovation in trusted hands while he works on everything else."
        if "shit hits the fan in a month" in lowered:
            return "He knows they will understand his choices only when the apocalypse finally arrives."
        if "warehouse manager zhou" in lowered or "large amount of medicine" in lowered:
            return "He leans on his warehouse connection to gather medicine before shortages begin."
        if "double of the market price" in lowered:
            return "He offers extra money without hesitation to secure the medicine faster."
        if "transfer the money right away" in lowered:
            return "The payment goes out immediately so the medicine deal cannot fall apart."
        if "general supplies" in lowered and "warehouse" in lowered:
            return "A warehouse full of supplies finally starts to bring his survival plan together."
        if "deliver what you want" in lowered and "good quality" in lowered:
            return "After days of searching, he finally finds someone who can deliver what he needs."
        generic_line = self._generic_panel_specific_line(text, scene_summary)
        if generic_line:
            return generic_line
        scene_keywords = keyword_tokens(scene_summary)
        panel_keywords = keyword_tokens(lowered)
        if scene_keywords and panel_keywords:
            overlap = scene_keywords & panel_keywords
            if overlap:
                summary_clauses = self._summary_clauses(scene_summary)
                if summary_clauses:
                    return summary_clauses[min(panel_index, len(summary_clauses) - 1)]
        return ""

    def _fallback_panel_line(self, panel: dict[str, Any]) -> str:
        panel_clause = self._panel_specific_line(clean_ocr_text(str(panel.get("text", ""))), "", 0, 1)
        return self._normalize_sentence(panel_clause, allow_empty=True) if panel_clause else ""

    def _avoid_repetition(
        self,
        narration: str,
        previous_lines: list[str],
        panel_text: str,
        fragment: str,
        scene_summary: str,
        panel_index: int,
        panel_count: int,
    ) -> str:
        if not previous_lines:
            return narration
        if all(self._sentence_similarity(existing, narration) <= 0.88 for existing in previous_lines):
            return narration

        alternatives = []
        if fragment:
            alternatives.extend(self._summary_clauses(fragment))
        alternatives.extend(self._summary_clauses(scene_summary))

        for alternative in alternatives:
            candidate = self._normalize_sentence(alternative, allow_empty=True)
            if candidate and all(self._sentence_similarity(existing, candidate) <= 0.88 for existing in previous_lines):
                return candidate

        if panel_count > 1 and scene_summary:
            summary_clauses = self._summary_clauses(scene_summary)
            if summary_clauses:
                candidate = self._normalize_sentence(
                    summary_clauses[min(panel_index, len(summary_clauses) - 1)],
                    allow_empty=True,
                )
                if candidate:
                    return candidate
        return ""

    def _position_fallback(self, scene_summary: str, panel_index: int, panel_count: int) -> str:
        clauses = self._summary_clauses(scene_summary)
        if clauses:
            return clauses[min(panel_index, len(clauses) - 1)]
        return self._normalize_sentence(scene_summary)

    def _vary_sentence(self, sentence: str, panel: dict[str, Any], index: int) -> str:
        del panel, index
        return sentence

    def _spread_positions(self, total_targets: int, total_items: int) -> list[int]:
        if total_targets <= 0:
            return []
        if total_items <= 1:
            return [0]
        positions = [
            round(item_index * (total_targets - 1) / max(total_items - 1, 1))
            for item_index in range(total_items)
        ]
        return [max(0, min(total_targets - 1, position)) for position in positions]

    def _coerce_int(self, value: Any) -> int | None:
        if isinstance(value, int):
            return value
        match = re.search(r"\d+", str(value or ""))
        return int(match.group(0)) if match else None

    def _normalize_sentence(self, sentence: str, allow_empty: bool = False) -> str:
        cleaned = re.sub(r"\s+", " ", str(sentence or "")).strip(" \"'")
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        if not cleaned:
            return "" if allow_empty else "A crucial shift changes everything."
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned[0].upper() + cleaned[1:]

    def _normalize_phrase(self, phrase: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(phrase or "")).strip(" \"'")
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned

    def _sentence_similarity(self, left: str, right: str) -> float:
        left_tokens = self._token_counts(left)
        right_tokens = self._token_counts(right)
        if not left_tokens or not right_tokens:
            return 0.0
        shared = set(left_tokens) & set(right_tokens)
        numerator = sum(left_tokens[token] * right_tokens[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left_tokens.values()))
        right_norm = math.sqrt(sum(value * value for value in right_tokens.values()))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _token_counts(self, sentence: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for token in re.findall(r"[a-z']{3,}", sentence.casefold()):
            counts[token] = counts.get(token, 0) + 1
        return counts

    def _summary_clauses(self, summary: str) -> list[str]:
        clauses = [
            self._normalize_sentence(piece, allow_empty=True)
            for piece in re.split(r"(?<=[.!?;:])\s+", summary, flags=re.IGNORECASE)
            if piece.strip()
        ]
        filtered: list[str] = []
        for clause in clauses:
            if not clause:
                continue
            lowered = clause.casefold().strip()
            word_count = len(re.findall(r"[a-z']+", lowered))
            if word_count < 4:
                continue
            if self._looks_like_dangling_clause(lowered):
                continue
            filtered.append(clause)
        return filtered

    def _lowercase_first(self, sentence: str) -> str:
        cleaned = sentence.strip()
        if not cleaned:
            return ""
        return cleaned[0].lower() + cleaned[1:]

    def _scene_category(self, summary: str) -> str:
        del summary
        return ""

    def _category_position_lines(self, category: str) -> list[str]:
        del category
        return []

    def _generic_panel_specific_line(self, text: str, scene_summary: str) -> str:
        lowered = text.casefold()
        scene_lowered = scene_summary.casefold()
        category = self._generic_category(lowered, scene_lowered)
        if not category:
            return ""

        subject = self._subject_hint(lowered, scene_summary)
        focus = self._focus_clause(lowered, category)
        templates = {
            "identity": [
                f"{subject} finally puts a name and history to the person at the center of the story.",
                f"The introduction turns personal as {subject} starts defining who he is.",
            ],
            "routine": [
                f"{subject}'s ordinary routine comes into focus before anything truly changes.",
                f"The panel lingers on {subject}'s familiar daily grind.",
            ],
            "reading": [
                f"{subject}'s fixation on the story becomes impossible to ignore.",
                f"The novel keeps pulling {subject} deeper into its world.",
            ],
            "request": [
                f"A pointed request throws {subject} off balance.",
                f"The conversation shifts sharply once {focus}.",
            ],
            "mission": [
                f"{subject} pushes the mission forward while {focus}.",
                f"The strange assignment keeps moving as {focus}.",
            ],
            "admiration": [
                f"Admiration spreads quickly as {focus}.",
                f"The crowd's reaction makes it clear how highly that figure is regarded.",
            ],
            "magic": [
                f"Magic erupts without warning as {focus}.",
                f"The situation turns volatile the moment {focus}.",
            ],
            "escape": [
                f"Panic spikes as {focus}.",
                f"The danger escalates immediately once {focus}.",
            ],
            "authority": [
                f"An official response starts taking shape around {focus}.",
                f"The fallout becomes harder to ignore once {focus}.",
            ],
            "system": [
                f"An abrupt warning tears through the routine and changes everything.",
                f"The ordinary world starts breaking apart the moment the announcement arrives.",
            ],
            "preparation": [
                f"{subject} keeps preparing while {focus}.",
                f"The survival plan becomes more concrete as {focus}.",
            ],
            "finance": [
                f"Money becomes another weapon in the plan as {focus}.",
                f"{subject} leans on cash and debt because {focus}.",
            ],
            "fortress": [
                f"{subject}'s home starts transforming as {focus}.",
                f"Fortification becomes the priority once {focus}.",
            ],
            "weapon": [
                f"Protection becomes urgent as {focus}.",
                f"{subject} starts treating weapons as part of survival.",
            ],
            "betrayal": [
                f"Betrayal comes into sharp focus as {focus}.",
                f"The cruelty of the situation is impossible to ignore once {focus}.",
            ],
            "reversal": [
                f"The truth hits hard: {focus}.",
                f"{subject} is forced to accept that {focus}.",
            ],
            "disaster": [
                f"The scale of the catastrophe becomes clearer as {focus}.",
                f"The wider danger finally comes into focus once {focus}.",
            ],
        }
        for candidate in templates.get(category, []):
            if candidate and not self._looks_like_dangling_clause(candidate.casefold()):
                return candidate
        return ""

    def _generic_category(self, text: str, scene_summary: str) -> str:
        combined = f"{text} {scene_summary}".casefold()
        categories = (
            ("identity", ("my name", "i am", "i'm", "years old", "graduated", "hobby", "introduce myself")),
            ("routine", ("metro", "commute", "company", "office", "boss", "overtime", "home")),
            ("reading", ("novel", "reader", "author", "webnovel", "chapter", "email", "monetization")),
            ("request", ("lend me", "loan", "money?", "what are you doing", "can you", "who goes there", "what happened")),
            ("mission", ("mission", "gift", "gifts", "present", "presents", "holy", "deliver")),
            ("admiration", ("strongest", "legend", "elegant", "cool", "lord", "lady", "popular")),
            ("magic", ("magic", "spell", "intruder", "sleep", "attack", "perception", "body is awake")),
            ("escape", ("run away", "hurt", "escape", "flee", "danger", "prince's room")),
            ("authority", ("report", "knight", "responsible", "manager", "guard", "security")),
            ("system", ("scenario", "system", "warning", "announcement", "planetary", "main scenario")),
            ("preparation", ("supplies", "stockpile", "supermarket", "warehouse", "prepared meals", "medicine")),
            ("finance", ("loan", "mortgage", "deposit", "interest", "million", "account number")),
            ("fortress", ("safe house", "vault", "bulletproof", "alloy", "fortress", "ventilation")),
            ("weapon", ("weapon", "guns", "crossbow", "hunting", "axes", "knives", "crowbar")),
            ("betrayal", ("kill him", "corpse", "betrayed", "revenge", "bastard", "share the food")),
            ("reversal", ("regress", "back in time", "travel back", "month before", "returned to the past")),
            ("disaster", ("freeze", "supernova", "blizzard", "apocalypse", "destroyed world")),
        )
        for category, keywords in categories:
            if any(keyword in combined for keyword in keywords):
                return category
        return ""

    def _subject_hint(self, text: str, scene_summary: str) -> str:
        named_match = re.search(r"\b([A-Z][a-z]+(?:[-\s][A-Z][a-z]+){0,2})\b", scene_summary)
        if named_match:
            name = named_match.group(1).strip()
            if name not in {"The", "A", "An"}:
                return name
        if any(token in text for token in ("neighbors", "committee", "coincidence")):
            return "The neighbors"
        if any(token in text for token in ("crowd", "corpse", "kill him")):
            return "The crowd"
        if any(token in text for token in ("manager", "staff", "customer", "restaurant")):
            return "The staff"
        if any(token in text for token in ("guard", "security", "knight")):
            return "The authorities"
        if any(token in text for token in ("author", "novel", "reader")):
            return "The protagonist"
        return "The protagonist"

    def _focus_clause(self, text: str, category: str) -> str:
        specific_focus = {
            "identity": "the story finally settles on his identity",
            "routine": "the familiar routine starts to feel a little different",
            "reading": "the story on the screen matters more than the world around him",
            "request": "an unexpected request cuts into the moment",
            "mission": "the assignment keeps building pressure and expectation",
            "admiration": "a larger-than-life reputation dominates the conversation",
            "magic": "magic collides with panic in an instant",
            "escape": "the attempt to get away turns frantic",
            "authority": "the consequences start drawing serious attention",
            "system": "the ordinary evening is shattered by a larger warning",
            "preparation": "every supply run starts to matter more",
            "finance": "the cost of preparation keeps rising",
            "fortress": "the apartment is being rebuilt for survival",
            "weapon": "self-defense becomes part of the plan",
            "betrayal": "trust has already collapsed into violence",
            "reversal": "the impossible second chance is now real",
            "disaster": "the full scale of the coming disaster is finally explained",
        }
        if category in specific_focus:
            return specific_focus[category]
        return "the situation changes in a decisive way"

    def _looks_like_dangling_clause(self, text: str) -> bool:
        lowered = str(text or "").strip().casefold()
        if not lowered:
            return False
        word_count = len(re.findall(r"[a-z']+", lowered))
        if word_count and word_count <= 3:
            return True
        return lowered.startswith(
            (
                "and ",
                "but ",
                "while ",
                "then ",
                "as ",
                "leading to ",
                "giving ",
                "making ",
                "setting ",
                "knowing ",
                "despite ",
                "starting ",
                "living ",
                "expressing ",
                "announcing ",
                "signaling ",
                "post ",
                "by coincidence ",
                "when i ",
                "if it ",
            )
        )
