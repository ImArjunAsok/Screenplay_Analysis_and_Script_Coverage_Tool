import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DialogueLine:
    character: str
    text: str
    line_number: int


@dataclass
class Scene:
    index: int
    heading: str
    interior_exterior: Optional[str]   # INT / EXT / INT./EXT.
    location: str
    time_of_day: Optional[str]         # DAY / NIGHT / CONTINUOUS etc.
    action_lines: list[str] = field(default_factory=list)
    dialogue: list[DialogueLine] = field(default_factory=list)
    raw_text: str = ""

    @property
    def full_text(self) -> str:
        """All text in this scene as a single string — used for NLP later."""
        parts = [self.heading] + self.action_lines
        for d in self.dialogue:
            parts.append(f"{d.character}: {d.text}")
        return " ".join(parts)


@dataclass
class ParsedScreenplay:
    title: str
    scenes: list[Scene] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    front_matter: list[str] = field(default_factory=list)
    parser_notes: list[str] = field(default_factory=list)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    @property
    def dialogue_count(self) -> int:
        return sum(len(s.dialogue) for s in self.scenes)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "scene_count": self.scene_count,
            "dialogue_count": self.dialogue_count,
            "character_count": len(self.characters),
            "characters": self.characters,
            "front_matter": self.front_matter,
            "parser_notes": self.parser_notes,
            "scenes": [
                {
                    "index": s.index,
                    "heading": s.heading,
                    "interior_exterior": s.interior_exterior,
                    "location": s.location,
                    "time_of_day": s.time_of_day,
                    "action_lines": s.action_lines,
                    "dialogue": [asdict(d) for d in s.dialogue],
                    "full_text": s.full_text,
                }
                for s in self.scenes
            ],
        }


# ─── Regex patterns ─────────────────────────────────────────────────────────

# Scene headings: INT. COFFEE SHOP - DAY  /  EXT. STREET - NIGHT
# Also accepts variants seen across different scripts/eras:
#   EXTERIOR - LA - DAY        (spelled out, no period)
#   INT & EXT CAVES - NIGHT    (ampersand instead of slash, no period)
#   EXT  THE STREAM - DAY      (no period, irregular spacing)
SCENE_HEADING = re.compile(
    r"^(INT\.?\s*[/&]\s*EXT\.?|EXT\.?\s*[/&]\s*INT\.?|I/E|"
    r"INT\.?|EXT\.?|INTERIOR|EXTERIOR)\s+(.+?)(?:\s*[-–—]\s*(.+))?$",
    re.IGNORECASE,
)

# Some scripts (older drafts especially) drop INT./EXT. entirely and just
# give a location + time, e.g. "MEL'S DRIVE-IN--DUSK". This only matches
# lines that are already fully uppercase (character class excludes
# lowercase), so it can't accidentally swallow ordinary action-line prose.
TIME_OF_DAY_KEYWORDS = (
    r"DAY|NIGHT|MORNING|EVENING|AFTERNOON|DUSK|DAWN|CONTINUOUS|LATER|"
    r"MOMENTS LATER|SAME TIME|SUNSET|SUNRISE|PRESENT"
)
BARE_SCENE_HEADING = re.compile(
    r"^([A-Z0-9][A-Z0-9 '\-,\.]{2,50})\s*[-–—]{1,2}\s*(" + TIME_OF_DAY_KEYWORDS + r")\s*$"
)

# "Shooting script" drafts often number every scene heading, sometimes on
# both ends of the line, e.g.:
#   "1   EXT. LOWER MANHATTAN SKYLINE - BROOKLYN BRIDGE - MORNING    1"
# Strip a leading/trailing bare number before heading-matching. This can't
# cause false positives on its own -- the stripped candidate still has to
# pass the real SCENE_HEADING or BARE_SCENE_HEADING check afterward.
LEADING_SCENE_NUMBER = re.compile(r"^\d+[A-Za-z]?\s+")
TRAILING_SCENE_NUMBER = re.compile(r"\s+\d+[A-Za-z]?\s*$")

# Character cues: all-caps, optionally followed by (V.O.) / (O.S.) / (CONT'D)
CHARACTER_CUE = re.compile(
    r"^([A-Z][A-Z\s\-\'\.]{1,40})(?:\s*\((?:V\.O\.|O\.S\.|O\.C\.|CONT\'D|CONTINUED)\))?$"
)

# Parentheticals: (beat) / (sighs) — skip these, they're stage directions
PARENTHETICAL = re.compile(r"^\(.*\)$")

# Lines that are clearly page numbers or CONTINUED headers — skip
SKIP_LINE = re.compile(
    r"^(CONTINUED:|FADE IN:|FADE OUT\.|CUT TO:|SMASH CUT TO:|DISSOLVE TO:|THE END|[0-9]+\.?\s*$)",
    re.IGNORECASE,
)

class ScreenplayParser:

    def parse_file(self, filepath: str | Path) -> ParsedScreenplay:
        path = Path(filepath)
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return self.parse_text(raw, title=path.stem)

    def parse_text(self, raw: str, title: str = "Unknown") -> ParsedScreenplay:
        lines = raw.splitlines()
        lines = self._clean_lines(lines)

        self._cue_indent_threshold, self._indentation_reliable = (
            self._compute_cue_indent_threshold(lines)
        )

        strict_heading_count = sum(
            1 for line in lines
            if self._strict_heading_match(line.strip()) is not None
        )
        self._use_bare_heading_fallback = (
            len(lines) > 200 and strict_heading_count < max(3, len(lines) // 300)
        )

        notes = []
        if not self._indentation_reliable:
            notes.append(
                "Indentation was unreliable in this file (most ALL-CAPS "
                "lines were flush-left, so cue depth couldn't be measured). "
                "Fell back to a blank-line heuristic for character cues -- "
                "spot-check the dialogue/character output on this one."
            )
        if self._use_bare_heading_fallback:
            notes.append(
                f"Only {strict_heading_count} standard INT./EXT. scene "
                "headings found in a file this long, so bare "
                "\"LOCATION--TIME\" headings (no INT./EXT.) were also "
                "matched -- spot-check scene boundaries on this one."
            )

        scenes, front_matter = self._extract_scenes(lines)
        characters = self._extract_characters(scenes)

        if not scenes and front_matter:
            notes.append(
                "No scene headings of any kind were found in this file, "
                "even with the bare-heading fallback enabled. This usually "
                "means the source isn't a standard screenplay-formatted "
                "script (e.g. a dialogue-only fan transcript with no "
                "scene descriptions) rather than a parser failure -- "
                "recommend excluding it rather than forcing structure "
                "onto it."
            )

        return ParsedScreenplay(
            title=title,
            scenes=scenes,
            characters=characters,
            front_matter=front_matter,
            parser_notes=notes,
        )

    def _is_any_heading(self, stripped: str) -> bool:
        """True if this line is a scene heading under whichever mode is
        active for this file (strict INT/EXT, plus bare LOCATION--TIME
        headings if that fallback is enabled)."""
        if self._strict_heading_match(stripped) is not None:
            return True
        if self._use_bare_heading_fallback and BARE_SCENE_HEADING.match(stripped):
            return True
        return False

    def _strict_heading_match(self, stripped: str):
        """SCENE_HEADING match, but only counts if the line is actually
        uppercase -- rejects ordinary lowercase prose that happens to
        start with the word "interior"/"exterior" used as English words
        rather than as a slugline (e.g. "interior and stops on the
        vacated shoes..."), which the case-insensitive regex alone would
        otherwise match. Also strips a leading/trailing scene number
        (numbered shooting-script format) before matching."""
        if not stripped or stripped != stripped.upper():
            return None
        candidate = LEADING_SCENE_NUMBER.sub("", stripped, count=1)
        candidate = TRAILING_SCENE_NUMBER.sub("", candidate, count=1)
        return SCENE_HEADING.match(candidate)


    def _clean_lines(self, lines: list[str]) -> list[str]:
        cleaned = []
        for line in lines:
            line = line.rstrip()
            if SKIP_LINE.match(line.strip()):
                continue
            cleaned.append(line)
        return cleaned

    def _extract_scenes(self, lines: list[str]) -> tuple[list[Scene], list[str]]:
        scenes: list[Scene] = []
        front_matter: list[str] = []
        # No placeholder scene anymore. Anything encountered before the
        # first recognized scene heading is front matter (title page,
        # byline, "FADE IN:", SUPER: cards, etc), not a scene -- see
        # CHANGELOG note at the bottom of this file for why this changed.
        current_scene: Optional[Scene] = None
        i = 0
        n = len(lines)

        while i < n:
            stripped = lines[i].strip()

            heading_match = self._strict_heading_match(stripped)
            bare_match = None
            if heading_match is None and self._use_bare_heading_fallback:
                bare_match = BARE_SCENE_HEADING.match(stripped)

            if heading_match or bare_match:
                if current_scene is not None and (
                    current_scene.action_lines or current_scene.dialogue
                ):
                    scenes.append(current_scene)

                if heading_match:
                    ie = heading_match.group(1).upper().replace(" ", "")
                    location = (heading_match.group(2) or "").strip()
                    tod = (heading_match.group(3) or "").strip() or None
                else:
                    ie = None
                    location = bare_match.group(1).strip()
                    tod = bare_match.group(2).strip()

                current_scene = Scene(
                    index=len(scenes),
                    heading=stripped,
                    interior_exterior=ie,
                    location=location,
                    time_of_day=tod,
                )
                i += 1
                continue

            if current_scene is None:
                # Nothing recognized as a scene heading yet. Keep the
                # content, but as front matter -- it's real information
                # (title, author, etc.) but not part of the story.
                if stripped and not PARENTHETICAL.match(stripped):
                    front_matter.append(stripped)
                i += 1
                continue

            # Dialogue block detection:
            # A character cue is normally identified by indentation. When
            # indentation isn't trustworthy for this file, we instead look
            # at blank-line context: a short ALL-CAPS line sitting alone
            # (blank line before it) and immediately followed by content
            # that isn't itself a heading or another cue reads as a cue.

            prev_blank = (i == 0) or (lines[i - 1].strip() == "")
            next_nonblank = self._peek_next_nonblank(lines, i)

            if self._is_character_cue(lines[i], prev_blank, next_nonblank):
                character = stripped.split("(")[0].strip()
                dialogue_lines = []
                i += 1

                while i < n:
                    next_stripped = lines[i].strip()

                    if not next_stripped:
                        # A blank line right after the cue (or between
                        # parentheticals) is just formatting — skip it.
                        # A blank line AFTER we've already collected some
                        # dialogue text means the block is actually over.
                        if dialogue_lines:
                            break
                        i += 1
                        continue
                    if self._is_any_heading(next_stripped):  # new scene
                        break

                    inner_prev_blank = lines[i - 1].strip() == ""
                    inner_next_nonblank = self._peek_next_nonblank(lines, i)
                    if self._is_character_cue(lines[i], inner_prev_blank, inner_next_nonblank):
                        break  # next character
                    if PARENTHETICAL.match(next_stripped):  # skip (beat) etc.
                        i += 1
                        continue

                    dialogue_lines.append(next_stripped)
                    i += 1

                if dialogue_lines:
                    current_scene.dialogue.append(
                        DialogueLine(
                            character=character,
                            text=" ".join(dialogue_lines),
                            line_number=i,
                        )
                    )
                continue

            # Action line
            if stripped and not PARENTHETICAL.match(stripped):
                current_scene.action_lines.append(stripped)

            i += 1

        if current_scene is not None and (current_scene.action_lines or current_scene.dialogue):
            scenes.append(current_scene)

        return scenes, front_matter

    def _peek_next_nonblank(self, lines: list[str], i: int) -> Optional[str]:
        for j in range(i + 1, len(lines)):
            s = lines[j].strip()
            if s:
                return s
        return None

    def _compute_cue_indent_threshold(self, lines: list[str]) -> tuple[int, bool]:
        """
        Work out how far character cues are indented in THIS script, rather
        than assuming a fixed number — different scripts use very different
        conventions (3 spaces vs. 35+ spaces).

        Also reports whether that indentation signal can actually be
        trusted. If most short ALL-CAPS lines in the file sit flush-left,
        that's a sign the source has had its whitespace stripped or
        normalised (very common after scraping or copy-pasting from a
        webpage) -- in which case indentation can't reliably separate
        character cues from action lines or page furniture, and the caller
        should fall back to a different heuristic instead of silently
        finding zero dialogue.
        """
        from collections import Counter

        caps_indents: Counter = Counter()
        flush_left_caps = 0
        total_caps = 0

        for line in lines:
            stripped = line.strip()
            if len(stripped) < 2 or len(stripped) > 40:
                continue
            if not stripped.isupper():
                continue
            total_caps += 1
            indent = len(line) - len(line.lstrip())
            if indent == 0:
                flush_left_caps += 1
                continue  # flush-left is often furniture, not a name
            caps_indents[indent] += 1

        if caps_indents:
            threshold = caps_indents.most_common(1)[0][0]
        else:
            threshold = 1  # nothing indented at all — fall back to "any indent"

        # If the large majority of short ALL-CAPS lines are flush-left,
        # indentation isn't a usable signal for this file.
        reliable = total_caps > 0 and (flush_left_caps / total_caps) < 0.6

        return threshold, reliable

    def _is_character_cue(
        self,
        line: str,
        prev_line_blank: bool = False,
        next_nonblank_line: Optional[str] = None,
    ) -> bool:
        """
        Character cues are ALL-CAPS and, in a well-formatted file, indented
        deeper than ordinary prose. When indentation isn't reliable for
        this file (see _compute_cue_indent_threshold), fall back to a
        blank-line heuristic: a short ALL-CAPS line standing alone (blank
        line before it), followed by content that isn't itself a scene
        heading, reads as a character cue introducing dialogue.
        """
        stripped = line.strip()
        if not stripped:
            return False
        if not CHARACTER_CUE.match(stripped):
            return False
        if stripped.endswith("."):
            return False

        if self._indentation_reliable:
            indent = len(line) - len(line.lstrip())
            return indent >= self._cue_indent_threshold

        if not prev_line_blank:
            return False
        if next_nonblank_line is None:
            return False
        if self._is_any_heading(next_nonblank_line):
            return False
        return True


    def _extract_characters(self, scenes: list[Scene]) -> list[str]:
        """Return unique character names sorted by number of dialogue lines."""
        counts: dict[str, int] = {}
        for scene in scenes:
            for d in scene.dialogue:
                name = d.character.strip()
                if name:
                    counts[name] = counts.get(name, 0) + 1

        # Filter noise: characters must speak at least twice
        filtered = {k: v for k, v in counts.items() if v >= 2}
        return sorted(filtered, key=lambda k: filtered[k], reverse=True)



def parse_script(filepath: str, output_json: bool = False) -> ParsedScreenplay:
    parser = ScreenplayParser()
    result = parser.parse_file(filepath)

    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  {result.title}")
        print(f"{'='*60}")
        print(f"  Scenes    : {result.scene_count}")
        print(f"  Characters: {len(result.characters)}")
        print(f"  Dialogue  : {result.dialogue_count} lines")
        if result.front_matter:
            print(f"  Front matter lines (not counted as a scene): {len(result.front_matter)}")
        for note in result.parser_notes:
            print(f"  NOTE: {note}")
        print(f"\n  Top characters:")
        for char in result.characters[:10]:
            count = sum(1 for s in result.scenes for d in s.dialogue if d.character == char)
            print(f"    {char:<30} {count} lines")
        print(f"\n  First 3 scenes:")
        for scene in result.scenes[:3]:
            print(f"    [{scene.interior_exterior}] {scene.location} — {scene.time_of_day}")
            print(f"      {len(scene.action_lines)} action lines, {len(scene.dialogue)} dialogue blocks")

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python screenplay_parser.py <script.txt> [--json]")
        sys.exit(1)

    filepath = sys.argv[1]
    as_json = "--json" in sys.argv
    parse_script(filepath, output_json=as_json)
