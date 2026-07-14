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


# ─── Parser ─────────────────────────────────────────────────────────────────

class ScreenplayParser:

    def parse_file(self, filepath: str | Path) -> ParsedScreenplay:
        path = Path(filepath)
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return self.parse_text(raw, title=path.stem)

    def parse_text(self, raw: str, title: str = "Unknown") -> ParsedScreenplay:
        lines = raw.splitlines()
        lines = self._clean_lines(lines)

        # Different scripts indent character cues very differently (some as
        # little as 3 spaces, some 35+). Rather than assuming a fixed
        # number, work it out from this specific file before parsing.
        self._cue_indent_threshold = self._compute_cue_indent_threshold(lines)

        scenes = self._extract_scenes(lines)
        characters = self._extract_characters(scenes)

        return ParsedScreenplay(title=title, scenes=scenes, characters=characters)

    # ── Preprocessing ────────────────────────────────────────────────────────

    def _clean_lines(self, lines: list[str]) -> list[str]:
        cleaned = []
        for line in lines:
            # Strip trailing whitespace but preserve leading (indentation matters)
            line = line.rstrip()
            # Skip obvious junk
            if SKIP_LINE.match(line.strip()):
                continue
            cleaned.append(line)
        return cleaned

    # ── Scene extraction ─────────────────────────────────────────────────────

    def _extract_scenes(self, lines: list[str]) -> list[Scene]:
        scenes: list[Scene] = []
        # Start with a placeholder "scene" instead of None. This means any
        # content that appears before the first recognized scene heading
        # (or, in the worst case, a script where NO heading is ever
        # recognized because of an unfamiliar slugline format) still gets
        # captured instead of being silently discarded. If it stays empty,
        # it's dropped rather than appended.
        current_scene: Optional[Scene] = Scene(
            index=0,
            heading="(no scene heading detected before this point)",
            interior_exterior=None,
            location="",
            time_of_day=None,
        )
        i = 0

        while i < len(lines):
            stripped = lines[i].strip()

            heading_match = SCENE_HEADING.match(stripped)

            if heading_match:
                # Save previous scene, but only if it actually has content —
                # otherwise it's just the empty placeholder or a stray match.
                if current_scene is not None and (
                    current_scene.action_lines or current_scene.dialogue
                ):
                    scenes.append(current_scene)

                ie = heading_match.group(1).upper().replace(" ", "")
                location = (heading_match.group(2) or "").strip()
                tod = (heading_match.group(3) or "").strip() or None

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
                i += 1
                continue

            # ── Inside a scene ───────────────────────────────────────────────
            # Dialogue block detection:
            # A character cue is typically indented (>= 20 spaces or ~4 tabs)
            # followed by dialogue lines, possibly with parentheticals

            if self._is_character_cue(lines[i]):
                character = stripped.split("(")[0].strip()
                dialogue_lines = []
                i += 1

                while i < len(lines):
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
                    if SCENE_HEADING.match(next_stripped):  # new scene
                        break
                    if self._is_character_cue(lines[i]):    # next character
                        break
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

        return scenes

    def _compute_cue_indent_threshold(self, lines: list[str]) -> int:
        """
        Work out how far character cues are indented in THIS script, rather
        than assuming a fixed number — different scripts use very different
        conventions (3 spaces vs. 35+ spaces), and some even indent cues at
        the SAME depth as action/dialogue text, which rules out simply
        comparing against a "prose baseline".

        What's reliably true across almost all screenplay formatting: page
        furniture (CONTINUED, scene numbers, THE END, CUT TO:, etc.) sits
        flush-left at indent 0, while actual character names are indented
        at least a little, whatever the exact convention. So: ignore
        flush-left ALL-CAPS lines as noise, and use the most common
        indentation among the rest as the cue depth.
        """
        from collections import Counter

        caps_indents: Counter = Counter()
        for line in lines:
            stripped = line.strip()
            if len(stripped) < 2 or len(stripped) > 40:
                continue
            if not stripped.isupper():
                continue
            indent = len(line) - len(line.lstrip())
            if indent == 0:
                continue  # flush-left is almost always technical furniture, not a name
            caps_indents[indent] += 1

        if caps_indents:
            return caps_indents.most_common(1)[0][0]
        return 1  # nothing indented at all found — fall back to "any indent"

    def _is_character_cue(self, line: str) -> bool:
        """
        Character cues are ALL-CAPS and indented deeper than ordinary prose.
        The exact depth varies per script, so we compare against a threshold
        computed for this specific file (see _compute_cue_indent_threshold),
        not a fixed number.
        """
        stripped = line.strip()
        if not stripped:
            return False
        indent = len(line) - len(line.lstrip())
        if indent < getattr(self, "_cue_indent_threshold", 15):
            return False
        return bool(CHARACTER_CUE.match(stripped))

    # ── Character extraction ─────────────────────────────────────────────────

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


# ─── CLI helper ─────────────────────────────────────────────────────────────

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