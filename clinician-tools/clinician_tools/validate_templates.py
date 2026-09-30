from __future__ import annotations
import json
import sys
from pathlib import Path

REQUIRED = {"template_id", "move_id", "text", "reviewer", "review_date"}
ALLOWED_SLOTS = {"emotion_word"}

def validate(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    ids: set[str] = set()
    templates = data.get("templates")
    if not isinstance(templates, list) or not templates:
        return ["templates must be a non-empty list"]
    for index, template in enumerate(templates):
        missing = REQUIRED - template.keys()
        if missing:
            errors.append(f"templates[{index}] missing: {sorted(missing)}")
        tid = template.get("template_id")
        if tid in ids:
            errors.append(f"duplicate template_id: {tid}")
        ids.add(tid)
        text = template.get("text", "")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"templates[{index}] text must be non-empty")
        slots = {part.split("}", 1)[0] for part in text.split("{")[1:] if "}" in part}
        unknown = slots - ALLOWED_SLOTS
        if unknown:
            errors.append(f"{tid}: unsupported slots {sorted(unknown)}")
        if not template.get("reviewer"):
            errors.append(f"{tid}: reviewer is required")
        if len(text) > 2000:
            errors.append(f"{tid}: text exceeds 2000 characters")
    return errors

def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "templates/templates.json")
    errors = validate(path)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"validated {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
