from pathlib import Path
from clinician_tools.validate_templates import validate

def test_registry_is_valid():
    path = Path(__file__).parents[2] / "templates" / "templates.json"
    assert validate(path) == []
