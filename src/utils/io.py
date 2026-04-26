"""
File I/O helpers.

Utilities for loading configs (YAML/JSON) and saving experiment results.
"""

import json
from pathlib import Path
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data as formatted JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_jsonl(records: list, path: str) -> None:
    """Save a list of records as JSONL (one JSON object per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
