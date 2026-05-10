import json
from pathlib import Path
from typing import List

def load_jsonl(file_path: Path) -> List[dict]:
    """Load jsonl file"""
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data: List[dict], file_path: Path):
    """Save to jsonl"""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
