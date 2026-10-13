"""Atomic JSON updates that tolerate short Windows reader-sharing conflicts."""

import json
from pathlib import Path
import time


def atomic_write_json(path, value, attempts=10):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(.02 * (attempt + 1), .15))
