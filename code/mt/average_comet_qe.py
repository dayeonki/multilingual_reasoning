from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path


def load_scores(path: Path) -> list[float]:
    scores: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARN {path}:{line_no}: {e}", file=sys.stderr)
                continue
            v = row.get("comet_qe")
            if v is not None:
                scores.append(float(v))
    return scores


def main() -> None:
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("aime_*.jsonl"))
    if not files:
        print(f"No aime_*.jsonl under {here}", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[str, int, float, float]] = []
    for path in files:
        m = re.match(r"aime_(.+)\.jsonl$", path.name)
        lang = m.group(1) if m else path.stem
        scores = load_scores(path)
        if not scores:
            mean = float("nan")
        else:
            mean = statistics.mean(scores)
        rows.append((lang, len(scores), mean, statistics.stdev(scores) if len(scores) > 1 else 0.0))

    rows.sort(key=lambda r: r[0])
    print(f"{'lang':<6} {'n':>4} {'mean_comet_qe':>14} {'stdev':>10}")
    for lang, n, mean, stdev in rows:
        print(f"{lang:<6} {n:>4} {mean:>14.4f} {stdev:>10.4f}")


if __name__ == "__main__":
    main()
