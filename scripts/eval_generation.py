#!/usr/bin/env python3
"""R8 pilot: run generation on gold set and compare answerable flag."""

import json
import subprocess
import sys
from pathlib import Path

GOLD = Path("data/gold/gold.jsonl")

def main():
    gold = [json.loads(l) for l in GOLD.read_text().strip().splitlines()]
    correct = 0
    for g in gold:
        print(f"\n--- {g['id']}: {g['question'][:60]}...")
        proc = subprocess.run(
            [sys.executable, "scripts/generate_and_validate.py", g["question"]],
            capture_output=True, text=True
        )
        out = proc.stdout
        # crude parse of the JSON block
        if '"answerable": true' in out:
            pred = True
        elif '"answerable": false' in out:
            pred = False
        else:
            pred = None
        expected = g["answerable"]
        ok = pred == expected
        correct += int(ok)
        print(f"  expected={expected}  predicted={pred}  {'OK' if ok else 'MISMATCH'}")
    print(f"\nAnswerable accuracy: {correct}/{len(gold)} = {correct/len(gold):.2f}")

if __name__ == "__main__":
    main()
