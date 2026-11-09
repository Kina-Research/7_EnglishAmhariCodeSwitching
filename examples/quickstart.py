#!/usr/bin/env python3
"""
examples/quickstart.py
======================
Minimal usage example for EthioSwitch-Bench.

Demonstrates:
  1. Loading the dataset
  2. Printing corpus-level statistics
  3. Filtering by topic, split, and CS frequency
  4. Iterating over turns in a dialogue
  5. Computing per-topic mean annotation scores

Run from the repository root:
    python examples/quickstart.py
"""
from __future__ import annotations
import sys
import io

# Ensure UTF-8 output on all platforms (Windows cp1252 can't render Amharic script)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import os
from collections import defaultdict
from typing import Dict, List

# ---------------------------------------------------------------------------
# Path setup — works when run from repo root or examples/ directory
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
V2_PATH     = os.path.join(REPO_ROOT, "code", "data", "ethioswitch_bench_v2.json")
V3_PATH     = os.path.join(REPO_ROOT, "code", "data", "ethioswitch_bench_v3.json")


def load_dataset(path: str) -> dict:
    """Load an EthioSwitch-Bench JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. Load both versions
# ---------------------------------------------------------------------------
print_header("EthioSwitch-Bench — Quick Start")

v2 = load_dataset(V2_PATH)
v3 = load_dataset(V3_PATH)

print(f"\n  Loaded v{v2['version']}: {v2['total_dialogues']} dialogues  ({os.path.basename(V2_PATH)})")
print(f"  Loaded v{v3['version']}: {v3['total_dialogues']} dialogues  ({os.path.basename(V3_PATH)})")

# ---------------------------------------------------------------------------
# 2. Corpus-level statistics
# ---------------------------------------------------------------------------
print_header("Corpus Statistics — v3.0")

dialogues: List[dict] = v3["dialogues"]

n_total  = len(dialogues)
n_turns  = sum(len(d["turns"]) for d in dialogues)
topics   = set(d["topic"] for d in dialogues)
splits   = {s: sum(1 for d in dialogues if d["split"] == s) for s in ("train", "validation", "test")}

print(f"\n  Total dialogues : {n_total}")
print(f"  Total turns     : {n_turns}")
print(f"  Topics covered  : {len(topics)}")
print(f"  Splits          : train={splits['train']}, "
      f"validation={splits['validation']}, test={splits['test']}")

# ---------------------------------------------------------------------------
# 3. Filter examples
# ---------------------------------------------------------------------------
print_header("Filtering Examples")

# Filter by topic
health_dialogues = [d for d in dialogues if d["topic"] == "health"]
print(f"\n  Health-topic dialogues : {len(health_dialogues)}")

# Filter by CS frequency
high_cs = [d for d in dialogues if d["code_switch_frequency"] == "high"]
print(f"  High CS-frequency      : {len(high_cs)}")

# Filter by split and dominance
test_balanced = [d for d in dialogues
                 if d["split"] == "test" and d["language_dominance"] == "balanced"]
print(f"  Test + balanced        : {len(test_balanced)}")

# ---------------------------------------------------------------------------
# 4. Inspect a single dialogue
# ---------------------------------------------------------------------------
print_header("Sample Dialogue — ID 1")

sample = next(d for d in dialogues if d["dialogue_id"] == 1)
print(f"\n  Topic          : {sample['topic']}")
print(f"  CS Frequency   : {sample['code_switch_frequency']}")
print(f"  Dominance      : {sample['language_dominance']}")
print(f"  Source         : {sample['source']} ({sample['generation_method']})")
print(f"\n  Turns:")
for turn in sample["turns"]:
    lang_str = " + ".join(turn["language_tags"])
    print(f"    [{turn['speaker']:9s}] ({lang_str}) : {turn['utterance']}")
ann = sample["annotations"]
print(f"\n  Annotations:")
print(f"    Coherence              : {ann['coherence_score']}")
print(f"    Semantic consistency   : {ann['semantic_consistency']}")
print(f"    Switch naturalness     : {ann['switch_point_naturalness']}")
print(f"    Fluency                : {ann['fluency_rating']}")
print(f"    IAA kappa              : {ann['annotator_agreement_kappa']}")

# ---------------------------------------------------------------------------
# 5. Per-topic mean annotation scores
# ---------------------------------------------------------------------------
print_header("Per-Topic Mean Scores (v3.0)")

topic_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
for d in dialogues:
    t = d["topic"]
    a = d["annotations"]
    topic_data[t]["coherence"].append(a["coherence_score"])
    topic_data[t]["fluency"].append(a["fluency_rating"])
    topic_data[t]["naturalness"].append(a["switch_point_naturalness"])

print(f"\n  {'Topic':<22} {'Coherence':>9} {'Fluency':>9} {'Naturalness':>12}")
print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*12}")
for topic in sorted(topic_data.keys()):
    vals = topic_data[topic]
    coh  = sum(vals["coherence"])    / len(vals["coherence"])
    flu  = sum(vals["fluency"])      / len(vals["fluency"])
    nat  = sum(vals["naturalness"])  / len(vals["naturalness"])
    print(f"  {topic:<22} {coh:>9.3f} {flu:>9.3f} {nat:>12.3f}")

# ---------------------------------------------------------------------------
# 6. Switch-type distribution
# ---------------------------------------------------------------------------
print_header("Switch-Type Distribution (v3.0, first 1000 dialogues)")

from collections import Counter
switch_counts: Counter = Counter()
for d in dialogues[:1000]:
    for turn in d["turns"]:
        switch_counts[turn["switch_type"]] += 1

print(f"\n  {'Switch Type':<25} {'Count':>6}")
print(f"  {'-'*25} {'-'*6}")
for st, cnt in switch_counts.most_common(10):
    print(f"  {st:<25} {cnt:>6}")

print("\n  [Done] EthioSwitch-Bench quickstart complete.")
