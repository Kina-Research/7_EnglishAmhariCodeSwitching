#!/usr/bin/env python3
"""
01_dataset_builder.py
=====================
Loads, validates, and prints statistics for EthioSwitch-Bench v2.0.
Also computes derived corpus-level features (switch density, dominance).

Usage:
    python 01_dataset_builder.py
"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter
from typing import List, Dict, Any

# Allow imports from sibling 'utils' package
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_models import Dialogue, DialogueTurn
from utils.text_utils import detect_language_dominance, compute_switch_density

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "data", "ethioswitch_bench_v3.json")

ANNOTATION_RANGES = {
    "coherence_score":          (1.0, 5.0),
    "semantic_consistency":     (0.0, 1.0),
    "switch_point_naturalness": (1.0, 5.0),
    "fluency_rating":           (1.0, 5.0),
    "annotator_agreement_kappa":(0.0, 1.0),
}
MIN_KAPPA  = 0.60
MIN_TURNS  = 4


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_dataset(path: str = DATASET_PATH) -> List[Dialogue]:
    """
    Load EthioSwitch-Bench JSON and parse into Dialogue dataclasses.

    Parameters
    ----------
    path : path to ethioswitch_bench_v2.json

    Returns
    -------
    list of Dialogue objects
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dialogues: List[Dialogue] = []
    for d in raw["dialogues"]:
        turns = [
            DialogueTurn(
                turn_id=t["turn_id"],
                speaker=t["speaker"],
                utterance=t["utterance"],
                language_tags=t["language_tags"],
                switch_type=t["switch_type"],
                switch_point_token=t.get("switch_point_token", ""),
            )
            for t in d["turns"]
        ]
        ann = d["annotations"]
        dialogues.append(Dialogue(
            dialogue_id=d["dialogue_id"],
            topic=d["topic"],
            code_switch_frequency=d["code_switch_frequency"],
            language_dominance=d["language_dominance"],
            turns=turns,
            coherence_score=ann["coherence_score"],
            semantic_consistency=ann["semantic_consistency"],
            switch_point_naturalness=ann["switch_point_naturalness"],
            fluency_rating=ann["fluency_rating"],
            annotator_agreement_kappa=ann["annotator_agreement_kappa"],
        ))
    return dialogues


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataset(dialogues: List[Dialogue]) -> List[str]:
    """
    Validate all dialogues against schema constraints.

    Returns
    -------
    list of error strings (empty list = all valid)
    """
    errors: List[str] = []
    for d in dialogues:
        prefix = f"Dialogue {d.dialogue_id}"

        # Minimum turns
        if len(d.turns) < MIN_TURNS:
            errors.append(f"{prefix}: only {len(d.turns)} turns (min {MIN_TURNS})")

        # Annotation ranges
        vals = {
            "coherence_score":          d.coherence_score,
            "semantic_consistency":     d.semantic_consistency,
            "switch_point_naturalness": d.switch_point_naturalness,
            "fluency_rating":           d.fluency_rating,
            "annotator_agreement_kappa":d.annotator_agreement_kappa,
        }
        for field, (lo, hi) in ANNOTATION_RANGES.items():
            v = vals[field]
            if not (lo <= v <= hi):
                errors.append(f"{prefix}: {field}={v} outside [{lo},{hi}]")

        # IAA threshold
        if d.annotator_agreement_kappa < MIN_KAPPA:
            errors.append(f"{prefix}: kappa={d.annotator_agreement_kappa} < {MIN_KAPPA}")

        # Language dominance field
        if d.language_dominance not in ("english_dominant", "amharic_dominant", "balanced"):
            errors.append(f"{prefix}: unknown language_dominance={d.language_dominance}")

    return errors


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(dialogues: List[Dialogue]) -> Dict[str, Any]:
    """Compute corpus-level descriptive statistics."""
    topic_counts   = Counter(d.topic for d in dialogues)
    freq_counts    = Counter(d.code_switch_frequency for d in dialogues)
    dom_counts     = Counter(d.language_dominance for d in dialogues)
    total_turns    = sum(len(d.turns) for d in dialogues)
    avg_turns      = total_turns / len(dialogues)
    densities      = [compute_switch_density(d.turns) for d in dialogues]
    avg_density    = sum(densities) / len(densities)

    mean_coherence = sum(d.coherence_score for d in dialogues) / len(dialogues)
    mean_sem_con   = sum(d.semantic_consistency for d in dialogues) / len(dialogues)
    mean_fluency   = sum(d.fluency_rating for d in dialogues) / len(dialogues)
    mean_nat       = sum(d.switch_point_naturalness for d in dialogues) / len(dialogues)
    mean_kappa     = sum(d.annotator_agreement_kappa for d in dialogues) / len(dialogues)

    return {
        "total_dialogues": len(dialogues),
        "total_turns": total_turns,
        "avg_turns_per_dialogue": round(avg_turns, 2),
        "avg_switch_density_per_100_tokens": round(avg_density, 2),
        "topic_distribution": dict(topic_counts),
        "cs_frequency_distribution": dict(freq_counts),
        "language_dominance_distribution": dict(dom_counts),
        "mean_coherence_score": round(mean_coherence, 3),
        "mean_semantic_consistency": round(mean_sem_con, 3),
        "mean_fluency_rating": round(mean_fluency, 3),
        "mean_switch_point_naturalness": round(mean_nat, 3),
        "mean_iaa_kappa": round(mean_kappa, 3),
    }


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print a formatted statistics table."""
    print("\n" + "=" * 60)
    print("  EthioSwitch-Bench v2.0  |  Corpus Statistics")
    print("=" * 60)
    print(f"  Total dialogues        : {stats['total_dialogues']}")
    print(f"  Total turns            : {stats['total_turns']}")
    print(f"  Avg turns/dialogue     : {stats['avg_turns_per_dialogue']}")
    print(f"  Avg switch density     : {stats['avg_switch_density_per_100_tokens']} / 100 tokens")
    print()
    print("  Topic distribution:")
    for k, v in sorted(stats["topic_distribution"].items()):
        print(f"    {k:<20}: {v}")
    print()
    print("  CS frequency distribution:")
    for k, v in sorted(stats["cs_frequency_distribution"].items()):
        print(f"    {k:<20}: {v}")
    print()
    print("  Language dominance distribution:")
    for k, v in sorted(stats["language_dominance_distribution"].items()):
        print(f"    {k:<20}: {v}")
    print()
    print("  Annotation means (human-validated):")
    print(f"    Coherence             : {stats['mean_coherence_score']}")
    print(f"    Semantic consistency  : {stats['mean_semantic_consistency']}")
    print(f"    Fluency               : {stats['mean_fluency_rating']}")
    print(f"    Switch naturalness    : {stats['mean_switch_point_naturalness']}")
    print(f"    IAA κ                 : {stats['mean_iaa_kappa']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[01] Loading dataset from:\n     {DATASET_PATH}\n")
    dialogues = load_dataset()
    print(f"[01] Loaded {len(dialogues)} dialogues.")

    print("[01] Validating dataset schema...")
    errors = validate_dataset(dialogues)
    if errors:
        print(f"[01] VALIDATION FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"     [Error] {e}")
        sys.exit(1)
    print(f"[01] Success: All {len(dialogues)} dialogues passed validation.")

    stats = compute_statistics(dialogues)
    print_statistics(stats)

    out_path = os.path.join(SCRIPT_DIR, "data", "dataset_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n[01] Statistics saved -> {out_path}")
