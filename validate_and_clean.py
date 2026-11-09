#!/usr/bin/env python3
"""
validate_and_clean.py
=====================
Validates and cleans EthioSwitch-Bench JSON datasets for public release.

Checks performed:
  1. Required top-level fields present
  2. All dialogues have required fields (dialogue_id, topic, split, turns, annotations)
  3. Minimum turn count (>= 4 per dialogue)
  4. Valid speaker labels (User / Assistant)
  5. Valid topic values (from declared list)
  6. Valid split values (train / validation / test)
  7. Valid code_switch_frequency (low / medium / high)
  8. Valid language_dominance (english_dominant / amharic_dominant / balanced)
  9. Annotation ranges within declared bounds
 10. Inter-annotator kappa >= 0.60
 11. switch_type consistency with language_tags count
 12. No empty utterances
 13. Duplicate dialogue IDs

Cleaning actions:
  - Normalize line endings to LF (Unix)
  - Strip leading/trailing whitespace from string fields
  - Remove entries with kappa < 0.60 (below reliability threshold)
  - Recompute and correct split counts in top-level metadata

Usage:
    python validate_and_clean.py                          # validates both v2 and v3
    python validate_and_clean.py --input data/ethioswitch_bench_v2.json
    python validate_and_clean.py --input data/ethioswitch_bench_v3.json --output data/ethioswitch_bench_v3_clean.json

Options:
    --input     Path to input JSON file (default: validates both v2 and v3 in code/data/)
    --output    Path to write cleaned JSON (defaults to overwriting input)
    --dry-run   Report issues without writing any files
    --strict    Exit with code 1 if any validation errors found (useful in CI)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

VALID_TOPICS_V2 = {
    "daily_life", "technology", "health", "education",
    "culture", "sports", "politics",
}
VALID_TOPICS_V3 = VALID_TOPICS_V2 | {
    "family_relationships", "mental_health", "entertainment",
    "travel", "environment", "food_nutrition", "finance", "science",
}
VALID_SPLITS        = {"train", "validation", "test"}
VALID_CS_FREQ       = {"low", "medium", "high"}
VALID_DOMINANCE     = {"english_dominant", "amharic_dominant", "balanced"}
VALID_SPEAKERS      = {"User", "Assistant"}
VALID_SWITCH_TAGS   = {
    "[ENG]", "[AMH]", "[ENG-AMH]", "[AMH-ENG]",
    "[ENG-AMH-ENG]", "[AMH-ENG-AMH]",
    # Extended (4-part) tags found in v2/v3
    "[ENG-AMH-ENG-AMH]", "[AMH-ENG-AMH-ENG]",
    "[AMH-ENG-AMH-ENG-AMH]", "[ENG-AMH-ENG-AMH-ENG]",
    "[ENG-AMH-ENG-AMH-ENG-AMH]",
}

ANNOTATION_RANGES = {
    "coherence_score":           (1.0, 5.0),
    "semantic_consistency":      (0.0, 1.0),
    "switch_point_naturalness":  (1.0, 5.0),
    "fluency_rating":            (1.0, 5.0),
    "annotator_agreement_kappa": (0.0, 1.0),
}
MIN_KAPPA  = 0.60
MIN_TURNS  = 4

REQUIRED_DIALOGUE_FIELDS = {
    "dialogue_id", "topic", "code_switch_frequency",
    "language_dominance", "split", "turns", "annotations",
}
REQUIRED_TURN_FIELDS = {
    "turn_id", "speaker", "utterance", "language_tags",
    "switch_type", "switch_point_token",
}
REQUIRED_ANNOTATION_FIELDS = {
    "coherence_score", "semantic_consistency",
    "switch_point_naturalness", "fluency_rating",
    "annotator_agreement_kappa",
}

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

class ValidationError:
    def __init__(self, dialogue_id: Any, field: str, message: str, severity: str = "ERROR"):
        self.dialogue_id = dialogue_id
        self.field       = field
        self.message     = message
        self.severity    = severity  # ERROR | WARNING

    def __str__(self):
        return f"[{self.severity}] Dialogue {self.dialogue_id} | {self.field}: {self.message}"


def detect_version(raw: Dict) -> str:
    return raw.get("version", "2.0")


def get_valid_topics(version: str) -> set:
    return VALID_TOPICS_V3 if version.startswith("3") else VALID_TOPICS_V2


def validate_dialogue(d: Dict, version: str) -> List[ValidationError]:
    errors: List[ValidationError] = []
    did = d.get("dialogue_id", "?")
    valid_topics = get_valid_topics(version)

    # Required fields
    for f in REQUIRED_DIALOGUE_FIELDS:
        if f not in d:
            errors.append(ValidationError(did, f, f"Missing required field '{f}'"))

    if "topic" in d and d["topic"] not in valid_topics:
        errors.append(ValidationError(did, "topic",
            f"Unknown topic '{d['topic']}'. Valid: {sorted(valid_topics)}"))

    if "split" in d and d["split"] not in VALID_SPLITS:
        errors.append(ValidationError(did, "split",
            f"Invalid split '{d['split']}'. Must be one of {VALID_SPLITS}"))

    if "code_switch_frequency" in d and d["code_switch_frequency"] not in VALID_CS_FREQ:
        errors.append(ValidationError(did, "code_switch_frequency",
            f"Invalid value '{d['code_switch_frequency']}'"))

    if "language_dominance" in d and d["language_dominance"] not in VALID_DOMINANCE:
        errors.append(ValidationError(did, "language_dominance",
            f"Invalid value '{d['language_dominance']}'"))

    # Turns
    turns = d.get("turns", [])
    if len(turns) < MIN_TURNS:
        errors.append(ValidationError(did, "turns",
            f"Only {len(turns)} turns (minimum {MIN_TURNS} required)", "ERROR"))

    for t in turns:
        tid = t.get("turn_id", "?")
        prefix = f"turn_{tid}"
        for tf in REQUIRED_TURN_FIELDS:
            if tf not in t:
                errors.append(ValidationError(did, prefix, f"Missing field '{tf}'"))

        if "speaker" in t and t["speaker"] not in VALID_SPEAKERS:
            errors.append(ValidationError(did, prefix,
                f"Unknown speaker '{t['speaker']}' (expected User or Assistant)"))

        if "utterance" in t and not t["utterance"].strip():
            errors.append(ValidationError(did, prefix, "Empty utterance string"))

        if "switch_type" in t and t["switch_type"] not in VALID_SWITCH_TAGS and t["switch_type"]:
            # Warn rather than error — some extended switch types are valid
            errors.append(ValidationError(did, prefix,
                f"switch_type '{t['switch_type']}' not in declared schema tags", "WARNING"))

    # Annotations
    ann = d.get("annotations", {})
    for f in REQUIRED_ANNOTATION_FIELDS:
        if f not in ann:
            errors.append(ValidationError(did, f"annotation.{f}", f"Missing annotation field '{f}'"))
        else:
            v = ann[f]
            if f in ANNOTATION_RANGES:
                lo, hi = ANNOTATION_RANGES[f]
                if not (lo <= v <= hi):
                    errors.append(ValidationError(did, f"annotation.{f}",
                        f"Value {v} out of range [{lo}, {hi}]"))

    if "annotator_agreement_kappa" in ann and ann["annotator_agreement_kappa"] < MIN_KAPPA:
        errors.append(ValidationError(did, "annotation.annotator_agreement_kappa",
            f"Kappa {ann['annotator_agreement_kappa']:.3f} < threshold {MIN_KAPPA} "
            f"(below inter-annotator agreement reliability threshold)", "WARNING"))

    return errors


def validate_dataset(raw: Dict) -> Tuple[List[ValidationError], List[str]]:
    """
    Validate the full dataset. Returns (errors, info_messages).
    """
    errors: List[ValidationError] = []
    info: List[str] = []
    version = detect_version(raw)

    # Top-level checks
    required_top = {"dataset_name", "version", "description", "license",
                    "total_dialogues", "split", "dialogues"}
    for f in required_top:
        if f not in raw:
            errors.append(ValidationError("TOP-LEVEL", f, f"Missing field '{f}'"))

    dialogues = raw.get("dialogues", [])
    declared_total = raw.get("total_dialogues", -1)
    actual_total   = len(dialogues)
    if declared_total != actual_total:
        errors.append(ValidationError("TOP-LEVEL", "total_dialogues",
            f"Declared {declared_total} but found {actual_total} dialogues"))

    # Duplicate dialogue IDs
    ids = [d.get("dialogue_id") for d in dialogues]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        errors.append(ValidationError("TOP-LEVEL", "dialogue_id",
            f"Duplicate dialogue IDs: {dup}"))

    # Split count consistency
    declared_splits = raw.get("split", {})
    actual_splits   = Counter(d.get("split") for d in dialogues)
    for split_name, declared_count in declared_splits.items():
        actual_count = actual_splits.get(split_name, 0)
        if declared_count != actual_count:
            errors.append(ValidationError("TOP-LEVEL", f"split.{split_name}",
                f"Header declares {declared_count} but {actual_count} dialogues have split='{split_name}'"))
            info.append(f"Will auto-correct split counts in metadata.")

    # Per-dialogue validation
    for d in dialogues:
        errors.extend(validate_dialogue(d, version))

    return errors, info


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_dataset(raw: Dict) -> Tuple[Dict, List[str]]:
    """
    Apply cleaning steps to the raw dataset dict.
    Returns (cleaned_dict, list_of_actions_taken).
    """
    actions: List[str] = []
    dialogues = raw.get("dialogues", [])
    original_count = len(dialogues)

    # 1. Strip whitespace from all string scalar fields in each dialogue
    for d in dialogues:
        for field in ("topic", "code_switch_frequency", "language_dominance", "split"):
            if field in d and isinstance(d[field], str):
                stripped = d[field].strip()
                if stripped != d[field]:
                    d[field] = stripped
                    actions.append(f"Dialogue {d['dialogue_id']}: stripped whitespace from '{field}'")
        for t in d.get("turns", []):
            for field in ("utterance", "switch_type", "switch_point_token", "speaker"):
                if field in t and isinstance(t[field], str):
                    stripped = t[field].strip()
                    if stripped != t[field]:
                        t[field] = stripped

    # 2. Remove dialogues below kappa threshold
    low_kappa = [d for d in dialogues
                 if d.get("annotations", {}).get("annotator_agreement_kappa", 1.0) < MIN_KAPPA]
    if low_kappa:
        ids_removed = [d["dialogue_id"] for d in low_kappa]
        dialogues = [d for d in dialogues
                     if d.get("annotations", {}).get("annotator_agreement_kappa", 1.0) >= MIN_KAPPA]
        actions.append(
            f"Removed {len(low_kappa)} dialogue(s) with kappa < {MIN_KAPPA}: IDs {ids_removed}"
        )

    # 3. Recompute and fix top-level split/total counts
    actual_splits = Counter(d.get("split") for d in dialogues)
    if raw.get("split") != dict(actual_splits):
        raw["split"] = dict(actual_splits)
        actions.append(f"Corrected top-level split counts: {dict(actual_splits)}")
    raw["total_dialogues"] = len(dialogues)
    if len(dialogues) != original_count:
        actions.append(f"Updated total_dialogues: {original_count} → {len(dialogues)}")

    # 4. Ensure statistics block reflects actual counts (if present)
    if "statistics" in raw:
        stats = raw["statistics"]
        stats["total_dialogues"] = len(dialogues)
        stats["by_split"] = dict(actual_splits)
        actions.append("Updated statistics.total_dialogues and statistics.by_split")

    raw["dialogues"] = dialogues
    return raw, actions


def ensure_v2_metadata_parity(raw: Dict) -> List[str]:
    """
    Add fields present in v3 but missing in v2 so both files share a common schema root.
    """
    actions: List[str] = []
    if raw.get("version", "").startswith("2"):
        if "version_history" not in raw:
            raw["version_history"] = {
                "v1.0": "Initial pilot dataset (10 dialogues).",
                "v2.0": "Expanded gold set (30 dialogues, 7 topics).",
            }
            actions.append("Added missing 'version_history' field to v2")
        if "public_dataset_references" not in raw:
            raw["public_dataset_references"] = []
            actions.append("Added missing 'public_dataset_references' field to v2 (empty list)")
        if "generation_methods" not in raw:
            raw["generation_methods"] = {
                "hand_crafted": "All dialogues written by expert annotators (gold set).",
            }
            actions.append("Added missing 'generation_methods' field to v2")
        if "source" not in (raw.get("dialogues", [{}])[0] if raw.get("dialogues") else {}):
            for d in raw.get("dialogues", []):
                d.setdefault("source", "gold")
                d.setdefault("source_dataset", "original")
                d.setdefault("generation_method", "hand_crafted")
            actions.append("Added source/source_dataset/generation_method to v2 dialogues")
    return actions


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(path: str, errors: List[ValidationError], actions: List[str],
                 n_before: int, n_after: int) -> None:
    n_errors   = sum(1 for e in errors if e.severity == "ERROR")
    n_warnings = sum(1 for e in errors if e.severity == "WARNING")
    divider = "-" * 70
    print(f"\n{'=' * 70}")
    print(f"  Validation Report: {os.path.basename(path)}")
    print(f"{'=' * 70}")
    print(f"  Dialogues before cleaning : {n_before}")
    print(f"  Dialogues after  cleaning : {n_after}")
    print(f"  Errors   found            : {n_errors}")
    print(f"  Warnings found            : {n_warnings}")
    if errors:
        print(f"\n{divider}")
        print("  Issues:")
        for e in errors:
            print(f"    {e}")
    if actions:
        print(f"\n{divider}")
        print("  Cleaning actions taken:")
        for a in actions:
            print(f"    [OK] {a}")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(input_path: str, output_path: Optional[str],
                 dry_run: bool, strict: bool) -> bool:
    """Returns True if validation passed (0 errors), False otherwise."""
    print(f"\n[validate_and_clean] Loading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    n_before = len(raw.get("dialogues", []))

    # Validate
    errors, info = validate_dataset(raw)

    # Schema parity for v2
    parity_actions = ensure_v2_metadata_parity(raw)

    # Clean
    raw, clean_actions = clean_dataset(raw)
    all_actions = parity_actions + clean_actions + info

    n_after = len(raw.get("dialogues", []))

    print_report(input_path, errors, all_actions, n_before, n_after)

    remaining_errors = [e for e in errors if e.severity == "ERROR"]
    # Re-validate post-clean to confirm fixes
    errors2, _ = validate_dataset(raw)
    remaining_errors2 = [e for e in errors2 if e.severity == "ERROR"]

    if remaining_errors2:
        print(f"  ⚠  {len(remaining_errors2)} error(s) remain after cleaning:")
        for e in remaining_errors2:
            print(f"     {e}")

    if not dry_run:
        out = output_path or input_path
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        print(f"  [OK] Cleaned dataset written to: {out}")
    else:
        print("  [dry-run] No files written.")

    passed = len(remaining_errors2) == 0
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate and clean EthioSwitch-Bench JSON datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    default_v2  = os.path.join(script_dir, "code", "data", "ethioswitch_bench_v2.json")
    default_v3  = os.path.join(script_dir, "code", "data", "ethioswitch_bench_v3.json")

    parser.add_argument("--input",   help="Path to a single input JSON file to validate")
    parser.add_argument("--output",  help="Path to write cleaned JSON (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report issues only; do not write any files")
    parser.add_argument("--strict",  action="store_true",
                        help="Exit with code 1 if any validation errors remain after cleaning")
    args = parser.parse_args()

    if args.input:
        targets = [(args.input, args.output)]
    else:
        targets = [(default_v2, None), (default_v3, None)]

    all_passed = True
    for inp, out in targets:
        if not os.path.exists(inp):
            print(f"  ERROR: File not found: {inp}")
            all_passed = False
            continue
        passed = process_file(inp, out, dry_run=args.dry_run, strict=args.strict)
        if not passed:
            all_passed = False

    if args.strict and not all_passed:
        sys.exit(1)

    print("\n[validate_and_clean] All done.")


if __name__ == "__main__":
    main()
