"""
test_dataset.py
===============
Unit and integration tests for EthioSwitch-Bench dataset structure,
schema validation, linguistics checks, and distribution balance.
"""
import json
import os
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH_V3 = os.path.join(PROJECT_ROOT, "code", "data", "ethioswitch_bench_v3.json")
DATA_PATH_V2 = os.path.join(PROJECT_ROOT, "code", "data", "ethioswitch_bench_v2.json")

EXPECTED_TOPICS = {
    "daily_life", "technology", "health", "education", "culture",
    "sports", "politics", "family_relationships", "mental_health",
    "entertainment", "travel", "environment", "food_nutrition",
    "finance", "science"
}

VALID_DOMINANCE = {"english_dominant", "amharic_dominant", "balanced"}
VALID_FREQUENCY = {"low", "medium", "high"}
VALID_SPLITS = {"train", "validation", "test"}


@pytest.fixture(scope="session")
def dataset_v3():
    assert os.path.exists(DATA_PATH_V3), f"Dataset file not found: {DATA_PATH_V3}"
    with open(DATA_PATH_V3, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="session")
def dataset_v2():
    assert os.path.exists(DATA_PATH_V2), f"Dataset file not found: {DATA_PATH_V2}"
    with open(DATA_PATH_V2, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_v3_dialogue_count_and_turns(dataset_v3):
    dialogues = dataset_v3.get("dialogues", [])
    assert len(dialogues) == 2000, f"Expected 2000 dialogues, got {len(dialogues)}"
    
    total_turns = sum(len(d.get("turns", [])) for d in dialogues)
    assert total_turns == 10000, f"Expected 10,000 total turns, got {total_turns}"
    
    for d in dialogues:
        assert len(d.get("turns", [])) >= 4, f"Dialogue {d.get('dialogue_id')} has fewer than 4 turns"


def test_v3_schema_integrity(dataset_v3):
    required_keys = {"dialogue_id", "topic", "code_switch_frequency", "language_dominance", "split", "turns", "annotations"}
    for d in dataset_v3.get("dialogues", []):
        assert required_keys.issubset(d.keys()), f"Missing keys in dialogue {d.get('dialogue_id')}"
        assert d["topic"] in EXPECTED_TOPICS, f"Unexpected topic: {d['topic']}"
        assert d["language_dominance"] in VALID_DOMINANCE, f"Invalid dominance: {d['language_dominance']}"
        assert d["code_switch_frequency"] in VALID_FREQUENCY, f"Invalid frequency: {d['code_switch_frequency']}"
        assert d["split"] in VALID_SPLITS, f"Invalid split: {d['split']}"
        
        # Check annotations
        ann = d["annotations"]
        assert 1.0 <= ann["coherence_score"] <= 5.0
        assert 0.0 <= ann["semantic_consistency"] <= 1.0
        assert 1.0 <= ann["fluency_rating"] <= 5.0
        assert 1.0 <= ann["switch_point_naturalness"] <= 5.0
        assert -1.0 <= ann["annotator_agreement_kappa"] <= 1.0


def test_v3_split_distribution(dataset_v3):
    splits = {}
    for d in dataset_v3.get("dialogues", []):
        s = d.get("split")
        splits[s] = splits.get(s, 0) + 1
    
    assert splits.get("train") == 1400, f"Expected 1400 train dialogues, got {splits.get('train')}"
    assert splits.get("validation") == 298, f"Expected 298 validation dialogues, got {splits.get('validation')}"
    assert splits.get("test") == 302, f"Expected 302 test dialogues, got {splits.get('test')}"


def test_v2_gold_benchmark(dataset_v2):
    dialogues = dataset_v2.get("dialogues", [])
    assert len(dialogues) == 30, f"Expected 30 gold dialogues in v2, got {len(dialogues)}"
    for d in dialogues:
        assert len(d.get("turns", [])) == 5, "Expected 5 turns per gold dialogue"
        for t in d["turns"]:
            assert t["utterance"].strip() != "", "Empty utterance found"

