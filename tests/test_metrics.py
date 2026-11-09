"""
test_metrics.py
===============
Unit tests for evaluation metrics: BLEU, ROUGE-L, Switch F1,
and quadratically weighted Cohen's Kappa.
"""
import pytest
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from utils.metrics import (
    compute_bleu,
    compute_rouge_l,
    compute_weighted_cohen_kappa,
    compute_perplexity_stub
)
from utils.text_utils import compute_switch_f1, detect_switch_type, has_amharic, has_english


def test_bleu_identical():
    text = "Machine learning is useful ለብዙ ስራዎች"
    score = compute_bleu(text, text)
    assert score > 0.90, f"Expected near 1.0 BLEU for identical strings, got {score}"


def test_bleu_empty():
    assert compute_bleu("", "Hello world") == 0.0
    assert compute_bleu("Hello world", "") == 0.0


def test_rouge_identical():
    text = "ML is a subset of AI. ኮምፒዩተሮች patterns ይማራሉ from data."
    score = compute_rouge_l(text, text)
    assert score == 1.0


def test_switch_f1_exact():
    preds = ["[ENG-AMH]", "[AMH-ENG]"]
    refs = ["[ENG-AMH]", "[AMH-ENG]"]
    assert compute_switch_f1(preds, refs) == 1.0


def test_switch_f1_mismatch():
    preds = ["[ENG-AMH]", "[ENG]"]
    refs = ["[AMH-ENG]", "[AMH]"]
    assert compute_switch_f1(preds, refs) == 0.0


def test_weighted_cohen_kappa():
    # Perfect agreement
    r1 = [1, 2, 3, 4, 5]
    r2 = [1, 2, 3, 4, 5]
    assert compute_weighted_cohen_kappa(r1, r2, weight="quadratic") == 1.0

    # Near agreement on ordinal scale
    r_a = [4, 4, 5, 3, 4, 5, 2, 4, 3, 5]
    r_b = [4, 5, 5, 3, 4, 4, 2, 4, 3, 5]
    kappa = compute_weighted_cohen_kappa(r_a, r_b, weight="quadratic")
    assert kappa > 0.70, f"Expected high weighted kappa for minor 1-step differences, got {kappa}"


def test_script_detection():
    assert has_amharic("ሰላም") is True
    assert has_amharic("Hello") is False
    assert has_english("Hello") is True
    assert has_english("ሰላም") is False
    assert detect_switch_type("ሰላም! How are you?") in ["[AMH-ENG]", "[ENG-AMH]"]
