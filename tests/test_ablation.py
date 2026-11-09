"""
test_ablation.py
================
Unit tests for the ablation and sensitivity analysis module (08_ablation_and_sensitivity.py).
"""
import pytest
import os
import json
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

import importlib
ablation_mod = importlib.import_module("08_ablation_and_sensitivity")


def test_few_shot_ablation():
    res = ablation_mod.run_few_shot_ablation()
    assert "GPT-4" in res
    assert "k_0" in res["GPT-4"]
    assert "k_5" in res["GPT-4"]
    assert res["GPT-4"]["k_5"]["coherence"] >= res["GPT-4"]["k_0"]["coherence"]


def test_temperature_sensitivity():
    res = ablation_mod.run_temperature_sensitivity()
    assert "LLaMA-2-7B" in res
    assert "T_0.0" in res["LLaMA-2-7B"]
    assert "T_1.0" in res["LLaMA-2-7B"]
    # Higher temperature should yield higher script hallucination in open models
    assert res["LLaMA-2-7B"]["T_1.0"]["script_hallucination_pct"] > res["LLaMA-2-7B"]["T_0.0"]["script_hallucination_pct"]


def test_tokenizer_fertility():
    fert = ablation_mod.compute_tokenizer_fertility_analysis()
    llama_key = [k for k in fert if "LLaMA-2" in k][0]
    mbert_key = [k for k in fert if "mBERT" in k][0]
    # LLaMA should have higher fertility disparity than mBERT due to byte fallback
    assert fert[llama_key]["fertility_disparity_ratio"] > fert[mbert_key]["fertility_disparity_ratio"]
