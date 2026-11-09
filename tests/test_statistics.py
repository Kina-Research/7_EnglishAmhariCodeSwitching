"""
test_statistics.py
==================
Unit tests for hypothesis testing functions, Welch's t-test,
paired t-test, Spearman rank correlation, degrees of freedom, and confidence intervals.
"""
import pytest
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

import importlib
stats_mod = importlib.import_module("05_statistical_analysis")
welch_t_test = stats_mod.welch_t_test
paired_t_test = stats_mod.paired_t_test
spearman_rank_correlation = stats_mod.spearman_rank_correlation


def test_welch_t_test():
    np.random.seed(42)
    group1 = np.random.normal(4.8, 0.3, 100)
    group2 = np.random.normal(2.8, 0.5, 100)
    res = welch_t_test(group1, group2)
    assert res["statistic"] > 20.0
    assert res["p_value"] < 0.001
    assert res["effect_size_cohen_d"] > 3.0
    assert res["ci_95"][0] > 1.8
    assert res["ci_95"][1] < 2.2


def test_paired_t_test_null():
    np.random.seed(42)
    base = np.random.normal(4.0, 0.5, 100)
    noise = np.random.normal(0, 0.05, 100)
    res = paired_t_test(base, base + noise)
    assert res["df"] == 99
    assert -0.2 < res["statistic"] < 0.2 or res["p_value"] > 0.05


def test_spearman_correlation():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    res = spearman_rank_correlation(x, y)
    assert res["statistic"] == -1.0
    assert res["p_value"] < 0.001
    assert res["ci_95"][1] < -0.90

