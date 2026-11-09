"""
test_pipeline.py
================
Integration tests verifying pipeline artifacts, figure generation,
and report consistency.
"""
import pytest
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "code", "data")
FIG_DIR = os.path.join(DATA_DIR, "figures")


def test_artifact_files_exist():
    expected_files = [
        "auto_eval_results.json",
        "human_eval_results.json",
        "statistical_results.json",
        "research_summary.md",
        "model_outputs.json"
    ]
    for fn in expected_files:
        p = os.path.join(DATA_DIR, fn)
        assert os.path.exists(p), f"Artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Artifact empty: {p}"


def test_figure_files_exist():
    expected_figs = [
        "fig1_model_comparison_bar.png", "fig1_model_comparison_bar.pdf",
        "fig2_cs_freq_vs_consistency.png", "fig2_cs_freq_vs_consistency.pdf",
        "fig3_multi_metric_radar.png", "fig3_multi_metric_radar.pdf",
        "fig4_coherence_distribution_box.png", "fig4_coherence_distribution_box.pdf",
        "fig5_fluency_language_dominance_bar.png", "fig5_fluency_language_dominance_bar.pdf",
        "fig6_metric_correlation_heatmap.png", "fig6_metric_correlation_heatmap.pdf"
    ]
    for fig in expected_figs:
        p = os.path.join(FIG_DIR, fig)
        assert os.path.exists(p), f"Figure missing: {p}"
        assert os.path.getsize(p) > 1000, f"Figure corrupted/empty: {p}"


def test_statistical_results_schema():
    p = os.path.join(DATA_DIR, "statistical_results.json")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "hypotheses" in data
    assert "directional_switch_accuracy" in data
    assert "cs_frequency_tier_breakdown" in data
    assert len(data["hypotheses"]) == 5
    for h in data["hypotheses"]:
        assert "verdict" in h
        assert "statistic" in h
        assert "p_value" in h
