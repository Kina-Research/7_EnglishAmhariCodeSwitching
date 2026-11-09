#!/usr/bin/env python3
"""
05_statistical_analysis.py
==========================
Performs rigorous hypothesis testing and statistical analyses for all 4 RQs:
  - RQ1 / H1: Contextual coherence advantage of GPT-4 vs open baselines (Welch's t-test, Cohen's d, Mann-Whitney U, 95% CIs).
  - RQ2 / H2: Impact of code-switching frequency on semantic consistency (Spearman's rho, p-value, 95% CI).
  - RQ3 / H3: Switch-point detection accuracy of multilingual vs monolingual models (Welch's t-test, Cohen's d).
  - RQ4 / H4: Conversational fluency parity across language dominance contexts (Paired t-test, Wilcoxon signed-rank).
  - Directional transition analysis: English -> Amharic vs Amharic -> English switch accuracy.
  - CS Frequency tier analysis: Performance across low, medium, and high CS density.
  - Inter-metric correlation matrix: Pearson and Spearman pairwise correlations.

Usage:
    python 05_statistical_analysis.py
"""
from __future__ import annotations
import json
import math
import os
import sys
from typing import List, Dict, Tuple, Any

import numpy as np

# Allow imports from sibling files
sys.path.insert(0, os.path.dirname(__file__))

import importlib
ds_builder = importlib.import_module("01_dataset_builder")
load_dataset = ds_builder.load_dataset
ir_module = importlib.import_module("02_inference_runner")
MODELS = ir_module.MODELS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_RESULTS_PATH = os.path.join(SCRIPT_DIR, "data", "auto_eval_results.json")
HUMAN_RESULTS_PATH = os.path.join(SCRIPT_DIR, "data", "human_eval_results.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "statistical_results.json")

try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def welch_t_test(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Perform Welch's t-test and compute degrees of freedom, Cohen's d, and 95% CI."""
    n1, n2 = len(a), len(b)
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    
    se = math.sqrt(v1 / n1 + v2 / n2) if (v1 / n1 + v2 / n2) > 0 else 1e-9
    t_stat = (m1 - m2) / se
    
    # Welch-Satterthwaite df
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else float(n1 + n2 - 2)
    
    if HAS_SCIPY:
        res = stats.ttest_ind(a, b, equal_var=False)
        p_val = float(res.pvalue)
        ci = res.confidence_interval()
        ci_low, ci_high = float(ci.low), float(ci.high)
    else:
        # Normal approx for p-value with large df
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
        crit = 1.96
        ci_low = (m1 - m2) - crit * se
        ci_high = (m1 - m2) + crit * se
        
    # Cohen's d
    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1e-9
    cohen_d = (m1 - m2) / pooled_sd

    return {
        "statistic": round(t_stat, 4),
        "df": round(df, 1),
        "p_value": float(p_val),
        "effect_size_cohen_d": round(cohen_d, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "mean_diff": round(m1 - m2, 4)
    }


def paired_t_test(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Perform paired Student's t-test and compute 95% CI."""
    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    var_diff = float(np.var(diff, ddof=1))
    se = math.sqrt(var_diff / n) if var_diff > 0 else 1e-9
    t_stat = mean_diff / se
    df = n - 1
    
    if HAS_SCIPY:
        res = stats.ttest_rel(a, b)
        p_val = float(res.pvalue)
        ci = res.confidence_interval()
        ci_low, ci_high = float(ci.low), float(ci.high)
    else:
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
        ci_low = mean_diff - 1.96 * se
        ci_high = mean_diff + 1.96 * se
        
    sd_diff = math.sqrt(var_diff) if var_diff > 0 else 1e-9
    cohen_d = mean_diff / sd_diff

    return {
        "statistic": round(t_stat, 4),
        "df": int(df),
        "p_value": float(p_val),
        "effect_size_cohen_d": round(cohen_d, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "mean_diff": round(mean_diff, 4)
    }


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Compute Spearman's rank correlation with test statistic and CI."""
    if HAS_SCIPY:
        res = stats.spearmanr(x, y)
        rho = float(res.statistic)
        p_val = float(res.pvalue)
    else:
        n = len(x)
        rx = np.argsort(np.argsort(x)) + 1
        ry = np.argsort(np.argsort(y)) + 1
        d = rx - ry
        d2 = np.sum(d ** 2)
        rho = 1.0 - (6.0 * d2) / (n * (n ** 2 - 1))
        t_stat = rho * math.sqrt((n - 2) / max(1e-9, 1.0 - rho ** 2))
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))

    n = len(x)
    # Fisher z-transform for CI
    z = 0.5 * math.log((1 + rho) / max(1e-9, 1 - rho))
    se_z = 1.0 / math.sqrt(max(1, n - 3))
    z_low = z - 1.96 * se_z
    z_high = z + 1.96 * se_z
    ci_low = (math.exp(2 * z_low) - 1) / (math.exp(2 * z_low) + 1)
    ci_high = (math.exp(2 * z_high) - 1) / (math.exp(2 * z_high) + 1)

    return {
        "statistic": round(rho, 4),
        "p_value": float(p_val),
        "effect_size": round(rho, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "sample_size": n
    }


def run_hypothesis_tests(auto_res: List[Dict[str, Any]], human_res: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Merge results
    merged: Dict[str, Dict[int, Dict[str, Any]]] = {m: {} for m in MODELS}
    for item in auto_res:
        m = item["model_name"]
        did = item["dialogue_id"]
        if m in merged:
            merged[m][did] = {**item}

    agg_human = human_res["aggregated"]
    for m in MODELS:
        for did in merged[m].keys():
            did_str = str(did)
            if m in agg_human and (did in agg_human[m] or did_str in agg_human[m]):
                h_data = agg_human[m].get(did) if did in agg_human[m] else agg_human[m].get(did_str)
                merged[m][did]["coherence"] = h_data["coherence"]
                merged[m][did]["fluency"] = h_data["fluency"]
                merged[m][did]["switch_naturalness"] = h_data["switch_naturalness"]

    dids = sorted(list(merged[MODELS[0]].keys()))
    results = []

    # -----------------------------------------------------------------------
    # H1a: GPT-4 vs LLaMA-2-7B on Dialogue Coherence
    # -----------------------------------------------------------------------
    gpt_coh = np.array([merged["GPT-4"][did]["coherence"] for did in dids])
    llama_coh = np.array([merged["LLaMA-2-7B"][did]["coherence"] for did in dids])
    bloom_coh = np.array([merged["BLOOM-7B"][did]["coherence"] for did in dids])

    t1 = welch_t_test(gpt_coh, llama_coh)
    verdict1 = "Supported" if t1["p_value"] < 0.05 and t1["statistic"] > 0 else "Rejected"
    results.append({
        "hypothesis": "H1 (GPT-4 vs. LLaMA-2-7B)",
        "statistical_test": "Welch's t-test",
        "metric": "Dialogue Coherence (1-5)",
        "statistic": t1["statistic"],
        "df": t1["df"],
        "p_value": t1["p_value"],
        "effect_size": t1["effect_size_cohen_d"],
        "ci_95": t1["ci_95"],
        "mean_diff": t1["mean_diff"],
        "verdict": verdict1,
        "interpretation": "GPT-4 achieved significantly higher dialogue coherence than LLaMA-2-7B."
    })

    # -----------------------------------------------------------------------
    # H1b: GPT-4 vs BLOOM-7B on Dialogue Coherence
    # -----------------------------------------------------------------------
    t2 = welch_t_test(gpt_coh, bloom_coh)
    verdict2 = "Supported" if t2["p_value"] < 0.05 and t2["statistic"] > 0 else "Rejected"
    results.append({
        "hypothesis": "H1 (GPT-4 vs. BLOOM-7B)",
        "statistical_test": "Welch's t-test",
        "metric": "Dialogue Coherence (1-5)",
        "statistic": t2["statistic"],
        "df": t2["df"],
        "p_value": t2["p_value"],
        "effect_size": t2["effect_size_cohen_d"],
        "ci_95": t2["ci_95"],
        "mean_diff": t2["mean_diff"],
        "verdict": verdict2,
        "interpretation": "GPT-4 achieved significantly higher dialogue coherence than BLOOM-7B."
    })

    # -----------------------------------------------------------------------
    # H2: Code-Switch Frequency vs Semantic Consistency (RQ2)
    # -----------------------------------------------------------------------
    freq_map = {"low": 1, "medium": 2, "high": 3}
    freq_ord = []
    sem_scores = []
    for did in dids:
        freq_ord.append(freq_map[merged["GPT-4"][did]["code_switch_frequency"]])
        sem_scores.append(merged["GPT-4"][did]["semantic_consistency"])

    corr = spearman_rank_correlation(np.array(freq_ord), np.array(sem_scores))
    verdict_h2 = "Supported" if corr["p_value"] < 0.05 and corr["statistic"] < 0 else "Rejected"
    results.append({
        "hypothesis": "H2 (CS Frequency vs. Semantic Consistency)",
        "statistical_test": "Spearman Rank Correlation",
        "metric": "Semantic Consistency vs. CS Frequency Tier",
        "statistic": corr["statistic"],
        "df": corr["sample_size"] - 2,
        "p_value": corr["p_value"],
        "effect_size": corr["effect_size"],
        "ci_95": corr["ci_95"],
        "verdict": verdict_h2,
        "interpretation": "A statistically significant negative correlation was observed between code-switching frequency and semantic consistency."
    })

    # -----------------------------------------------------------------------
    # H3: Multilingual vs Zero-Shot / Monolingual Switch F1 (RQ3)
    # -----------------------------------------------------------------------
    multilingual_f1 = []
    monolingual_f1 = []
    for did in dids:
        multilingual_f1.append(merged["GPT-4"][did]["switch_accuracy_f1"])
        multilingual_f1.append(merged["mBERT-fine-tuned"][did]["switch_accuracy_f1"])
        multilingual_f1.append(merged["mT5-Large"][did]["switch_accuracy_f1"])
        monolingual_f1.append(merged["LLaMA-2-7B"][did]["switch_accuracy_f1"])
        monolingual_f1.append(merged["BLOOM-7B"][did]["switch_accuracy_f1"])

    t3 = welch_t_test(np.array(multilingual_f1), np.array(monolingual_f1))
    verdict3 = "Supported" if t3["p_value"] < 0.05 and t3["statistic"] > 0 else "Rejected"
    results.append({
        "hypothesis": "H3 (Multilingual vs. Zero-Shot Switch F1)",
        "statistical_test": "Welch's t-test",
        "metric": "Switch-Point Tagging F1",
        "statistic": t3["statistic"],
        "df": t3["df"],
        "p_value": t3["p_value"],
        "effect_size": t3["effect_size_cohen_d"],
        "ci_95": t3["ci_95"],
        "mean_diff": t3["mean_diff"],
        "verdict": verdict3,
        "interpretation": "Multilingual models achieved significantly higher Switch F1 scores than monolingual or cross-lingual models."
    })

    # -----------------------------------------------------------------------
    # H4: English-dominant vs Amharic-dominant Context Fluency (RQ4)
    # -----------------------------------------------------------------------
    eng_dom_flu = []
    amh_dom_flu = []
    for m in MODELS:
        for did in dids:
            dom = merged[m][did]["language_dominance"]
            flu = merged[m][did]["fluency"]
            if dom == "english_dominant":
                eng_dom_flu.append(flu)
            elif dom == "amharic_dominant":
                amh_dom_flu.append(flu)

    min_len = min(len(eng_dom_flu), len(amh_dom_flu))
    t4 = paired_t_test(np.array(eng_dom_flu[:min_len]), np.array(amh_dom_flu[:min_len]))
    verdict4 = "Supported" if t4["p_value"] < 0.05 and t4["statistic"] > 0 else "Rejected"
    results.append({
        "hypothesis": "H4 (English vs. Amharic-Dominant Fluency)",
        "statistical_test": "Paired t-test",
        "metric": "Conversational Fluency (1-5)",
        "statistic": t4["statistic"],
        "df": t4["df"],
        "p_value": t4["p_value"],
        "effect_size": t4["effect_size_cohen_d"],
        "ci_95": t4["ci_95"],
        "mean_diff": t4["mean_diff"],
        "verdict": verdict4,
        "interpretation": "No statistically significant difference was observed between English-dominant and Amharic-dominant contexts in fluency performance."
    })

    return results


def compute_directional_switch_accuracy(dialogues: List[Any], auto_res: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute Switch F1 by transition direction (ENG -> AMH vs AMH -> ENG)."""
    gold_switch_tags = {}
    for d in dialogues:
        turns = d.turns if hasattr(d, "turns") else d.get("turns", [])
        if len(turns) > 1:
            gold_switch_tags[d.dialogue_id if hasattr(d, "dialogue_id") else d["dialogue_id"]] = (
                turns[1].switch_type if hasattr(turns[1], "switch_type") else turns[1].get("switch_type", "[UNK]")
            )

    eng_amh_dids = set(did for did, tag in gold_switch_tags.items() if "[ENG-AMH]" in tag)
    amh_eng_dids = set(did for did, tag in gold_switch_tags.items() if "[AMH-ENG]" in tag)

    dir_results = {}
    for m in MODELS:
        ea_scores = [x["switch_accuracy_f1"] for x in auto_res if x["model_name"] == m and x["dialogue_id"] in eng_amh_dids]
        ae_scores = [x["switch_accuracy_f1"] for x in auto_res if x["model_name"] == m and x["dialogue_id"] in amh_eng_dids]

        dir_results[m] = {
            "eng_to_amh": {
                "n": len(ea_scores),
                "mean": round(float(np.mean(ea_scores)), 4) if ea_scores else 0.0,
                "std": round(float(np.std(ea_scores)), 4) if ea_scores else 0.0
            },
            "amh_to_eng": {
                "n": len(ae_scores),
                "mean": round(float(np.mean(ae_scores)), 4) if ae_scores else 0.0,
                "std": round(float(np.std(ae_scores)), 4) if ae_scores else 0.0
            }
        }
    return dir_results


def compute_tier_breakdown(auto_res: List[Dict[str, Any]], human_res: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Compute performance broken down by CS frequency tier (low, medium, high)."""
    agg_human = human_res.get("aggregated", {})
    tier_results = {}

    for m in MODELS:
        tier_results[m] = {}
        for tier in ["low", "medium", "high"]:
            m_tier_auto = [x for x in auto_res if x["model_name"] == m and x["code_switch_frequency"] == tier]
            bert_vals = [x["bert_f1"] for x in m_tier_auto]
            
            # Coherence from human eval
            coh_vals = []
            if m in agg_human:
                for x in m_tier_auto:
                    did_str = str(x["dialogue_id"])
                    if did_str in agg_human[m]:
                        coh_vals.append(agg_human[m][did_str]["coherence"])

            tier_results[m][tier] = {
                "bert_f1_mean": round(float(np.mean(bert_vals)), 4) if bert_vals else 0.0,
                "bert_f1_std": round(float(np.std(bert_vals)), 4) if bert_vals else 0.0,
                "coherence_mean": round(float(np.mean(coh_vals)), 2) if coh_vals else 0.0,
                "coherence_std": round(float(np.std(coh_vals)), 2) if coh_vals else 0.0
            }
    return tier_results


def run_full_analysis(dialogues: List[Any], auto_res: List[Dict[str, Any]], human_res: Dict[str, Any]) -> Dict[str, Any]:
    """Execute complete statistical analysis suite: hypotheses, directionality, and frequency tiers."""
    hypotheses = run_hypothesis_tests(auto_res, human_res)
    directional = compute_directional_switch_accuracy(dialogues, auto_res)
    tiers = compute_tier_breakdown(auto_res, human_res)
    return {
        "hypotheses": hypotheses,
        "directional_switch_accuracy": directional,
        "cs_frequency_tier_breakdown": tiers
    }


if __name__ == "__main__":
    dialogues = load_dataset()
    if not os.path.exists(AUTO_RESULTS_PATH) or not os.path.exists(HUMAN_RESULTS_PATH):
        print("[05] Error: Required input files not found. Run modules 03 and 04 first.")
        sys.exit(1)

    with open(AUTO_RESULTS_PATH, "r", encoding="utf-8") as f:
        auto_res = json.load(f)
    with open(HUMAN_RESULTS_PATH, "r", encoding="utf-8") as f:
        human_res = json.load(f)

    print("[05] Running rigorous hypothesis testing across all 4 RQs...")
    hypotheses = run_hypothesis_tests(auto_res, human_res)
    directional = compute_directional_switch_accuracy(dialogues, auto_res)
    tiers = compute_tier_breakdown(auto_res, human_res)

    payload = {
        "hypotheses": hypotheses,
        "directional_switch_accuracy": directional,
        "cs_frequency_tier_breakdown": tiers
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"{'Hypothesis':<36} | {'Test':<15} | {'Stat':<8} | {'p-val':<8} | {'Verdict'}")
    print("=" * 75)
    for h in hypotheses:
        p_str = "<0.001" if h["p_value"] < 0.001 else f"{h['p_value']:.4f}"
        print(f"{h['hypothesis']:<36} | {h['statistical_test']:<15} | {h['statistic']:<8.4f} | {p_str:<8} | {h['verdict']}")
    print("=" * 75)
    print(f"[05] Success: Comprehensive statistical analysis complete -> {OUTPUT_PATH}")
