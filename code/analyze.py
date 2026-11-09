import json
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_PATH = os.path.join(SCRIPT_DIR, "data", "auto_eval_results.json")
HUMAN_PATH = os.path.join(SCRIPT_DIR, "data", "human_eval_results.json")
BENCH_PATH = os.path.join(SCRIPT_DIR, "data", "ethioswitch_bench_v3.json")
OUTPUTS_PATH = os.path.join(SCRIPT_DIR, "data", "model_outputs.json")

def analyze():
    # Load auto results
    with open(AUTO_PATH, "r", encoding="utf-8") as f:
        auto_res = json.load(f)
    
    # Load human results
    with open(HUMAN_PATH, "r", encoding="utf-8") as f:
        human_res = json.load(f)
        
    # Load dataset
    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    # Let's inspect splits in dataset
    dialogues = bench_data["dialogues"]
    print(f"Total dialogues in dataset: {len(dialogues)}")
    
    # Check split of each dialogue
    splits = {}
    dialogues_by_id = {}
    for d in dialogues:
        dialogues_by_id[d["dialogue_id"]] = d
        split = d.get("split", "unknown")
        splits[split] = splits.get(split, 0) + 1
    print("Dialogue splits in dataset:", splits)
    
    models = sorted(list(set(x["model_name"] for x in auto_res)))
    print("Models:", models)

    # The user request asks for:
    # 1. Aggregated performance scores across all models and metrics on the test split.
    # 2. Statistical significance testing using Welch's t-test between models.
    # 3. Evaluation results on the on-bench test set. (Wait, is ethioswitch_bench_v2.json the on-bench test set? Let's check its size and what it contains!)
    # 4. TBERTScore F1 and Coherence scores broken down by code-switch frequency tier.
    # 5. Switch-Point F1 scores analyzed by transition direction (English->Amharic and Amharic->English).

    # Let's see what results we have on the test split (if split is present in dialogues)
    test_dids = [d["dialogue_id"] for d in dialogues if d.get("split") == "test"]
    if not test_dids:
        # If no split is explicitly labeled "test", maybe there is another split or we use all dialogues or first 300 dialogues?
        # Let's check if the split field actually exists in each dialogue object.
        print("Checking first dialogue keys:", dialogues[0].keys())
        # Let's check if "split" exists as a separate top-level key or in the dialogues themselves
        # Wait, the dataset might have a top-level key "split" or dialogues have "split" key. Let's see!
        if "split" in dialogues[0]:
            print("First dialogue split:", dialogues[0]["split"])
        else:
            # Let's check if there are other splits in the bench_data
            print("Top-level keys in bench_data:", bench_data.keys())
            if "split" in bench_data:
                print("bench_data['split']:", bench_data["split"].keys() if isinstance(bench_data["split"], dict) else bench_data["split"])
    
    # Let's print metrics on the test split (dialogues with split == "test")
    # If dialogues don't have "split" field, let's see if we have them in auto_res or if we can define it.
    # Wait, let's write a python script to run this and print everything!
    # Let's group auto eval results and human eval results on the test split
    print("\n--- Test Split Averages ---")
    agg_human = human_res.get("aggregated", {})
    for m in models:
        # Filter auto_res by model and if dialogue is in test split
        # If no split is found, use all dialogues
        m_auto = [x for x in auto_res if x["model_name"] == m and (not test_dids or x["dialogue_id"] in test_dids)]
        bleu = [x["bleu"] for x in m_auto]
        rouge = [x["rouge_l"] for x in m_auto]
        bert = [x["bert_f1"] for x in m_auto]
        sw_f1 = [x["switch_accuracy_f1"] for x in m_auto]
        ppl = [x["perplexity"] for x in m_auto]
        
        coh = []
        flu = []
        nat = []
        if m in agg_human:
            for did, h_data in agg_human[m].items():
                did_int = int(did)
                if not test_dids or did_int in test_dids:
                    coh.append(h_data["coherence"])
                    flu.append(h_data["fluency"])
                    nat.append(h_data["switch_naturalness"])
        
        print(f"Model: {m}")
        print(f"  BLEU: {np.mean(bleu):.4f}  {np.std(bleu):.4f}")
        print(f"  ROUGE-L: {np.mean(rouge):.4f}  {np.std(rouge):.4f}")
        print(f"  BERTScore F1: {np.mean(bert):.4f}  {np.std(bert):.4f}")
        print(f"  Switch F1: {np.mean(sw_f1):.4f}  {np.std(sw_f1):.4f}")
        print(f"  Perplexity: {np.mean(ppl):.2f}  {np.std(ppl):.2f}")
        print(f"  Human Coherence: {np.mean(coh):.4f}  {np.std(coh):.4f}" if coh else "  Human Coherence: N/A")
        print(f"  Human Fluency: {np.mean(flu):.4f}  {np.std(flu):.4f}" if flu else "  Human Fluency: N/A")
        print(f"  Human Switch Naturalness: {np.mean(nat):.4f}  {np.std(nat):.4f}" if nat else "  Human Switch Naturalness: N/A")

    # Let's inspect the ethioswitch_bench_v2.json to see what it is
    v2_file = os.path.join(SCRIPT_DIR, "data", "ethioswitch_bench_v2.json")
    if os.path.exists(v2_file):
        with open(v2_file, "r", encoding="utf-8") as f:
            v2_data = json.load(f)
        print(f"\nethioswitch_bench_v2.json dialogues: {len(v2_data['dialogues'])}")
        # Let's see if we have results for v2 dialogues in auto_res (dialogue_id 1 to 30?)
        v2_dids = [d["dialogue_id"] for d in v2_data["dialogues"]]
        print("ethioswitch_bench_v2.json dialogue IDs:", v2_dids)
        
        # Let's compute scores for v2 dialogues
        print("\n--- v2.0 Benchmark (Original Hand-crafted 30 Dialogues) Averages ---")
        for m in models:
            m_auto = [x for x in auto_res if x["model_name"] == m and x["dialogue_id"] in v2_dids]
            bleu = [x["bleu"] for x in m_auto]
            rouge = [x["rouge_l"] for x in m_auto]
            bert = [x["bert_f1"] for x in m_auto]
            sw_f1 = [x["switch_accuracy_f1"] for x in m_auto]
            ppl = [x["perplexity"] for x in m_auto]
            
            coh = []
            if m in agg_human:
                for did, h_data in agg_human[m].items():
                    did_int = int(did)
                    if did_int in v2_dids:
                        coh.append(h_data["coherence"])
            print(f"Model: {m} | BLEU: {np.mean(bleu):.4f} | ROUGE-L: {np.mean(rouge):.4f} | BERTScore F1: {np.mean(bert):.4f} | Switch F1: {np.mean(sw_f1):.4f} | PPL: {np.mean(ppl):.2f} | Coherence: {np.mean(coh):.2f}")

    # Let's break down Switch F1 by transition direction:
    # Wait, how do we get transition direction for each dialogue?
    # In dialogue turns, the first code-switched turn has a switch_type like [ENG-AMH] or [AMH-ENG]
    # Let's check which switch direction each dialogue has, and compute the Switch F1 for that direction!
    # Wait, how does 03_auto_evaluator.py compute switch F1?
    # It compares predicted switch type (e.g. from detect_switch_type(gen)) with gold switch type from GOLD_SWITCH_TAGS.
    # So for each dialogue, we have a gold switch type (like [ENG-AMH] or [AMH-ENG] or [ENG-AMH-ENG] or [AMH-ENG-AMH]).
    # Let's group dialogues by their gold switch type, and compute the Switch F1 for each group!
    print("\n--- Switch F1 by Gold Switch Type ---")
    # Gold switch types in the dataset are:
    # [ENG-AMH], [AMH-ENG], [ENG-AMH-ENG], [AMH-ENG-AMH], etc.
    # For a model, we can compute F1 score for each gold switch type.
    # Let's calculate:
    # predicted tag is in o["detected_switch_type"] or x["switch_accuracy_f1"] is computed on a dialogue level.
    # Wait, on a dialogue level, how is switch_accuracy_f1 computed?
    # In 03_auto_evaluator.py:
    # switch_f1 = compute_switch_f1([pred_switch], [gold_switch])
    # Since it's a single prediction and single reference, if they match, F1 is 1.0, otherwise 0.0!
    # So the average switch_accuracy_f1 on a group of dialogues is just the accuracy of predicting the switch type!
    # Let's check if this is the case.
    # Yes! Let's compute accuracy/F1 for each transition type for each model:
    # Transition types to analyze:
    # 1. English -> Amharic (like [ENG-AMH] or [ENG-AMH-ENG])
    # 2. Amharic -> English (like [AMH-ENG] or [AMH-ENG-AMH])
    # Let's group by gold switch type starting with ENG vs AMH or containing ENG-AMH vs AMH-ENG.
    # Let's write code to compute the accuracy (average of switch_accuracy_f1) for each model for:
    # - Gold switch types: [ENG-AMH] or [ENG-AMH-ENG] (which represent English to Amharic transition directions)
    # - Gold switch types: [AMH-ENG] or [AMH-ENG-AMH] (which represent Amharic to English transition directions)
    
    # Let's get the gold switch tag for each dialogue
    gold_switch_tags = {}
    for d in dialogues:
        turns = d.get("turns", [])
        gold_switch_tags[d["dialogue_id"]] = turns[1]["switch_type"] if len(turns) > 1 else "[UNK]"
        
    print("\n--- Switch F1 by Transition Direction ---")
    for m in models:
        print(f"Model: {m}")
        # English -> Amharic transitions: Gold switch type is [ENG-AMH] or starts with [ENG-AMH]
        eng_amh_dids = [did for did, tag in gold_switch_tags.items() if "[ENG-AMH]" in tag]
        # Amharic -> English transitions: Gold switch type is [AMH-ENG] or starts with [AMH-ENG]
        amh_eng_dids = [did for did, tag in gold_switch_tags.items() if "[AMH-ENG]" in tag]
        
        eng_amh_scores = [x["switch_accuracy_f1"] for x in auto_res if x["model_name"] == m and x["dialogue_id"] in eng_amh_dids]
        amh_eng_scores = [x["switch_accuracy_f1"] for x in auto_res if x["model_name"] == m and x["dialogue_id"] in amh_eng_dids]
        
        print(f"  ENG->AMH transitions: {np.mean(eng_amh_scores):.4f}  {np.std(eng_amh_scores):.4f} (n={len(eng_amh_scores)})")
        print(f"  AMH->ENG transitions: {np.mean(amh_eng_scores):.4f}  {np.std(amh_eng_scores):.4f} (n={len(amh_eng_scores)})")

    
if __name__ == "__main__":
    analyze()
