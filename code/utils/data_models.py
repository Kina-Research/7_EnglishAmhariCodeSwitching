"""
EthioSwitch-Bench Research Pipeline
====================================
Shared data models (dataclasses) used across all pipeline modules.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class DialogueTurn:
    """Represents a single conversational turn."""
    turn_id: int
    speaker: str                          # "User" or "Assistant"
    utterance: str
    language_tags: List[str]              # e.g. ["[ENG]", "[AMH]"]
    switch_type: str                      # e.g. "[ENG-AMH]"
    switch_point_token: str = ""          # First token at the switch boundary


@dataclass
class Dialogue:
    """Represents a complete annotated dialogue."""
    dialogue_id: int
    topic: str
    code_switch_frequency: str            # "low" | "medium" | "high"
    language_dominance: str               # "english_dominant" | "amharic_dominant" | "balanced"
    turns: List[DialogueTurn]
    # Human-validated annotations
    coherence_score: float                # 1-5
    semantic_consistency: float           # 0.0-1.0
    switch_point_naturalness: float       # 1-5
    fluency_rating: float                 # 1-5
    annotator_agreement_kappa: float      # Cohen's κ for this dialogue


@dataclass
class ModelOutput:
    """Stores a model's generated response for a dialogue."""
    model_name: str
    dialogue_id: int
    prompt_mode: str                      # "zero_shot" | "few_shot"
    generated_text: str
    detected_switch_type: str = ""


@dataclass
class EvaluationResult:
    """Automatic evaluation metrics for a (model, dialogue) pair."""
    model_name: str
    dialogue_id: int
    topic: str
    code_switch_frequency: str
    language_dominance: str
    bleu: float = 0.0
    rouge_l: float = 0.0
    bert_f1: float = 0.0
    switch_accuracy_f1: float = 0.0
    coherence: float = 0.0               # from human eval
    fluency: float = 0.0                 # from human eval
    switch_naturalness: float = 0.0      # from human eval
    semantic_consistency: float = 0.0   # from human eval
    perplexity: float = 0.0


@dataclass
class AggregatedResult:
    """Mean metrics aggregated over all dialogues for a model."""
    model_name: str
    bleu: float = 0.0
    rouge_l: float = 0.0
    bert_f1: float = 0.0
    switch_accuracy_f1: float = 0.0
    coherence: float = 0.0
    fluency: float = 0.0
    switch_naturalness: float = 0.0
    semantic_consistency: float = 0.0
    perplexity: float = 0.0
    # Standard deviations
    bleu_std: float = 0.0
    rouge_l_std: float = 0.0
    bert_f1_std: float = 0.0
    coherence_std: float = 0.0
    fluency_std: float = 0.0


@dataclass
class HypothesisResult:
    """Result of a single hypothesis test."""
    hypothesis: str
    test_name: str
    statistic: float
    p_value: float
    effect_size: float                    # Cohen's d or Spearman ρ
    confidence_interval: tuple
    verdict: str                          # "Supported" | "Rejected" | "Inconclusive"
    interpretation: str
