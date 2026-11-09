"""
EthioSwitch-Bench Research Pipeline
====================================
Text utilities: language detection, switch-point analysis, prompt building.
"""
from __future__ import annotations
import re
from typing import List, Tuple, Dict
from utils.data_models import Dialogue, DialogueTurn


# ---------------------------------------------------------------------------
# Unicode Ranges
# ---------------------------------------------------------------------------
ETHIOPIC_RANGE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")
LATIN_WORD     = re.compile(r"[A-Za-z]{2,}")


# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------

def has_amharic(text: str) -> bool:
    """Return True if text contains Ethiopic script characters."""
    return bool(ETHIOPIC_RANGE.search(text))


def has_english(text: str) -> bool:
    """Return True if text contains Latin words of ≥2 chars."""
    return bool(LATIN_WORD.search(text))


def detect_switch_type(text: str) -> str:
    """
    Heuristic switch-type detection from raw text using script order.

    Returns one of: "[ENG]", "[AMH]", "[ENG-AMH]", "[AMH-ENG]",
    "[ENG-AMH-ENG]", "[AMH-ENG-AMH]", "[UNK]"
    """
    eng = has_english(text)
    amh = has_amharic(text)

    if not eng and not amh:
        return "[UNK]"
    if eng and not amh:
        return "[ENG]"
    if amh and not eng:
        return "[AMH]"

    # Both scripts present — determine order
    first_eng = LATIN_WORD.search(text)
    first_amh = ETHIOPIC_RANGE.search(text)
    if first_eng and first_amh:
        amh_first = first_amh.start() < first_eng.start()
        # Check if there is a second switch back
        if amh_first:
            after_eng = ETHIOPIC_RANGE.search(text, first_eng.start())
            return "[AMH-ENG-AMH]" if after_eng else "[AMH-ENG]"
        else:
            after_amh = LATIN_WORD.search(text, first_amh.start())
            return "[ENG-AMH-ENG]" if after_amh else "[ENG-AMH]"
    return "[UNK]"


def count_script_tokens(text: str) -> Tuple[int, int]:
    """
    Return (n_english_words, n_amharic_words) in the text.
    Ethiopic words are whitespace-delimited tokens containing Ethiopic chars.
    """
    tokens = text.split()
    eng_count = sum(1 for t in tokens if LATIN_WORD.search(t) and not ETHIOPIC_RANGE.search(t))
    amh_count = sum(1 for t in tokens if ETHIOPIC_RANGE.search(t))
    return eng_count, amh_count


def detect_language_dominance(turns: List[DialogueTurn]) -> str:
    """
    Classify a dialogue's language dominance based on word-level counts.

    Returns: "english_dominant" | "amharic_dominant" | "balanced"
    """
    total_eng, total_amh = 0, 0
    for t in turns:
        e, a = count_script_tokens(t.utterance)
        total_eng += e
        total_amh += a

    total = total_eng + total_amh
    if total == 0:
        return "balanced"
    ratio = total_eng / total
    if ratio >= 0.60:
        return "english_dominant"
    elif ratio <= 0.40:
        return "amharic_dominant"
    return "balanced"


def compute_switch_density(turns: List[DialogueTurn]) -> float:
    """
    Compute number of language switches per 100 tokens across all turns.
    A switch is counted each time the dominant script changes between consecutive tokens.
    """
    switches = 0
    total_tokens = 0

    for turn in turns:
        tokens = turn.utterance.split()
        total_tokens += len(tokens)
        prev_script = None
        for token in tokens:
            if ETHIOPIC_RANGE.search(token):
                cur = "amh"
            elif LATIN_WORD.search(token):
                cur = "eng"
            else:
                cur = prev_script
            if prev_script and cur and cur != prev_script:
                switches += 1
            if cur:
                prev_script = cur

    if total_tokens == 0:
        return 0.0
    return round(switches / total_tokens * 100, 2)


# ---------------------------------------------------------------------------
# Switch F1
# ---------------------------------------------------------------------------

def compute_switch_f1(predictions: List[str], references: List[str]) -> float:
    """
    Macro-F1 for switch-type tag prediction.

    Parameters
    ----------
    predictions : list of predicted switch tags per utterance
    references  : list of gold switch tags per utterance

    Returns
    -------
    float : macro-averaged F1 score
    """
    if not predictions or not references:
        return 0.0

    labels = list(set(references))
    f1_scores = []

    for label in labels:
        tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
        fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
        fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    return round(float(sum(f1_scores) / len(f1_scores)), 4) if f1_scores else 0.0


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

ZERO_SHOT_TEMPLATE = """\
You are a helpful bilingual assistant fluent in both English and Amharic.
Continue the following English-Amharic code-switched conversation naturally.
Maintain the same code-switching style (mixing English and Amharic) as the user.

Conversation:
{conversation_history}
Assistant:"""

FEW_SHOT_TEMPLATE = """\
You are a helpful bilingual assistant fluent in both English and Amharic.
Below are examples of natural English-Amharic conversations, followed by a
new conversation for you to continue.

Example 1:
User: ሰላም! How are you today?
Assistant: ደህና ነኝ፣ thank you! ስለ አንተ ምን ይሰማሃል?

Example 2:
User: What is your favorite Ethiopian food?
Assistant: Injera with doro wat is wonderful! ብዙ ጊዜ እመገዳለሁ እሱን.

Example 3:
User: Technology ዓለምን ለውጦታልን?
Assistant: አዎ! It has transformed every aspect of life. ትምህርት, health, communication — ሁሉም ተቀይሯል.

Now continue this conversation:
{conversation_history}
Assistant:"""


def build_prompt(dialogue: Dialogue, mode: str = "zero_shot",
                 context_turns: int = 3) -> str:
    """
    Build a model prompt from the last `context_turns` of a dialogue.

    Parameters
    ----------
    dialogue      : the Dialogue object
    mode          : "zero_shot" or "few_shot"
    context_turns : number of recent turns to include as context

    Returns
    -------
    str : formatted prompt string
    """
    recent  = dialogue.turns[-context_turns:]
    history = "\n".join(f"{t.speaker}: {t.utterance}" for t in recent)
    template = ZERO_SHOT_TEMPLATE if mode == "zero_shot" else FEW_SHOT_TEMPLATE
    return template.format(conversation_history=history)
