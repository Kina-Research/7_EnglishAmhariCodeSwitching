# Dataset Card for EthioSwitch-Bench

## Dataset Summary

**EthioSwitch-Bench** is the first large-scale benchmark corpus for evaluating Large Language Model (LLM) performance on **English–Amharic code-switching (CS)** in multi-turn conversational dialogue. The dataset consists of richly annotated, human-validated dialogues that naturally mix English and Amharic — reflecting authentic bilingual communication patterns among Ethiopian speakers.

The benchmark provides two tiers:
- **v2.0 (Gold)**: 30 expert hand-crafted dialogues across 7 topics — the highest-quality, fully human-validated subset.
- **v3.0 (Full Benchmark)**: 2,000 dialogues across 15 topics, combining the 30 gold dialogues with 970 synthetic and 1,000 corpus-adapted dialogues.

---

## Dataset Details

### Dataset Description

| Field | Value |
|-------|-------|
| **Languages** | English (`en`), Amharic (`am`) |
| **Task categories** | Conversational AI, Code-switching NLP, Multilingual benchmarking |
| **Modalities** | Text |
| **Format** | JSON |
| **Size (v2.0)** | 30 dialogues, 150 turns (~61 KB) |
| **Size (v3.0)** | 2,000 dialogues, 10,000 turns (~5.3 MB) |
| **License** | CC BY 4.0 |
| **DOI / Paper** | Under review |

### Supported Tasks

- **Dialogue coherence evaluation** — assessing whether LLM responses maintain coherence across language switches
- **Code-switch type detection** — classifying switch patterns (`[ENG-AMH]`, `[AMH-ENG-AMH]`, etc.)
- **Multilingual conversational response generation** — generating fluent, contextually appropriate code-switched replies
- **Switch-point naturalness scoring** — rating how natural language transitions are in bilingual dialogue

---

## Dataset Structure

### Data Instances

A single dialogue instance looks like:

```json
{
  "dialogue_id": 1,
  "topic": "daily_life",
  "code_switch_frequency": "high",
  "language_dominance": "balanced",
  "split": "train",
  "source": "gold",
  "source_dataset": "original",
  "generation_method": "hand_crafted",
  "turns": [
    {
      "turn_id": 1,
      "speaker": "User",
      "utterance": "ሰላም! How was your day today?",
      "language_tags": ["[AMH]", "[ENG]"],
      "switch_type": "[AMH-ENG]",
      "switch_point_token": "ሰላም!"
    },
    {
      "turn_id": 2,
      "speaker": "Assistant",
      "utterance": "ቀኑ ጥሩ ነበር፣ thank you for asking! I spent most of the morning በሥራ ተጠምጄ ነበር, so it was quite productive.",
      "language_tags": ["[AMH]", "[ENG]", "[AMH]"],
      "switch_type": "[AMH-ENG-AMH]",
      "switch_point_token": "thank you"
    }
  ],
  "annotations": {
    "coherence_score": 4.6,
    "semantic_consistency": 0.87,
    "switch_point_naturalness": 4.4,
    "fluency_rating": 4.3,
    "annotator_agreement_kappa": 0.78
  }
}
```

### Data Fields

#### Dialogue-level fields

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `dialogue_id` | int | 1–2000 | Unique identifier |
| `topic` | string | See topics | Conversation domain |
| `code_switch_frequency` | string | `low`, `medium`, `high` | Rate of language switches per dialogue |
| `language_dominance` | string | `english_dominant`, `amharic_dominant`, `balanced` | Which language has more content |
| `split` | string | `train`, `validation`, `test` | Data partition |
| `source` | string | `gold`, `synthetic`, `adapted` | Origin of the dialogue |
| `source_dataset` | string | `original`, `dailydialog`, etc. | Source corpus name |
| `generation_method` | string | `hand_crafted`, `template_cs`, `public_adaptation` | How dialogue was produced |

#### Turn-level fields

| Field | Type | Description |
|-------|------|-------------|
| `turn_id` | int | Turn index within dialogue |
| `speaker` | string | `User` or `Assistant` |
| `utterance` | string | Raw text (may contain both Amharic Ethiopic script and Latin script) |
| `language_tags` | list[str] | Language label per segment in the utterance |
| `switch_type` | string | Overall switch pattern for this turn |
| `switch_point_token` | string | First token marking a language switch |

#### Annotation fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `coherence_score` | float | 1–5 | Dialogue coherence (human-rated) |
| `semantic_consistency` | float | 0–1 | Cosine similarity of meaning across switch point |
| `switch_point_naturalness` | float | 1–5 | Naturalness of switch point (human-rated) |
| `fluency_rating` | float | 1–5 | Overall fluency of the dialogue (human-rated) |
| `annotator_agreement_kappa` | float | 0–1 | Cohen's κ inter-annotator agreement |

### Data Splits

#### v2.0

| Split | Dialogues |
|-------|-----------|
| Train | 21 |
| Validation | 4 |
| Test | 5 |
| **Total** | **30** |

#### v3.0

| Split | Dialogues |
|-------|-----------|
| Train | 1,400 |
| Validation | 298 |
| Test | 302 |
| **Total** | **2,000** |

---

## Dataset Creation

### Curation Rationale

Code-switching between English and Amharic is a widespread, natural phenomenon among urban Ethiopian speakers. Despite this, no prior benchmark existed to evaluate LLMs on this language pair in conversational settings. EthioSwitch-Bench was created to fill this gap and enable:
- Fair comparison of multilingual vs. monolingual LLMs on code-switched input
- Analysis of how CS frequency and language dominance affect model performance
- Development of CS-aware dialogue systems for Ethiopian language communities

### Source Data

#### v2.0 — Gold dialogues
All 30 dialogues were **hand-crafted by expert annotators** fluent in both English and Amharic, covering 7 daily-life domains. Each dialogue was independently reviewed by at least 2 annotators, and dialogues below Cohen's κ = 0.60 were excluded.

#### v3.0 — Synthetic dialogues (970)
Generated from topic-specific English templates using a controlled code-switching insertion algorithm (see `code/00_dataset_expander.py`). Language switches were inserted at semantically valid positions. Generation used RNG seed=42 for reproducibility.

#### v3.0 — Adapted dialogues (1,000)
English dialogues drawn from four public corpora were code-switched by the same algorithm:

| Source Corpus | Dialogues | License | Reference |
|---------------|-----------|---------|-----------|
| DailyDialog | 300 | CC BY-NC-SA 4.0 | Li et al., IJCNLP 2017 |
| PersonaChat | 250 | MIT | Zhang et al., ACL 2018 |
| EmpatheticDialogues | 200 | CC BY 4.0 | Rashkin et al., ACL 2019 |
| BlendedSkillTalk | 250 | MIT | Smith et al., ACL 2020 |

### Annotation Process

Human annotations were collected from bilingual (English–Amharic) annotators with the following rating scales:

- **Coherence** (1–5): Does the assistant's response make sense in context?
- **Fluency** (1–5): Does the dialogue read naturally despite language switching?
- **Switch-Point Naturalness** (1–5): Is the point where the language switches linguistically natural?
- **Semantic Consistency** (0–1): Is the meaning preserved across language switch boundaries?

Inter-annotator agreement (Cohen's κ) was computed per dialogue. The overall IAA κ for v2.0 is **0.74**, and for v3.0 is **0.724** — indicating substantial agreement.

### Personal and Sensitive Information

The dataset contains **no personal information**. All dialogues are fictional conversations about general topics (technology, health, education, culture, etc.) and contain no references to real individuals, locations, or identifiable data.

---

## Uses

### Direct Use

- Evaluating LLM responses to code-switched prompts
- Benchmarking multilingual dialogue systems
- Training code-switching language models
- Studying switch-point detection and naturalness in NLP

### Out-of-Scope Use

- Identifying individuals (no PII present)
- Real-time conversational AI without human oversight
- Applications in high-stakes domains (medical, legal) without validation by domain experts

---

## Dataset Statistics

### v3.0 Topic Distribution

| Topic | Count |
|-------|-------|
| daily_life | 178 |
| technology | 150 |
| science | 132 |
| health | 163 |
| education | 168 |
| culture | 144 |
| sports | 126 |
| politics | 102 |
| family_relationships | 114 |
| mental_health | 140 |
| entertainment | 121 |
| travel | 103 |
| environment | 118 |
| food_nutrition | 116 |
| finance | 125 |

### Annotation Means (v3.0)

| Metric | Mean |
|--------|------|
| Coherence score | 4.159 |
| Semantic consistency | 0.836 |
| Fluency rating | 3.951 |
| Switch-point naturalness | 4.013 |
| IAA κ | 0.724 |

---

## Limitations

1. **Language pair scope**: The benchmark covers English–Amharic only. Findings may not generalize to other CS pairs.
2. **Synthetic data quality**: The 970 synthetic and 1,000 adapted dialogues were algorithmically code-switched and may not fully capture all naturalistic CS phenomena.
3. **Topic coverage**: While 15 topics are covered, specialized domains (law, medicine, engineering) are underrepresented.
4. **Amharic dialect variation**: The dataset uses Standard Amharic (Addis Ababa variety) and does not cover regional dialect variation.
5. **Annotation subjectivity**: Naturalness and fluency ratings are inherently subjective; κ = 0.72–0.74 reflects substantial but imperfect agreement.

---

## Citation

```bibtex
@article{abebe2026ethioswitch,
  author    = {Abebe, Tewodros and Girma, Soliana and Kim, Jinhyuk and Fung, Pascale},
  title     = {Beyond Monolingual Bounds: Benchmarking Conversational Intelligence
               in {LLMs} via {English--Amharic} Code-Switching},
  journal   = {Computational Linguistics},
  year      = {2026},
  note      = {Under review}
}
```

---

*Dataset card format inspired by [Hugging Face Dataset Cards](https://huggingface.co/docs/hub/datasets-cards).*
