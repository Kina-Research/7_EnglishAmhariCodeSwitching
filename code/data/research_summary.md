# Benchmarking Conversational Intelligence in LLMs via English-Amharic Code-Switching

## Executive Summary

This report presents empirical findings from evaluating five Large Language Models on the **EthioSwitch-Bench v3.0** corpus comprising 2,000 multi-turn dialogues (10,000 conversational turns). The evaluation spans multi-turn continuation (Task 1) and switch-point boundary detection (Task 2), addressing four fundamental research questions (RQ1–RQ4).

## Table 1: Automatic Evaluation Metrics on Full Corpus

| Model | Architecture | BLEU ↑ | ROUGE-L ↑ | BERTScore F1 ↑ | Switch F1 ↑ | Perplexity (PPL) ↓ |
|---|---|---|---|---|---|---|
| **GPT-4** | Decoder-only (Proprietary) | 0.6086 ± 0.2009 | 0.8634 ± 0.1240 | 0.7996 ± 0.1370 | 0.7745 ± 0.4179 | 22.53 |
| **LLaMA-2-7B** | Decoder-only (Open-weight, En-centric) | 0.2764 ± 0.1618 | 0.7697 ± 0.1328 | 0.5556 ± 0.1361 | 0.0000 ± 0.0000 | 52.95 |
| **BLOOM-7B** | Decoder-only (Open-weight, Multilingual) | 0.2015 ± 0.1255 | 0.6848 ± 0.1260 | 0.4459 ± 0.1159 | 0.0000 ± 0.0000 | 63.76 |
| **mT5-Large** | Encoder-Decoder (Multilingual) | 0.4480 ± 0.1947 | 0.8030 ± 0.1318 | 0.6823 ± 0.1413 | 0.7545 ± 0.4304 | 43.59 |
| **mBERT-fine-tuned** | Encoder + Cross-Attn Decoder | 0.5341 ± 0.1963 | 0.8366 ± 0.1270 | 0.7440 ± 0.1380 | 0.7730 ± 0.4189 | 35.89 |

## Table 2: Human Qualitative Dialogue Quality Ratings (Scale 1–5)

| Model | Coherence ↑ | Fluency ↑ | Switch-Point Naturalness ↑ | Overall Assessment |
|---|---|---|---|---|
| **GPT-4** | 4.78 ± 0.34 | 4.36 ± 0.48 | 4.14 ± 0.42 | Superior contextual flow and natural switch integration |
| **LLaMA-2-7B** | 2.79 ± 0.52 | 2.12 ± 0.46 | 2.01 ± 0.34 | Severe Amharic degradation; frequent script collapse into hallucinations |
| **BLOOM-7B** | 3.00 ± 0.35 | 3.00 ± 0.35 | 2.70 ± 0.56 | Low coherence; fails to maintain context across language boundaries |
| **mT5-Large** | 3.82 ± 0.49 | 3.32 ± 0.57 | 3.10 ± 0.44 | Moderate semantic alignment; occasional repetition across switches |
| **mBERT-fine-tuned** | 4.00 ± 0.35 | 4.01 ± 0.35 | 3.99 ± 0.37 | Strong syntactic preservation; effective switch boundary tracking |

## Table 3: Formal Hypothesis Testing and Statistical Significance

| Hypothesis / RQ | Statistical Test | Target Metric | Test Stat | df | p-value | Effect Size | 95% CI | Verdict |
|---|---|---|---|---|---|---|---|---|
| **H1 (GPT-4 vs. LLaMA-2-7B)** | Welch's t-test | Dialogue Coherence (1-5) | 142.9391 | 3424.3 | <0.001 | 4.5201 | [1.9647, 2.0193] | **Supported** |
| **H1 (GPT-4 vs. BLOOM-7B)** | Welch's t-test | Dialogue Coherence (1-5) | 162.8392 | 3990.6 | <0.001 | 5.1494 | [1.7615, 1.8045] | **Supported** |
| **H2 (CS Frequency vs. Semantic Consistency)** | Spearman Rank Correlation | Semantic Consistency vs. CS Frequency Tier | -0.8319 | 1998 | <0.001 | -0.8319 | [-0.8449, -0.8179] | **Supported** |
| **H3 (Multilingual vs. Zero-Shot Switch F1)** | Welch's t-test | Switch-Point Tagging F1 | 140.6580 | 5999.0 | <0.001 | 2.3443 | [0.7566, 0.778] | **Supported** |
| **H4 (English vs. Amharic-Dominant Fluency)** | Paired t-test | Conversational Fluency (1-5) | 0.4437 | 3344 | 0.6573 | 0.0077 | [-0.0169, 0.0267] | **Rejected** |

## Table 4: Switch-Point Tagging Accuracy (Switch F1) by Transition Direction

| Model | ENG → AMH F1 (n = 490) | AMH → ENG F1 (n = 454) | Directional Gap (Δ) |
|---|---|---|---|
| **GPT-4** | 0.7531 ± 0.4312 | 0.5771 ± 0.4940 | +0.1760 |
| **LLaMA-2-7B** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.0000 |
| **BLOOM-7B** | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.0000 |
| **mT5-Large** | 0.7551 ± 0.4300 | 0.5419 ± 0.4982 | +0.2132 |
| **mBERT-fine-tuned** | 0.7551 ± 0.4300 | 0.5617 ± 0.4962 | +0.1934 |

## Table 5: Semantic Consistency & Coherence by Code-Switch Frequency Tier

| Model | Low CS BERTScore | Low CS Coherence | Med CS BERTScore | Med CS Coherence | High CS BERTScore | High CS Coherence |
|---|---|---|---|---|---|---|
| **GPT-4** | 0.8119 | 0.00 | 0.7992 | 0.00 | 0.7877 | 0.00 |
| **LLaMA-2-7B** | 0.5795 | 0.00 | 0.5679 | 0.00 | 0.5198 | 0.00 |
| **BLOOM-7B** | 0.4614 | 0.00 | 0.4528 | 0.00 | 0.4237 | 0.00 |
| **mT5-Large** | 0.6911 | 0.00 | 0.6789 | 0.00 | 0.6770 | 0.00 |
| **mBERT-fine-tuned** | 0.7516 | 0.00 | 0.7450 | 0.00 | 0.7355 | 0.00 |

## LaTeX Publication Snippets

```latex
\begin{table*}[t]
\centering
\small
\begin{tabular}{lcccccc}
\toprule
Model & Architecture & BLEU $\uparrow$ & ROUGE-L $\uparrow$ & BERTScore F1 $\uparrow$ & Switch F1 $\uparrow$ & PPL $\downarrow$ \\
\midrule
GPT-4 & Decoder-only (Proprietary) & 0.6086 \scriptsize{$\pm$ 0.2009} & 0.8634 \scriptsize{$\pm$ 0.1240} & 0.7996 \scriptsize{$\pm$ 0.1370} & 0.7745 \scriptsize{$\pm$ 0.4179} & 22.53 \\
LLaMA-2-7B & Decoder-only (Open-weight, En-centric) & 0.2764 \scriptsize{$\pm$ 0.1618} & 0.7697 \scriptsize{$\pm$ 0.1328} & 0.5556 \scriptsize{$\pm$ 0.1361} & 0.0000 \scriptsize{$\pm$ 0.0000} & 52.95 \\
BLOOM-7B & Decoder-only (Open-weight, Multilingual) & 0.2015 \scriptsize{$\pm$ 0.1255} & 0.6848 \scriptsize{$\pm$ 0.1260} & 0.4459 \scriptsize{$\pm$ 0.1159} & 0.0000 \scriptsize{$\pm$ 0.0000} & 63.76 \\
mT5-Large & Encoder-Decoder (Multilingual) & 0.4480 \scriptsize{$\pm$ 0.1947} & 0.8030 \scriptsize{$\pm$ 0.1318} & 0.6823 \scriptsize{$\pm$ 0.1413} & 0.7545 \scriptsize{$\pm$ 0.4304} & 43.59 \\
mBERT-fine-tuned & Encoder + Cross-Attn Decoder & 0.5341 \scriptsize{$\pm$ 0.1963} & 0.8366 \scriptsize{$\pm$ 0.1270} & 0.7440 \scriptsize{$\pm$ 0.1380} & 0.7730 \scriptsize{$\pm$ 0.4189} & 35.89 \\
\bottomrule
\end{tabular}
\caption{Overall automatic evaluation performance across five benchmark LLMs on EthioSwitch-Bench v3.0 (2,000 dialogues).}
\label{tab:overall_performance}
\end{table*}
```