# LaTeX Manuscript: Evaluating Conversational Intelligence in Large Language Models Using English–Amharic Code-Switching

This directory contains the exact, complete, and publication-ready LaTeX sources for the research manuscript.

## Directory Structure
- `main.tex`: Master LaTeX document containing the complete manuscript, all 7 sections, all mathematical formulations (Equations 1–9), all 11 tables, all 8 figure inclusions, and complete declarations.
- `references.bib`: BibTeX bibliography database containing all 41 peer-reviewed references cited in the manuscript.
- `figures/`: Vector PDF and high-resolution (300 DPI) PNG publication figures:
  - `fig0_experimental_workflow.png`: End-to-end architectural workflow.
  - `fig1_model_comparison_bar.pdf` / `.png`: Automatic evaluation metrics across 5 models.
  - `fig2_cs_freq_vs_consistency.pdf` / `.png`: Semantic consistency degradation across CS frequency tiers.
  - `fig3_multi_metric_radar.pdf` / `.png`: Multi-dimensional capability radar profile.
  - `fig4_coherence_distribution_box.pdf` / `.png`: Human contextual coherence distributions.
  - `fig5_fluency_language_dominance_bar.pdf` / `.png`: Conversational fluency across language dominance.
  - `fig6_metric_correlation_heatmap.pdf` / `.png`: Inter-metric Pearson correlation heatmap.
  - `fig7_ablation_sensitivity.pdf` / `.png`: Ablation and sensitivity analysis across few-shot scaling, temperatures, and tokenizer fertility.
- `generate_tex.py`: Python generator script maintaining exact synchronization with the Word DOCX manuscript.

## Compilation Instructions

The document is optimized for compilation with **XeLaTeX** to natively render Ethiopic (Ge'ez) script alongside Latin typography using `fontspec` and `ucharclasses` (with the native Windows `Nyala` font).

### Recommended Build Command (XeLaTeX)
```bash
# First pass
xelatex -interaction=nonstopmode main.tex

# Compile bibliography
bibtex main

# Second pass (resolve citations)
xelatex -interaction=nonstopmode main.tex

# Third pass (resolve cross-references and table of contents)
xelatex -interaction=nonstopmode main.tex
```

### Clean Temporary Files
```bash
# Windows PowerShell
Remove-Item -Path main.aux, main.bbl, main.blg, main.log, main.out, main.toc -ErrorAction SilentlyContinue
```
