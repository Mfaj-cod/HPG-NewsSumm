# NewsSumm Research Benchmark  
**Reproducible Multi-Document Summarization Research Framework**

---

## Overview

This repository provides a **clean, reproducible, research-grade codebase** for benchmarking and experimenting with **multi-document summarization models** on the **NewsSumm** dataset (Indian English news).

The project is designed to:

- Implement a **full data pipeline** (preprocessing + statistics)
- Provide a **reproducible experiment framework** with config-based runs
- Support at least **one strong baseline model** (LED)
- Propose and scaffold a **novel hierarchical summarization model**
- Ensure **every experiment is tracked and repeatable**

---

## Repository Structure
```bash
data/ # Raw and processed datasets
models/ # Baseline and novel model implementations
scripts/ # Training, evaluation, preprocessing scripts
configs/ # YAML configs for each experiment
results/ # Experiment outputs, logs, checkpoints
docs/ # Documentation and design specs
```

---

## 1. Environment Setup

### Create Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
##  2. Dataset Setup
Place the dataset file here: 
```bash
data/NewsSumm_Dataset.xlsx
```
## 3. Preprocessing Pipeline
Run: 
```bash
python scripts/preprocess.py --input data/NewsSumm_Dataset.xlsx --output data/newssumm_processed
```
### This will:
- Clean HTML and noise
- Normalize text
- Generate data/newssumm_processed/newssumm_processed.json

Convert dataset into a unified JSON format:
```bash
{
  "cluster_id": "...",
  "documents": ["doc1", "doc2", "..."],
  "summary": "...",
  "metadata": {...}
}
```
Output:
```bash
data/newssumm_processed/newssumm_processed.json
```

On Running:
```bash
python scripts/clean_dataset.py
```
### This will
1. Remove rows with missing article text or summary.
2. Remove HTML tags and markup.
3. Normalize whitespace and formatting.
4. Remove very short or corrupted entries.
5. Remove duplicate articles based on article_text.
6. Standardize column names.
7. Ultimately generate a cleaned dataset - data/NewsSumm_Cleaned.xlsx

Output:
```bash
data/NewsSumm_Cleaned.xlsx
```
---
## 4. Dataset Statistics
Run: 
```bash
python scripts/compute_stats.py --data data/newssumm_processed/newssumm_processed.json
```
This reports:
- Number of clusters
- Average document length
- Average summary length
- Documents per cluster

## 5. Experiment System (Reproducibility)
All experiments are driven by YAML config files in: configs/

- Each run automatically creates a folder in: results/<experiment_name>/

Containing:
- config.yaml — full config snapshot
- meta.json — device + time info
- summary.json — metrics + runtime
- checkpoint/ — saved model (if applicable)

---

## 6. Baseline Model (LED)
LED (Baseline):
```bash
python scripts/train_baseline.py \
  --config configs/led_baseline.yaml \
  --sample 1
```
- Increase the sample size if the RAM allows.
This will:

- Load LED
- Load dataset
- Run one safe training step (CPU-friendly)
- Compute a real loss
- Save a checkpoint
- Log the experiment

Note: Full training is not performed locally due to hardware constraints.
The pipeline is fully implemented and validated via a dry-run.
---

### Run Other Models:
LongT5:
```bash
python scripts/train_baseline.py \
  --config configs/longt5.yaml \
  --sample 1
```
FLAN-T5 (XL):
```bash
python scripts/train_baseline.py --config configs/flan_t5_xl.yaml --sample 1
```
PRIMERA:
```bash
python scripts/train_baseline.py --config configs/primera.yaml --sample 1
```
---
## 7. Evaluation
```bash
python scripts/evaluate.py --run results/<run_name>
```
- This will produce: results/<run_name>/evaluation.json

Computes:
- ROUGE-1, ROUGE-2, ROUGE-L
- BERTScore

## 8. Novel Model: Hierarchical Planner–Generator (HPG)

We propose a Hierarchical Planner–Generator architecture that separates:

Content Planning → Summary Generation

The full design is described in: docs/novel_model_spec.md

A prototype implementation exists in: models/novel_model.py

Training entry point: 
```bash
python scripts/train_novel.py --config configs/novel_model.yaml --sample 1
```
- NOTE: This is a research prototype architecture and currently runs a single-step sanity-check training pass to validate the design and pipeline integration.
---
## 9. Reproducing Any Experiment

To reproduce any run:
```bash
python scripts/train_baseline.py --config results/<run_name>/config.yaml
```
This guarantees:

- Same hyperparameters
- Same seed
- Same setup
- Similar results
---
## 10. Documentation

- docs/data_cleaning_report - NewsSumm Dataset Cleaning Report
- docs/data_pipeline.md — Dataset format & preprocessing
- docs/experiment_tracking.md — Reproducibility system
- docs/novel_model_spec.md — Novel model design

---

## 12. Summary

This repository provides:

- A complete research pipeline
- A reproducible experiment system
- A working baseline and other models integration
- A novel model proposal + prototype
- A clean foundation for real research work

# Final simple plan (practical + sane)
### Train (GPU heavy)

PRIMERA

LED

LongT5

Novel

### Prompt only (GPU light)

Flan-T5-XL/XXL

Mistral-7B

LLaMA-3-8B

Qwen-7B

Gemma-9B

Mixtral (if memory allows inference)