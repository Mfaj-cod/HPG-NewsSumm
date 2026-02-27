# NewsSumm – Hierarchical Planner–Generator (HPG)

**Reproducible Multi-Document Summarization Research Framework**  
Indian English News | Long-Context Modeling | Hierarchical Planning

---

## Overview

This repository provides a **research-grade, fully reproducible framework** for benchmarking multi-document summarization systems on the **NewsSumm** dataset (Indian English news corpus).

The project includes:

- End-to-end **data cleaning and preprocessing**
- Config-driven **experiment tracking**
- Multiple **long-context baselines**
- A novel **Hierarchical Planner–Generator (HPG)** architecture
- Fully reproducible **training and evaluation pipelines**

The framework is designed for structured experimentation, comparative benchmarking, and controlled ablation studies.

---

## Repository Structure

```bash
data/
├── NewsSumm_Dataset.xlsx
├── processed/
└── NewsSumm_Cleaned.xlsx

models/
├── baseline_led.py
├── baseline_generic.py
└── novel_model.py

scripts/
├── clean_dataset.py
├── preprocess.py
├── compute_stats.py
├── prompted_eval_all.py
├── train_baseline.py
├── train_novel.py
├── utils.py
└── evaluate.py

configs/
├── led_baseline.yaml
├── longt5.yaml
├── primera.yaml
├── flan_t5_xl.yaml
└── novel_model.yaml

results/
requirements.txt
```


---

# 1. System Requirements

## Hardware

Minimum:
- 1× GPU (16GB VRAM recommended)
- 32GB RAM
- 50GB disk space

For large models (PRIMERA, LongT5, Mixtral):
- 24–48GB VRAM recommended

---

# 2. Environment Setup

## Create Virtual Environment

```bash
python -m venv venv
```
## Activate
```bash
Linux / Mac: source venv/bin/activate
Windows: venv\Scripts\activate
```
## Install Dependencies
```bash
pip install -r requirements.txt
```

## Set Environment Variable
```bash
HF_TOKEN=your_huggingface_token
```

# 3. Dataset Setup
```bash
Place the dataset file at: data/NewsSumm_Dataset.xlsx
```
# 4. Data Cleaning Pipeline
```bash
data/NewsSumm_Dataset.xlsx
```
This step:

- Removes missing article or summary rows
- Cleans HTML tags and markup
- Normalizes whitespace
- Removes duplicates
- Filters corrupted entries
- Standardizes column names

Output:
```bash
data/NewsSumm_Cleaned.xlsx
```

# 5. Preprocessing Pipeline
Convert cleaned Excel data into structured JSON clusters:
```bash
python scripts/preprocess.py \
  --input data/NewsSumm_Cleaned.xlsx \
  --output data/processed
```
Generated format:
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
data/processed/newssumm_processed.json
```
# 6. Dataset Statistics

Compute dataset diagnostics:
```bash
python scripts/compute_stats.py \
--data data/processed/newssumm_processed.json
```

Reports:

- Number of clusters
- Avg tokens per cluster
- Max tokens per cluster
- Avg summary length
- Avg documents per cluster
- Compression ratio

# 7. Experiment Framework (Reproducibility)

All experiments are config-driven via YAML files in: ```bash /configs```
Each run automatically creates:
```bash
results/<experiment_name>/
  ├── config.yaml
  ├── meta.json
  ├── summary.json
  ├── evaluation.json
  └── checkpoint/
```

Every experiment snapshot includes:
- Hyperparameters
- Random seed
- Device info
- Metrics
- Runtime metadata

# 8. Baseline Models

a. LED (Longformer Encoder-Decoder)
```bash
python scripts/train_baseline.py \
  --config configs/led_baseline.yaml
```

b. LongT5
```bash
python scripts/train_baseline.py \
  --config configs/longt5.yaml
```

c. PRIMERA
```bash
python scripts/train_baseline.py \
  --config configs/primera.yaml
```

d. FLAN-T5-XL
```bash
python scripts/train_baseline.py \
  --config configs/flan_t5_xl.yaml
```

# 9. Novel Model – Hierarchical Planner–Generator (HPG)
HPG separates summarization into two explicit stages:

I. Document-Level Planning

- Learns salience across documents
- Redundancy-aware scoring
- Cross-document representation

II. Conditional Generation

- Guided decoding
- Structured content ordering
- Coverage-aware loss

## Train HPG
```bash
python scripts/train_novel.py \
  --config configs/novel_model.yaml
```

# 10. Evaluation
Evaluate any trained run:
```bash
python scripts/evaluate.py \
  --run results/<run_name>
```
Metrics computed:
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore

Results stored in:
```bash
results/<run_name>/evaluation.json
```
# 11. Reproducing a Past Experiment

To reproduce any completed run:
```bash
python scripts/train_baseline.py \
  --config results/<run_name>/config.yaml
``` 
This ensures:

- Same hyperparameters
- Same seed
- Same configuration
- Deterministic pipeline behavior

# 12. Running on a New System
Step 1 – Clone Repository
```bash
git clone <repo_url>
cd <repo_name>
```
Step 2 – Setup Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Step 3 – Place Dataset
```bash
data/NewsSumm_Dataset.xlsx
```
Step 4 – Run Full Pipeline
```bash
python scripts/clean_dataset.py
python scripts/preprocess.py --input data/NewsSumm_Cleaned.xlsx --output data/processed
python scripts/compute_stats.py --data data/processed/newssumm_processed.json
```
Step 5 – Train Model
```bash
python scripts/train_baseline.py --config configs/led_baseline.yaml --sample 25000
```
or Train HPG
```bash
python scripts/train_novel.py --config configs/novel_model.yaml --sample 25000
```
Step 6 – Evaluate
```bash
python scripts/evaluate.py --run results/<run_name> --sample 10000
```
### For Prompt Based Evaluation
```bash
python prompted_eval.py --model google/flan-t5-xl  --data data/newssumm_processed/newsumm_processed.json --sample 10000 --out_dir results/flan_prompt
```

# 13. Experiment Strategy
Heavy GPU Training

- PRIMERA
- LED
- LongT5
- HPG (Novel)

Prompt-Based / Light Inference

- Flan-T5-XL / XXL
- Mistral-7B-Instruct
- LLaMA-3-8B-Instruct
- Qwen2-7B-Instruct
- Gemma-2-9B-Instruct (if memory allows)
- Mixtral-8×7B-Instruct (if memory allows)

# 14. Research Goals

This repository supports:

- Long-context multi-document summarization
- Hierarchical planning architectures
- Redundancy-aware modeling
- Controlled baseline benchmarking
- Fully reproducible experiments