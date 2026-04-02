# Beyond Flat Attention: Hierarchical Content Planning for Multi-Document Abstractive News Summarization

**Reproducible Multi-Document Summarization Research Framework**  
Indian English News | Long-Context Modeling | Hierarchical Planning

---

## Overview

This repository provides a **research-grade, fully reproducible framework** for benchmarking multi-document summarization systems on the **NewsSumm** dataset (Indian English news corpus).

The project includes:

- End-to-end **data cleaning and preprocessing**
- Config-driven **experiment tracking**
- Multiple **long-context baselines**
- A novel **Hierarchical Plannerâ€“Generator (HPG)** architecture
- Fully reproducible **training and evaluation pipelines**

The framework is designed for structured experimentation, comparative benchmarking, and controlled ablation studies.

---

## Repository Structure

```bash
data/
  NewsSumm_Dataset.xlsx
  NewsSumm_Cleaned.xlsx
  enhanced/
    newssumm_enhanced.json

models/
  baseline_generic.py
  baseline_led.py
  HPG.py

scripts/
  __init__.py
  data_preparation/
    __init__.py
    clean_dataset.py
    compute_stats.py
    prepare_and_compute.py
  data_validator/
    __init__.py
    validate_json_schema.py
  diagram_generators/
    __init__.py
    data_preparation-diagram.py
    HPG-diagram.py
  evaluation/
    __init__.py
    prompted_eval.py
    run_evaluation_on_excel.py
    run_evaluation.py
  training/
    __init__.py
    train_baseline.py
    train_HPG.py

configs/
  flan_t5_xl.yaml
  led_baseline.yaml
  longt5.yaml
  novel_model.yaml
  primera.yaml

plots/
reports/
results/
requirements.txt
```


---

# 1. System Requirements

## Hardware

Minimum:
- 1Ã— GPU (16GB VRAM recommended)
- 32GB RAM
- 50GB disk space

For large models (PRIMERA, LongT5, Mixtral):
- 24â€“48GB VRAM recommended

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

## Additional NLP Assets (for enhanced dataset script)
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
```

# 3. Dataset Setup
```bash
Place the dataset file at: data/NewsSumm_Dataset.xlsx
```
# 4. Data Cleaning Pipeline
Input:
```bash
data/NewsSumm_Dataset.xlsx
```
run:
```bash
python scripts/data_preparation/clean_dataset.py
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
- Convert raw Excel data into structured JSON clusters.
- Enhanced preprocessing (full pipeline with cleaning, dedup, features, topics, clustering, and validation).
```bash
python scripts/data_preparation/prepare_and_compute.py \
  --input data/NewsSumm_Dataset.xlsx \
  --output_dir data/enhanced \
  --reports_dir reports \
  --docs_per_cluster 30 \
  --cluster_tfidf_max_features 10000 \
  --cluster_svd_components 64 \
  --preflight_only
```
This step:

- Computes baseline stats on the raw dataset
- Cleans and normalizes text
- Filters non-English content
- Removes duplicates (exact + fuzzy)
- Adds linguistic and NER features
- Generates topics and clusters
- Validates summary quality
- Writes enhanced dataset + comparison reports

Output:
```bash
data/enhanced/newssumm_enhanced.json
reports/baseline_stats.json
reports/enhanced_stats.json
reports/cleaning_log.json
reports/comparison_table.csv
```
# 6. Dataset Statistics

Compute dataset diagnostics:
```bash
python scripts/data_preparation/compute_stats.py \
--data data/enhanced/newssumm_enhanced.json
```

Reports:

- Number of clusters
- Avg tokens per cluster
- Max tokens per cluster
- Avg summary length
- Avg documents per cluster
- Compression ratio

# 6.1 Dataset Schema Validation

Validate JSON structure before training or evaluation:
```bash
python scripts/data_validator/validate_json_schema.py \
  --data data/enhanced/newssumm_enhanced.json
```

Strict mode with sample limit (Uses first n samples and enforces non-empty docs/summary and treats warnings as errors):
```bash
python scripts/data_validator/validate_json_schema.py \
  --data data/enhanced/newssumm_enhanced.json --strict --sample 10
```

# 6.2 Pipeline Diagrams

Generate visual diagrams for data preparation pipeline and HPG model architecture.

**Requirements:**
Graphviz must be installed on your system:
- **Ubuntu/Debian:** `sudo apt-get install graphviz`
- **macOS:** `brew install graphviz`
- **Windows:** Download from https://graphviz.org/download/ or `choco install graphviz`

## Data Preparation Pipeline Diagram

Generates a visual flowchart of the data preparation pipeline stages (loading, cleaning, filtering, deduplication, clustering, validation):

```bash
python scripts/diagram_generators/data_preparation-diagram.py
```

Output files:
```bash
dataprep_pipeline.ps    # PostScript format
dataprep_pipeline.jpg   # JPEG format
```

## HPG Model Architecture Diagram

Generates a visual representation of the Hierarchical Planner-Generator (HPG) model architecture, showing the encoder, planner, fusion, decoder, and loss function components:

```bash
python scripts/diagram_generators/HPG-diagram.py
```

Output files:
```bash
hpg_architecture.ps     # PostScript format
hpg_architecture.jpg    # JPEG format
```

Both diagrams are rendered at 300 DPI for high-quality publication-ready output.

# 7. Experiment Framework (Reproducibility)

All experiments are config-driven via YAML files in: ```bash /configs```
Each run automatically creates:
```bash
results/<experiment_name>/
  â”œâ”€â”€ config.yaml
  â”œâ”€â”€ meta.json
  â”œâ”€â”€ summary.json
  â”œâ”€â”€ evaluation.json
  â””â”€â”€ checkpoint/
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
python scripts/training/train_baseline.py \
  --config configs/led_baseline.yaml
```

b. LongT5
```bash
python scripts/training/train_baseline.py \
  --config configs/longt5.yaml
```

c. PRIMERA
```bash
python scripts/training/train_baseline.py \
  --config configs/primera.yaml
```

d. FLAN-T5-XL
```bash
python scripts/training/train_baseline.py \
  --config configs/flan_t5_xl.yaml
```

# 9. Novel Model "Hierarchical Planner Generator (HPG)"
HPG separates summarization into these stages:

#### I. SegmentPooler

- Builds fixed hierarchical segments from long token sequences (pseudo document-level structure).   

#### II. SalienceAwarePlanner
- Scores segment salience.
- Uses learned plan queries to extract multiple plan tokens from salient segments.
- Refines plan tokens with transformer layers.


####  III. PlanConditionedFusion 
- Lets encoder token states attend to plan tokens and fuse them through a learned gate.
- Produces plan-aware encoder states before decoding.


#### IV. Auxiliary planning objectives            
                                                                                                            
- planner_entropy term (focuses salience distribution).
- plan_redundancy penalty (reduces repetitive plan tokens).
- Added to generation loss with configurable weights.


## Train HPG
```bash
python scripts/training/train_HPG.py --data data/newssumm_processed/newssumm_processed.json
```

# 10. Evaluation
Evaluate any trained run:
```bash
python scripts/evaluation/run_evaluation_json.py \
  --run_dir results/<run_name> \
  --data data/newssumm_processed/newssumm_processed.json
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
python scripts/training/train_baseline.py \
  --config results/<run_name>/config.yaml
``` 
This ensures:

- Same hyperparameters
- Same seed
- Same configuration
- Deterministic pipeline behavior

# 12. Running on a New System
Step 1 â€“ Clone Repository
```bash
git clone <repo_url>
cd <repo_name>
```
Step 2 â€“ Setup Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Step 3 â€“ Place Dataset
```bash
data/NewsSumm_Dataset.xlsx
```
Step 4 â€“ Run Full Pipeline
```bash
python scripts/data_preparation/clean_dataset.py
python scripts/data_preparation/prepare_and_compute.py --input data/NewsSumm_Cleaned.xlsx --output_dir data/enhanced --reports_dir reports --docs_per_cluster 30 --cluster_tfidf_max_features 10000 --cluster_svd_components 64 --preflight_only
python scripts/data_preparation/prepare_and_compute.py --input data/NewsSumm_Cleaned.xlsx --output_dir data/enhanced --reports_dir reports --docs_per_cluster 30 --cluster_tfidf_max_features 10000 --cluster_svd_components 64 --skip_language_filter --skip_minhash_dedup --skip_tfidf_dedup
python scripts/data_preparation/compute_stats.py --data data/enhanced/newssumm_enhanced.json
```
Step 5 â€“ Train Model
```bash
python scripts/training/train_baseline.py --config configs/led_baseline.yaml --sample 25000
```
or Train HPG
```bash
python scripts/training/train_HPG.py --data data/newssumm_processed/newssumm_processed.json --run_name hpg_v2_run_001
```
Step 6 â€“ Evaluate
```bash
python scripts/evaluation/run_evaluation_json.py --run_dir results/<run_name> --data data/newssumm_processed/newssumm_processed.json --sample 10000
```
### For Prompt Based Evaluation
```bash
python scripts/evaluation/prompted_eval.py --model google/flan-t5-xl --data data/newssumm_processed/newssumm_processed.json --sample 10000 --out_dir results/flan_prompt
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
- Mixtral-8Ã—7B-Instruct (if memory allows)

# 14. Research Goals

This repository supports:

- Long-context multi-document summarization
- Hierarchical planning architectures
- Redundancy-aware modeling
- Controlled baseline benchmarking
- Fully reproducible experiments

