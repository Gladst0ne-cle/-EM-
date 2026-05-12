# GMM-EM 

## 1. Environment

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Run

```bash
python run.py
```

This command will automatically:

- run single experiment
- run batch robustness experiment
- run BIC model selection
- run submission file check

## 3. Project structure

- `em_gmm.py`: Core GMM-EM implementation from scratch
- `config.py`: Shared experiment settings
- `data_utils.py`: Dataset loading and preprocessing
- `evaluation.py`: Metrics, summaries, and output saving
- `run.py`: One-click entry point (recommended)
- `outputs/`: Auto-generated reports (`.json`, `.csv`)

## 4. Metrics and outputs

- Accuracy (best permutation mapping)
- ARI (Adjusted Rand Index)
- NMI (Normalized Mutual Information)
- Final log-likelihood
- BIC

Generated files:

- `outputs/single_run_report.json`
- `outputs/batch_summary.json`
- `outputs/batch_results.csv`
- `outputs/model_selection.json`


