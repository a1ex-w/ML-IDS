# ML-NIDS

Machine learning network intrusion detection system built for my GCU MS Computer Science capstone (CST-590). It classifies network traffic from the CICIDS2017 dataset as benign or attack using Random Forest, Logistic Regression, and Linear SVM, and shows the results in a Flask dashboard.

## Setup

1. Install Python 3.14 and [uv](https://docs.astral.sh/uv/)
2. Clone the repo and install dependencies:
   ```
   git clone https://github.com/a1ex-w/ML-IDS.git
   cd ML-IDS
   uv sync
   ```
3. Download the CICIDS2017 MachineLearningCVE CSVs from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) and put all 8 files in `data/MachineLearningCVE/`

## Run

```
uv run pipeline/dataset_manager.py
uv run pipeline/model_trainer.py
uv run pipeline/evaluator.py
uv run app.py
```

Then open http://127.0.0.1:5001

## Project Structure

- `pipeline/` – data cleaning, validation, training, and evaluation scripts
- `app.py` – Flask app
- `templates/`, `static/` – dashboard pages and styling

The `data/`, `models/`, and `static/plots/` folders are created by the pipeline and are not stored in the repo.