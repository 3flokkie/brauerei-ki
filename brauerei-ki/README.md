# Brauerei KI

Brauerei KI is a small command-line demo for training and inspecting a brewery quality-risk model.
It supports synthetic data by default and can also train from external CSV or SQLite inputs.

## Project Layout

- `main.py` - entrypoint for the CLI
- `brewing/` - Python package with the application logic
- `artifacts/` - generated models, logs, metrics, and plots
- `requirements.txt` - runtime dependencies

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train on synthetic data:

```bash
python3 main.py train --quiet
```

Train on CSV data:

```bash
python3 main.py train --train-csv data/real_brewing_data.csv --quiet
```

Train on SQLite data:

```bash
python3 main.py train --train-sqlite data/real_brewing_data.db --sqlite-query "SELECT * FROM brewery_data" --quiet
```

Predict a single batch:

```bash
python3 main.py predict --temperature 69 --pressure 1.7 --mash-time 57
```

Predict a batch file:

```bash
python3 main.py predict-batch --input-csv batches.csv --output-csv artifacts/batch_predictions.csv
```

Evaluate the saved model:

```bash
python3 main.py evaluate --quiet
```

Show saved model metadata:

```bash
python3 main.py info
```

## Notes

- The default model bundle is stored in `artifacts/model_bundle.joblib`.
- Generated metrics and plots are written into `artifacts/`.
- `script.py` is kept as a compatibility shim for the old entrypoint.
