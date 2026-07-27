from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError, RawDescriptionHelpFormatter
from datetime import datetime, timezone
import json
from pathlib import Path
import logging
import pickle
import sys
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

DEFAULT_RANDOM_STATE = 42
DEFAULT_SAMPLES = 1000
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
DEFAULT_SHAP_SAMPLE_SIZE = 150
DEFAULT_TEMPERATURE = 68.5
DEFAULT_PRESSURE = 1.9
DEFAULT_MASH_TIME = 58.0
CLI_VERSION = '1.1.0'
FEATURE_COLUMNS = ['Temperatur', 'Druck', 'Maischzeit']
TARGET_COLUMN = 'Qualitaetsrisiko'

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = SCRIPT_DIR / 'artifacts'
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACTS_DIR / 'model_bundle.pkl'
DEFAULT_LOG_PATH = DEFAULT_ARTIFACTS_DIR / 'brewing_cli.log'
DEFAULT_BATCH_OUTPUT_PATH = DEFAULT_ARTIFACTS_DIR / 'batch_predictions.csv'
DEFAULT_EVAL_METRICS_PATH = DEFAULT_ARTIFACTS_DIR / 'evaluation_metrics.json'


class _CLIFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    """Argument parser formatter that keeps multiline examples readable."""

    pass


def build_parser() -> ArgumentParser:
    """Create the command line interface."""
    parser = ArgumentParser(
        description='Train, predict, and explain a brewery quality-risk model.',
        formatter_class=_CLIFormatter,
        epilog=(
            'Examples:\n'
            '  python3 script.py train --quiet\n'
            '  python3 script.py predict --temperature 69 --pressure 1.7 --mash-time 57\n'
            '  python3 script.py predict --temperature 64 --pressure 1.4 --mash-time 62\n'
            '  python3 script.py predict-batch --input-csv batches.csv --output-csv artifacts/batch_predictions.csv\n'
            '  python3 script.py evaluate --quiet\n'
            '  python3 script.py info\n'
        ),
    )
    parser.add_argument('--version', action='version', version=f'brewery-cli {CLI_VERSION}')
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser(
        'train',
        help='Train a tuned model and save artifacts.',
        formatter_class=_CLIFormatter,
    )
    add_training_arguments(train_parser)
    train_parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE, help='Example temperature for the trained model to evaluate.')
    train_parser.add_argument('--pressure', type=float, default=DEFAULT_PRESSURE, help='Example pressure for the trained model to evaluate.')
    train_parser.add_argument('--mash-time', type=float, default=DEFAULT_MASH_TIME, help='Example mash time for the trained model to evaluate.')
    train_parser.add_argument('--skip-plots', action='store_true', help='Skip saving plot artifacts.')
    train_parser.add_argument('--skip-permutation', action='store_true', help='Skip the permutation-importance plot.')
    train_parser.add_argument('--skip-shap', action='store_true', help='Skip the SHAP summary plot.')
    train_parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_PATH, help='Where to save the trained model bundle.')
    train_parser.add_argument('--quiet', action='store_true', help='Reduce console output to the essential summary.')
    train_parser.set_defaults(handler=cmd_train)

    predict_parser = subparsers.add_parser(
        'predict',
        help='Load a saved model bundle and predict one batch.',
        formatter_class=_CLIFormatter,
    )
    predict_parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_PATH, help='Path to the saved model bundle.')
    predict_parser.add_argument('--output-dir', type=Path, default=DEFAULT_ARTIFACTS_DIR, help='Directory where logs are saved.')
    predict_parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE, help='Batch temperature in degrees Celsius.')
    predict_parser.add_argument('--pressure', type=float, default=DEFAULT_PRESSURE, help='Batch pressure in bar.')
    predict_parser.add_argument('--mash-time', type=float, default=DEFAULT_MASH_TIME, help='Batch mash time in minutes.')
    predict_parser.add_argument('--output-format', choices=['text', 'json'], default='text', help='Output format for prediction details.')
    predict_parser.add_argument('--quiet', action='store_true', help='Print only the final prediction.')
    predict_parser.set_defaults(handler=cmd_predict)

    batch_predict_parser = subparsers.add_parser(
        'predict-batch',
        help='Predict risk for multiple batches loaded from a CSV file.',
        formatter_class=_CLIFormatter,
    )
    batch_predict_parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_PATH, help='Path to the saved model bundle.')
    batch_predict_parser.add_argument('--input-csv', type=Path, required=True, help='Input CSV with feature columns.')
    batch_predict_parser.add_argument('--output-csv', type=Path, default=DEFAULT_BATCH_OUTPUT_PATH, help='Output CSV path for predictions.')
    batch_predict_parser.add_argument('--output-dir', type=Path, default=DEFAULT_ARTIFACTS_DIR, help='Directory where logs are saved.')
    batch_predict_parser.add_argument('--quiet', action='store_true', help='Print only a short summary.')
    batch_predict_parser.set_defaults(handler=cmd_predict_batch)

    explain_parser = subparsers.add_parser(
        'explain',
        help='Generate explainability artifacts for a saved model bundle.',
        formatter_class=_CLIFormatter,
    )
    add_training_arguments(explain_parser)
    explain_parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_PATH, help='Path to the saved model bundle.')
    explain_parser.add_argument('--skip-permutation', action='store_true', help='Skip the permutation-importance plot.')
    explain_parser.add_argument('--skip-shap', action='store_true', help='Skip the SHAP summary plot.')
    explain_parser.add_argument('--quiet', action='store_true', help='Reduce console output to the essential summary.')
    explain_parser.set_defaults(handler=cmd_explain)

    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a saved model on synthetic holdout data and save metrics.',
        formatter_class=_CLIFormatter,
    )
    add_training_arguments(evaluate_parser)
    evaluate_parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_PATH, help='Path to the saved model bundle.')
    evaluate_parser.add_argument('--metrics-json', type=Path, default=DEFAULT_EVAL_METRICS_PATH, help='Path to save evaluation metrics JSON.')
    evaluate_parser.add_argument('--skip-confusion-plot', action='store_true', help='Skip saving a confusion matrix plot.')
    evaluate_parser.add_argument('--quiet', action='store_true', help='Reduce console output to the essential summary.')
    evaluate_parser.set_defaults(handler=cmd_evaluate)

    info_parser = subparsers.add_parser(
        'info',
        help='Show metadata stored in a saved model bundle.',
        formatter_class=_CLIFormatter,
    )
    info_parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_PATH, help='Path to the saved model bundle.')
    info_parser.add_argument('--output-dir', type=Path, default=DEFAULT_ARTIFACTS_DIR, help='Directory where logs are saved.')
    info_parser.add_argument('--quiet', action='store_true', help='Print compact machine-readable metadata.')
    info_parser.set_defaults(handler=cmd_info)

    parser.set_defaults(handler=cmd_train, command='train')
    return parser


def add_training_arguments(parser: ArgumentParser) -> None:
    """Add shared training and explanation arguments."""
    parser.add_argument('--samples', type=positive_int, default=DEFAULT_SAMPLES, help='Number of synthetic samples to generate.')
    parser.add_argument('--random-state', type=int, default=DEFAULT_RANDOM_STATE, help='Random seed for reproducibility.')
    parser.add_argument('--test-size', type=fraction_between_zero_and_one, default=DEFAULT_TEST_SIZE, help='Fraction of data used for the holdout test split.')
    parser.add_argument('--cv-folds', type=minimum_two, default=DEFAULT_CV_FOLDS, help='Cross-validation folds for hyperparameter tuning.')
    parser.add_argument('--shap-sample-size', type=positive_int, default=DEFAULT_SHAP_SAMPLE_SIZE, help='Number of test rows to use for SHAP explanations.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_ARTIFACTS_DIR, help='Directory where logs and plots are saved.')


def positive_int(value: str) -> int:
    """Argparse type validator for positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise ArgumentTypeError('must be a positive integer')
    return parsed


def minimum_two(value: str) -> int:
    """Argparse type validator for integers greater than or equal to two."""
    parsed = int(value)
    if parsed < 2:
        raise ArgumentTypeError('must be an integer greater than or equal to 2')
    return parsed


def fraction_between_zero_and_one(value: str) -> float:
    """Argparse type validator for open interval (0, 1)."""
    parsed = float(value)
    if not (0.0 < parsed < 1.0):
        raise ArgumentTypeError('must be a float strictly between 0 and 1')
    return parsed


def setup_logging(log_path: Path) -> None:
    """Configure file and console logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(),
        ],
        force=True,
    )


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def ensure_feature_columns(frame: pd.DataFrame, context: str) -> None:
    """Validate that all required feature columns are present."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns {missing}. Expected columns: {FEATURE_COLUMNS}")


def save_metrics_json(metrics: dict[str, Any], output_path: Path) -> None:
    """Persist metrics dictionary as pretty JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file_handle:
        json.dump(metrics, file_handle, ensure_ascii=False, indent=2)


def compute_binary_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> dict[str, float]:
    """Compute standard binary-classification metrics."""
    metrics: dict[str, float] = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))

    return metrics


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save a confusion matrix heatmap."""
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Optimal', 'Warnung'])
    ax.set_yticklabels(['Optimal', 'Warnung'])

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, str(matrix[row_idx, col_idx]), ha='center', va='center', color='black')

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_brewery_data(n_samples: int, random_state: int) -> pd.DataFrame:
    """Generate synthetic brewery sensor data with a binary quality-risk label."""
    rng = np.random.default_rng(random_state)

    temperature = rng.normal(loc=65, scale=3, size=n_samples)
    pressure = rng.normal(loc=1.5, scale=0.2, size=n_samples)
    mash_time = rng.normal(loc=60, scale=5, size=n_samples)

    risk = (
        (temperature < 60)
        | (temperature > 70)
        | (pressure > 1.8)
        | (mash_time < 50)
    ).astype(int)

    return pd.DataFrame(
        {
            'Temperatur': temperature,
            'Druck': pressure,
            'Maischzeit': mash_time,
            'Qualitaetsrisiko': risk,
        }
    )


def split_data(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test partitions."""
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]

    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )


def tune_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    cv_folds: int,
) -> tuple[RandomForestClassifier, dict[str, Any], float]:
    """Tune hyperparameters using grid search and return the best estimator."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 8, 12],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv_folds,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)

    return search.best_estimator_, search.best_params_, float(search.best_score_)


def predict_risk(
    model: RandomForestClassifier,
    temperature: float,
    pressure: float,
    mash_time: float,
) -> int:
    """Predict whether a batch is risky."""
    risk, _ = predict_risk_with_probability(model, temperature, pressure, mash_time)
    return risk


def predict_risk_with_probability(
    model: RandomForestClassifier,
    temperature: float,
    pressure: float,
    mash_time: float,
) -> tuple[int, Optional[float]]:
    """Predict risk label and optional positive-class probability."""
    sample_batch = pd.DataFrame(
        [[temperature, pressure, mash_time]],
        columns=FEATURE_COLUMNS,
    )
    risk = int(model.predict(sample_batch)[0])
    probability: Optional[float] = None

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(sample_batch)
        if isinstance(probabilities, np.ndarray) and probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            probability = float(probabilities[0, 1])

    return risk, probability


def predict_batch_with_probability(
    model: RandomForestClassifier,
    batches: pd.DataFrame,
) -> pd.DataFrame:
    """Predict risk labels and optional probabilities for multiple rows."""
    ensure_feature_columns(batches, context='Batch prediction input')
    features = batches[FEATURE_COLUMNS]

    risk_predictions = model.predict(features).astype(int)
    risk_probability: Optional[np.ndarray] = None

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)
        if isinstance(probabilities, np.ndarray) and probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            risk_probability = probabilities[:, 1]

    result = batches.copy()
    result['Risikoklasse'] = risk_predictions
    result['Risikostatus'] = np.where(risk_predictions == 1, 'WARNUNG: Qualitätsabweichung!', 'Optimal')
    if risk_probability is not None:
        result['Risikowahrscheinlichkeit'] = risk_probability

    return result


def save_temperature_distribution(data: pd.DataFrame, output_path: Path) -> None:
    """Save a temperature histogram for the generated brewery data."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data['Temperatur'], bins=20, color='skyblue', edgecolor='black')
    ax.set_title('Temperaturverteilung der Brauereidaten')
    ax.set_xlabel('Temperatur (°C)')
    ax.set_ylabel('Häufigkeit')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_permutation_importance_plot(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
    random_state: int,
) -> pd.DataFrame:
    """Save a permutation importance chart and return the rankings."""
    importance = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_frame = pd.DataFrame(
        {
            'Merkmal': FEATURE_COLUMNS,
            'Mittelwert': importance.importances_mean,
            'StdAbw': importance.importances_std,
        }
    ).sort_values('Mittelwert', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        importance_frame['Merkmal'],
        importance_frame['Mittelwert'],
        xerr=importance_frame['StdAbw'],
        color='#4c78a8',
        alpha=0.9,
    )
    ax.set_title('Permutation Importances')
    ax.set_xlabel('Einfluss auf die Genauigkeit')
    ax.set_ylabel('Merkmal')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return importance_frame


def normalize_shap_values(shap_values: Any) -> np.ndarray:
    """Convert SHAP output into a 2D matrix for the positive class or single output."""
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_matrix = shap_values

    shap_matrix = np.asarray(shap_matrix)
    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[..., -1]
    elif shap_matrix.ndim > 2:
        shap_matrix = np.squeeze(shap_matrix)

    return shap_matrix


def save_shap_explanation_plot(
    model: RandomForestClassifier,
    x_background: pd.DataFrame,
    x_explain: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Save a SHAP summary plot and return mean absolute SHAP values."""
    explainer = shap.TreeExplainer(model, data=x_background)
    shap_values = explainer.shap_values(x_explain)
    shap_matrix = normalize_shap_values(shap_values)

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    shap_frame = pd.DataFrame(
        {
            'Merkmal': FEATURE_COLUMNS,
            'MeanAbsSHAP': mean_abs_shap,
        }
    ).sort_values('MeanAbsSHAP', ascending=True)

    shap.summary_plot(shap_matrix, x_explain, show=False, plot_type='bar')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return shap_frame


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    """Persist the trained model bundle to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as file_handle:
        pickle.dump(bundle, file_handle)


def load_bundle(path: Path) -> dict[str, Any]:
    """Load a persisted model bundle from disk."""
    if not path.exists():
        raise FileNotFoundError(f'Model bundle not found: {path}')

    with path.open('rb') as file_handle:
        bundle = pickle.load(file_handle)

    if not isinstance(bundle, dict) or 'model' not in bundle:
        raise ValueError(f'Invalid model bundle: {path}')

    model = bundle['model']
    if not hasattr(model, 'predict'):
        raise ValueError(f'Invalid model in bundle: {path}')

    bundle_features = bundle.get('feature_columns')
    if bundle_features is None:
        raise ValueError(f'Missing feature columns in bundle: {path}')
    if list(bundle_features) != FEATURE_COLUMNS:
        raise ValueError(
            f'Feature mismatch in bundle: expected {FEATURE_COLUMNS}, got {list(bundle_features)}'
        )

    return bundle


def log_run_summary(
    command_name: str,
    best_params: Optional[dict[str, Any]],
    cv_accuracy: Optional[float],
    test_accuracy: Optional[float],
    risk: Optional[int],
) -> None:
    """Log the main results of a run."""
    logging.info('Command: %s', command_name)
    if best_params is not None:
        logging.info('Best parameters: %s', best_params)
    if cv_accuracy is not None:
        logging.info('Cross-validation accuracy: %.2f%%', cv_accuracy * 100)
    if test_accuracy is not None:
        logging.info('Holdout test accuracy: %.2f%%', test_accuracy * 100)
    if risk is not None:
        logging.info('Example batch risk: %s', 'WARNUNG' if risk == 1 else 'Optimal')


def log_metrics(metrics: dict[str, float], label: str) -> None:
    """Log metrics in a uniform format."""
    for metric_name, metric_value in metrics.items():
        logging.info('%s %s: %.4f', label, metric_name, metric_value)


def build_example_batch(temperature: float, pressure: float, mash_time: float) -> tuple[float, float, float]:
    """Return the example b atch values."""
    return temperature, pressure, mash_time


def cmd_train(args: Any) -> int:
    """Train a model, save artifacts, and print a concise summary."""
    output_dir = args.output_dir
    log_path = output_dir / 'brewing_cli.log'
    bundle_path = args.model_path
    temperature_plot_path = output_dir / 'brewing_temperature_distribution.png'
    permutation_plot_path = output_dir / 'model_permutation_importance.png'
    shap_plot_path = output_dir / 'model_shap_summary.png'
    confusion_plot_path = output_dir / 'train_confusion_matrix.png'
    metrics_json_path = output_dir / 'train_metrics.json'

    ensure_directory(output_dir)
    setup_logging(log_path)

    try:
        data = generate_brewery_data(args.samples, args.random_state)
        x_train, x_test, y_train, y_test = split_data(data, args.test_size, args.random_state)
        model, best_params, cv_accuracy = tune_model(x_train, y_train, args.random_state, args.cv_folds)

        predictions = model.predict(x_test)
        probabilities: Optional[np.ndarray] = None
        if hasattr(model, 'predict_proba'):
            proba_matrix = model.predict_proba(x_test)
            if isinstance(proba_matrix, np.ndarray) and proba_matrix.ndim == 2 and proba_matrix.shape[1] >= 2:
                probabilities = proba_matrix[:, 1]

        metrics = compute_binary_metrics(y_test, predictions, probabilities)
        test_accuracy = metrics['accuracy']

        bundle = {
            'model': model,
            'feature_columns': FEATURE_COLUMNS,
            'best_params': best_params,
            'cv_accuracy': cv_accuracy,
            'test_accuracy': test_accuracy,
            'random_state': args.random_state,
            'samples': args.samples,
            'test_size': args.test_size,
            'metrics': metrics,
            'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        save_bundle(bundle, bundle_path)
        save_metrics_json(metrics, metrics_json_path)

        if not args.skip_plots:
            save_confusion_matrix_plot(y_test, predictions, confusion_plot_path)

        permutation_frame = pd.DataFrame(columns=['Merkmal', 'Mittelwert', 'StdAbw'])
        shap_frame = pd.DataFrame(columns=['Merkmal', 'MeanAbsSHAP'])

        if not args.skip_plots:
            save_temperature_distribution(data, temperature_plot_path)
            if not args.skip_permutation:
                try:
                    permutation_frame = save_permutation_importance_plot(
                        model,
                        x_test,
                        y_test,
                        permutation_plot_path,
                        args.random_state,
                    )
                except Exception:
                    logging.exception('Failed to generate permutation-importance plot')
                    if not args.quiet:
                        print('Warnung: Permutation-Importance konnte nicht erstellt werden.')
            if not args.skip_shap:
                shap_sample_size = min(args.shap_sample_size, len(x_test))
                background_size = min(100, len(x_train))
                if shap_sample_size == 0 or background_size == 0:
                    if not args.quiet:
                        print('Warnung: SHAP-Summary wurde übersprungen, da zu wenig Daten verfügbar sind.')
                else:
                    x_background = x_train.sample(n=background_size, random_state=args.random_state)
                    x_explain = x_test.sample(n=shap_sample_size, random_state=args.random_state)
                    try:
                        shap_frame = save_shap_explanation_plot(model, x_background, x_explain, shap_plot_path)
                    except Exception:
                        logging.exception('Failed to generate SHAP summary plot')
                        if not args.quiet:
                            print('Warnung: SHAP-Summary konnte nicht erstellt werden.')

        temperature, pressure, mash_time = build_example_batch(
            args.temperature,
            args.pressure,
            args.mash_time,
        )
        risk, risk_probability = predict_risk_with_probability(model, temperature, pressure, mash_time)

        log_run_summary('train', best_params, cv_accuracy, test_accuracy, risk)
        log_metrics(metrics, 'Train')

        if not args.quiet:
            print('--- Modell erfolgreich trainiert! ---')
            print(f'Modell gespeichert unter: {bundle_path}')
            print(f'Beste Hyperparameter: {best_params}')
            print(f'Cross-Validation-Genauigkeit: {cv_accuracy * 100:.2f}%')
            print(f'Prognosegenauigkeit des Risikos: {test_accuracy * 100:.2f}%\n')
            print(
                'Weitere Metriken: '
                f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
                + (f", ROC-AUC={metrics['roc_auc']:.3f}" if 'roc_auc' in metrics else '')
            )
            print(f'Testmessung: Temp={temperature:.1f}°C, Druck={pressure:.1f}bar, Zeit={mash_time:.0f}min')
            print(f"Ergebnis -> Risikostatus: {'WARNUNG: Qualitätsabweichung!' if risk == 1 else 'Optimal'}")
            if risk_probability is not None:
                print(f'Risikowahrscheinlichkeit: {risk_probability * 100:.2f}%')
            print(f'Metriken (JSON): {metrics_json_path}')
            print(f'Logdatei: {log_path}')
            if not args.skip_plots:
                print(f'Diagramm gespeichert unter: {temperature_plot_path}')
                print(f'Confusion Matrix gespeichert unter: {confusion_plot_path}')
                if not args.skip_permutation:
                    print(f'Permutation-Importance gespeichert unter: {permutation_plot_path}')
                    print('Permutation-Importances:')
                    for _, row in permutation_frame.iterrows():
                        print(f"- {row['Merkmal']}: {row['Mittelwert']:.4f} ± {row['StdAbw']:.4f}")
                if not args.skip_shap:
                    print(f'SHAP-Summary gespeichert unter: {shap_plot_path}')
                    print('SHAP-Mean-Absolute-Values:')
                    for _, row in shap_frame.iterrows():
                        print(f"- {row['Merkmal']}: {row['MeanAbsSHAP']:.4f}")
        else:
            print(f'Risikostatus: {"WARNUNG: Qualitätsabweichung!" if risk == 1 else "Optimal"}')
            if risk_probability is not None:
                print(f'Risikowahrscheinlichkeit: {risk_probability * 100:.2f}%')
            print(f'Modell gespeichert unter: {bundle_path}')
            print(f'Metriken (JSON): {metrics_json_path}')
            print(f'Logdatei: {log_path}')

        return 0
    except Exception:
        logging.exception('Training pipeline failed')
        print(f'Fehler: Der Train-Befehl ist fehlgeschlagen. Details stehen in {log_path}')
        return 1


def cmd_predict(args: Any) -> int:

    output_dir = args.output_dir
    log_path = output_dir / 'brewing_cli.log'
    ensure_directory(output_dir)
    setup_logging(log_path)

    try:
        bundle = load_bundle(args.model_path)
        model = bundle['model']
        temperature, pressure, mash_time = build_example_batch(
            args.temperature,
            args.pressure,
            args.mash_time,
        )
        risk, risk_probability = predict_risk_with_probability(model, temperature, pressure, mash_time)

        log_run_summary('predict', bundle.get('best_params'), bundle.get('cv_accuracy'), bundle.get('test_accuracy'), risk)

        if not args.quiet:
            print('--- Vorhersage abgeschlossen ---')
            print(f'Modell geladen von: {args.model_path}')
            print(f'Testmessung: Temp={temperature:.1f}°C, Druck={pressure:.1f}bar, Zeit={mash_time:.0f}min')
        if args.output_format == 'json':
            payload: dict[str, Any] = {
                'temperature': temperature,
                'pressure': pressure,
                'mash_time': mash_time,
                'risk_class': risk,
                'risk_status': 'WARNUNG: Qualitätsabweichung!' if risk == 1 else 'Optimal',
            }
            if risk_probability is not None:
                payload['risk_probability'] = risk_probability
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"Risikostatus: {'WARNUNG: Qualitätsabweichung!' if risk == 1 else 'Optimal'}")
            if risk_probability is not None:
                print(f'Risikowahrscheinlichkeit: {risk_probability * 100:.2f}%')
        print(f'Logdatei: {log_path}')
        return 0
    except Exception:
        logging.exception('Predict pipeline failed')
        print(f'Fehler: Der Predict-Befehl ist fehlgeschlagen. Details stehen in {log_path}')
        return 1


def cmd_predict_batch(args: Any) -> int:
    """Load a model and generate predictions for all rows in a CSV file."""
    output_dir = args.output_dir
    log_path = output_dir / 'brewing_cli.log'
    ensure_directory(output_dir)
    setup_logging(log_path)

    try:
        bundle = load_bundle(args.model_path)
        model = bundle['model']

        if not args.input_csv.exists():
            raise FileNotFoundError(f'Input CSV not found: {args.input_csv}')

        batches = pd.read_csv(args.input_csv)
        predicted = predict_batch_with_probability(model, batches)

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        predicted.to_csv(args.output_csv, index=False)

        risk_count = int((predicted['Risikoklasse'] == 1).sum())
        total_count = int(len(predicted))
        risk_ratio = (risk_count / total_count * 100.0) if total_count else 0.0

        logging.info('Batch prediction rows: %d', total_count)
        logging.info('Batch prediction risk rows: %d', risk_count)

        if not args.quiet:
            print('--- Batch-Vorhersage abgeschlossen ---')
            print(f'Modell geladen von: {args.model_path}')
            print(f'Input CSV: {args.input_csv}')
            print(f'Output CSV: {args.output_csv}')
            print(f'Anzahl Chargen: {total_count}')
            print(f'Warnungen: {risk_count} ({risk_ratio:.2f}%)')
        else:
            print(f'Batch-Vorhersage: {risk_count}/{total_count} Warnungen ({risk_ratio:.2f}%)')
        print(f'Logdatei: {log_path}')
        return 0
    except Exception:
        logging.exception('Batch predict pipeline failed')
        print(f'Fehler: Der Predict-Batch-Befehl ist fehlgeschlagen. Details stehen in {log_path}')
        return 1


def cmd_explain(args: Any) -> int:
    """Generate permutation and SHAP explanation artifacts for a saved model."""
    output_dir = args.output_dir
    log_path = output_dir / 'brewing_cli.log'
    permutation_plot_path = output_dir / 'model_permutation_importance.png'
    shap_plot_path = output_dir / 'model_shap_summary.png'
    ensure_directory(output_dir)
    setup_logging(log_path)

    try:
        bundle = load_bundle(args.model_path)
        model = bundle['model']
        data = generate_brewery_data(args.samples, args.random_state)
        x_train, x_test, y_train, y_test = split_data(data, args.test_size, args.random_state)

        permutation_frame = pd.DataFrame(columns=['Merkmal', 'Mittelwert', 'StdAbw'])
        shap_frame = pd.DataFrame(columns=['Merkmal', 'MeanAbsSHAP'])

        if not args.skip_permutation:
            try:
                permutation_frame = save_permutation_importance_plot(
                    model,
                    x_test,
                    y_test,
                    permutation_plot_path,
                    args.random_state,
                )
            except Exception:
                logging.exception('Failed to generate permutation-importance plot')
                if not args.quiet:
                    print('Warnung: Permutation-Importance konnte nicht erstellt werden.')
        if not args.skip_shap:
            shap_sample_size = min(args.shap_sample_size, len(x_test))
            background_size = min(100, len(x_train))
            if shap_sample_size == 0 or background_size == 0:
                if not args.quiet:
                    print('Warnung: SHAP-Summary wurde übersprungen, da zu wenig Daten verfügbar sind.')
            else:
                x_background = x_train.sample(n=background_size, random_state=args.random_state)
                x_explain = x_test.sample(n=shap_sample_size, random_state=args.random_state)
                try:
                    shap_frame = save_shap_explanation_plot(model, x_background, x_explain, shap_plot_path)
                except Exception:
                    logging.exception('Failed to generate SHAP summary plot')
                    if not args.quiet:
                        print('Warnung: SHAP-Summary konnte nicht erstellt werden.')

        log_run_summary('explain', bundle.get('best_params'), bundle.get('cv_accuracy'), bundle.get('test_accuracy'), None)

        if not args.quiet:
            print('--- Erklärung abgeschlossen ---')
            print(f'Modell geladen von: {args.model_path}')
            if not args.skip_permutation:
                print(f'Permutation-Importance gespeichert unter: {permutation_plot_path}')
                print('Permutation-Importances:')
                for _, row in permutation_frame.iterrows():
                    print(f"- {row['Merkmal']}: {row['Mittelwert']:.4f} ± {row['StdAbw']:.4f}")
            if not args.skip_shap:
                print(f'SHAP-Summary gespeichert unter: {shap_plot_path}')
                print('SHAP-Mean-Absolute-Values:')
                for _, row in shap_frame.iterrows():
                    print(f"- {row['Merkmal']}: {row['MeanAbsSHAP']:.4f}")
            print(f'Logdatei: {log_path}')
        else:
            print('Erklärung abgeschlossen.')
            print(f'Logdatei: {log_path}')

        return 0
    except Exception:
        logging.exception('Explain pipeline failed')
        print(f'Fehler: Der Explain-Befehl ist fehlgeschlagen. Details stehen in {log_path}')
        return 1


def cmd_evaluate(args: Any) -> int:
    """Evaluate a saved model on synthetic data and persist metrics."""
    output_dir = args.output_dir
    log_path = output_dir / 'brewing_cli.log'
    confusion_plot_path = output_dir / 'evaluation_confusion_matrix.png'
    ensure_directory(output_dir)
    setup_logging(log_path)

    try:
        bundle = load_bundle(args.model_path)
        model = bundle['model']
        data = generate_brewery_data(args.samples, args.random_state)
        x_train, x_test, _, y_test = split_data(data, args.test_size, args.random_state)
        ensure_feature_columns(x_train, context='Evaluation split')

        predictions = model.predict(x_test)
        probabilities: Optional[np.ndarray] = None
        if hasattr(model, 'predict_proba'):
            proba_matrix = model.predict_proba(x_test)
            if isinstance(proba_matrix, np.ndarray) and proba_matrix.ndim == 2 and proba_matrix.shape[1] >= 2:
                probabilities = proba_matrix[:, 1]

        metrics = compute_binary_metrics(y_test, predictions, probabilities)
        save_metrics_json(metrics, args.metrics_json)

        if not args.skip_confusion_plot:
            save_confusion_matrix_plot(y_test, predictions, confusion_plot_path)

        log_run_summary('evaluate', bundle.get('best_params'), bundle.get('cv_accuracy'), metrics['accuracy'], None)
        log_metrics(metrics, 'Evaluate')

        if not args.quiet:
            print('--- Evaluierung abgeschlossen ---')
            print(f'Modell geladen von: {args.model_path}')
            print(f'Accuracy: {metrics["accuracy"] * 100:.2f}%')
            print(f'Precision: {metrics["precision"]:.4f}')
            print(f'Recall: {metrics["recall"]:.4f}')
            print(f'F1: {metrics["f1"]:.4f}')
            if 'roc_auc' in metrics:
                print(f'ROC-AUC: {metrics["roc_auc"]:.4f}')
            print(f'Metriken (JSON): {args.metrics_json}')
            if not args.skip_confusion_plot:
                print(f'Confusion Matrix gespeichert unter: {confusion_plot_path}')
            print(f'Logdatei: {log_path}')
        else:
            print(f'Accuracy: {metrics["accuracy"] * 100:.2f}%')
            print(f'Metriken (JSON): {args.metrics_json}')
            print(f'Logdatei: {log_path}')

        return 0
    except Exception:
        logging.exception('Evaluate pipeline failed')
        print(f'Fehler: Der Evaluate-Befehl ist fehlgeschlagen. Details stehen in {log_path}')
        return 1


def cmd_info(args: Any) -> int:
    """Display saved model metadata and tracked metrics."""
    output_dir = args.output_dir
    log_path = output_dir / 'brewing_cli.log'
    ensure_directory(output_dir)
    setup_logging(log_path)

    try:
        bundle = load_bundle(args.model_path)
        payload = {
            'model_path': str(args.model_path),
            'feature_columns': bundle.get('feature_columns'),
            'best_params': bundle.get('best_params'),
            'cv_accuracy': bundle.get('cv_accuracy'),
            'test_accuracy': bundle.get('test_accuracy'),
            'metrics': bundle.get('metrics'),
            'samples': bundle.get('samples'),
            'test_size': bundle.get('test_size'),
            'random_state': bundle.get('random_state'),
            'trained_at_utc': bundle.get('trained_at_utc'),
        }

        logging.info('Model info loaded from: %s', args.model_path)

        if args.quiet:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print('--- Modell-Info ---')
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            print(f'Logdatei: {log_path}')

        return 0
    except Exception:
        logging.exception('Info command failed')
        print(f'Fehler: Der Info-Befehl ist fehlgeschlagen. Details stehen in {log_path}')
        return 1


def main() -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(['train'] if len(sys.argv) == 1 else None)
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
