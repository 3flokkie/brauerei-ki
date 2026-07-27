from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from .data import FEATURE_COLUMNS, TARGET_COLUMN, split_data
    from .evaluation import (
        compute_binary_metrics,
        save_confusion_matrix_plot,
        save_metrics_json,
        save_permutation_importance_plot,
        save_shap_explanation_plot,
        save_temperature_distribution,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from brewing.data import FEATURE_COLUMNS, TARGET_COLUMN, split_data
    from brewing.evaluation import (
        compute_binary_metrics,
        save_confusion_matrix_plot,
        save_metrics_json,
        save_permutation_importance_plot,
        save_shap_explanation_plot,
        save_temperature_distribution,
    )


def train_model_bundle(
    data: pd.DataFrame,
    output_path: Path,
    artifacts_dir: Path,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Train a classifier, save model bundle and key artefacts."""
    x_train, x_test, y_train, y_test = split_data(
        data=data,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1] if len(np.unique(y_test)) > 1 else None
    metrics = compute_binary_metrics(y_test, y_pred, y_proba)

    bundle = {
        'model': model,
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'metrics': metrics,
        'x_test': x_test,
        'y_test': y_test,
        'training_config': {
            'random_state': random_state,
            'test_size': test_size,
        },
        'version': '1.0.0',
    }

    save_model_bundle(bundle, output_path)
    save_metrics_json(metrics, artifacts_dir / 'train_metrics.json')
    save_confusion_matrix_plot(y_test, y_pred, artifacts_dir / 'confusion_matrix.png')
    save_temperature_distribution(data, artifacts_dir / 'temperature_distribution.png')
    save_permutation_importance_plot(
        model,
        x_test,
        y_test,
        artifacts_dir / 'permutation_importance.png',
        random_state=random_state,
    )
    save_shap_explanation_plot(
        model,
        x_train,
        x_test,
        artifacts_dir / 'shap_summary.png',
    )

    return bundle


def save_model_bundle(bundle: dict[str, Any], output_path: Path) -> None:
    """Persist the trained model bundle to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def load_model_bundle(input_path: Path) -> dict[str, Any]:
    """Load a serialized model bundle from disk."""
    if not input_path.exists():
        raise FileNotFoundError(f'Model bundle not found: {input_path}')
    return joblib.load(input_path)
