import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

try:
    from .data import FEATURE_COLUMNS
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from brewing.data import FEATURE_COLUMNS


def save_metrics_json(metrics: dict[str, Any], output_path: Path) -> None:
    """Persist metrics dictionary as pretty JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

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

    if y_proba is not None and len(np.unique(y_true)) > 1:
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
