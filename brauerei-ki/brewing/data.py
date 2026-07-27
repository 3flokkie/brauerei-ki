import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .utils import ensure_feature_columns
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from brewing.utils import ensure_feature_columns

FEATURE_COLUMNS = ['Temperatur', 'Druck', 'Maischzeit']
TARGET_COLUMN = 'Qualitaetsrisiko'


def generate_brewery_data(n_samples: int, random_state: int) -> pd.DataFrame:
    """Generate synthetic brewery sensor data with a noisy binary quality-risk label."""
    rng = np.random.default_rng(random_state)

    temperature = rng.normal(loc=65, scale=3, size=n_samples)
    pressure = rng.normal(loc=1.5, scale=0.2, size=n_samples)
    mash_time = rng.normal(loc=60, scale=5, size=n_samples)

    temperature_penalty = np.maximum(0.0, (60.0 - temperature) / 4.0) + np.maximum(0.0, (temperature - 70.0) / 4.0)
    pressure_penalty = np.maximum(0.0, (pressure - 1.6) / 0.15)
    mash_time_penalty = np.maximum(0.0, (55.0 - mash_time) / 4.0)

    latent_risk_score = (
        1.2 * temperature_penalty
        + 1.4 * pressure_penalty
        + 0.8 * mash_time_penalty
        + rng.normal(loc=0.0, scale=0.35, size=n_samples)
    )
    risk_probability = 1.0 / (1.0 + np.exp(-(latent_risk_score - 1.0)))
    risk = rng.binomial(1, risk_probability).astype(int)

    return pd.DataFrame(
        {
            'Temperatur': temperature,
            'Druck': pressure,
            'Maischzeit': mash_time,
            'Qualitaetsrisiko': risk,
        }
    )


def load_or_generate_data(
    csv_path: Optional[Path] = None,
    sqlite_path: Optional[Path] = None,
    sqlite_query: str = 'SELECT * FROM brewery_data',
    n_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load data from CSV or generate synthetic data when no CSV is provided."""
    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(f'CSV file not found: {csv_path}')
        data = pd.read_csv(csv_path)
        ensure_feature_columns(data, FEATURE_COLUMNS, context='CSV Data Loading')
        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' missing in CSV!")
        return data

    if sqlite_path is not None:
        if not sqlite_path.exists():
            raise FileNotFoundError(f'SQLite database not found: {sqlite_path}')
        with sqlite3.connect(sqlite_path) as connection:
            data = pd.read_sql_query(sqlite_query, connection)
        ensure_feature_columns(data, FEATURE_COLUMNS, context='SQLite Data Loading')
        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' missing in SQLite data!")
        return data

    return generate_brewery_data(n_samples=n_samples, random_state=random_state)


def split_data(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test partitions."""
    ensure_feature_columns(data, FEATURE_COLUMNS, context='Training data')
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]

    class_counts = target.value_counts()
    can_stratify = len(class_counts) >= 2 and class_counts.min() >= 2 and int(round(len(target) * test_size)) >= len(class_counts)

    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target if can_stratify else None,
        random_state=random_state,
    )
