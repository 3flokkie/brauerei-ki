from argparse import ArgumentTypeError
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd


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


def ensure_feature_columns(frame: pd.DataFrame, expected_columns: Iterable[str], context: str) -> None:
    """Validate that all required feature columns are present."""
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns {missing}. Expected columns: {list(expected_columns)}")
