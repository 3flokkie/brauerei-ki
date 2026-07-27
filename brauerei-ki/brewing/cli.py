from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .data import FEATURE_COLUMNS, load_or_generate_data
    from .model import load_model_bundle, train_model_bundle
    from .utils import ensure_directory
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from brewing.data import FEATURE_COLUMNS, load_or_generate_data
    from brewing.model import load_model_bundle, train_model_bundle
    from brewing.utils import ensure_directory


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description='Brewery quality-risk training and evaluation CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--train-csv', type=Path, default=None)
    train_parser.add_argument('--train-sqlite', type=Path, default=None)
    train_parser.add_argument('--sqlite-query', default='SELECT * FROM brewery_data')
    train_parser.add_argument('--samples', type=int, default=1000)
    train_parser.add_argument('--quiet', action='store_true')
    train_parser.set_defaults(handler=handle_train)

    predict_parser = subparsers.add_parser('predict', help='Predict a single batch')
    predict_parser.add_argument('--temperature', type=float, required=True)
    predict_parser.add_argument('--pressure', type=float, required=True)
    predict_parser.add_argument('--mash-time', type=float, required=True)
    predict_parser.set_defaults(handler=handle_predict)

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the saved model bundle')
    eval_parser.add_argument('--quiet', action='store_true')
    eval_parser.set_defaults(handler=handle_evaluate)

    info_parser = subparsers.add_parser('info', help='Show model metadata')
    info_parser.set_defaults(handler=handle_info)
    return parser


def handle_train(args: argparse.Namespace) -> int:
    """Train the model on synthetic or external data and write artifacts."""
    artifacts_dir = Path('artifacts')
    ensure_directory(artifacts_dir)

    data = load_or_generate_data(
        csv_path=args.train_csv,
        sqlite_path=args.train_sqlite,
        sqlite_query=args.sqlite_query,
        n_samples=args.samples,
    )

    bundle = train_model_bundle(
        data=data,
        output_path=artifacts_dir / 'model_bundle.joblib',
        artifacts_dir=artifacts_dir,
    )

    if not args.quiet:
        print(f'Trained model bundle at {artifacts_dir / "model_bundle.joblib"}')
        print(f'Metrics: {bundle["metrics"]}')
    return 0


def handle_predict(args: argparse.Namespace) -> int:
    """Predict a single batch using the saved model bundle."""
    bundle = load_model_bundle(Path('artifacts/model_bundle.joblib'))
    model = bundle['model']
    features = pd.DataFrame(
        [[args.temperature, args.pressure, args.mash_time]],
        columns=FEATURE_COLUMNS,
    )
    prediction = int(model.predict(features)[0])
    print(prediction)
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    """Print saved metrics from the model bundle."""
    bundle = load_model_bundle(Path('artifacts/model_bundle.joblib'))
    print(bundle['metrics'])
    return 0


def handle_info(args: argparse.Namespace) -> int:
    """Print basic metadata about the saved bundle."""
    bundle = load_model_bundle(Path('artifacts/model_bundle.joblib'))
    print({
        'feature_columns': bundle['feature_columns'],
        'target_column': bundle['target_column'],
        'version': bundle['version'],
    })
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
