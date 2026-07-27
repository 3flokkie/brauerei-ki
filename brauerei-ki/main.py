import sys

from brewing.cli import build_parser


def main() -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(['train'] if len(sys.argv) == 1 else None)
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
