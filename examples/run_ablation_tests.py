#!/usr/bin/env python3
"""
Run tests for a specific ablation setting and model.

Usage (via run_experiment.sh):
    bash run_experiment.sh examples/run_ablation_tests.py <setting> <model> [--split <split>]

Examples:
    bash run_experiment.sh examples/run_ablation_tests.py no_highlight claude-sonnet-4
    bash run_experiment.sh examples/run_ablation_tests.py with_cursor gemini-2.5-pro --split test
"""

import argparse
from pathlib import Path
from os import getenv
from edit_bench.evaluation import test_edits

def main():
    parser = argparse.ArgumentParser(
        description='Run tests for a specific ablation setting and model'
    )
    parser.add_argument(
        'setting',
        type=str,
        help='Setting name (e.g., no_highlight, with_cursor, with_highlight, with_cursor_and_highlight)'
    )
    parser.add_argument(
        'model',
        type=str,
        help='Model name (e.g., claude-sonnet-4, gemini-2.5-pro, etc.)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'complete'],
        help='Dataset split to use (default: test)'
    )

    args = parser.parse_args()

    # Get WORKDIR from environment (set by run_experiment.sh)
    workdir = Path(getenv("WORKDIR", "/project"))

    # Construct paths
    gen_path = workdir / "generations" / args.setting / args.model
    output_file = workdir / "results" / args.setting / f"{args.model}.json"

    # Verify generation path exists
    if not gen_path.exists():
        print(f"Error: Generation path does not exist: {gen_path}")
        print(f"Available settings in generations/:")
        generations_dir = workdir / "generations"
        if generations_dir.exists():
            for setting_dir in sorted(generations_dir.iterdir()):
                if setting_dir.is_dir():
                    print(f"  - {setting_dir.name}/")
                    for model_dir in sorted(setting_dir.iterdir()):
                        if model_dir.is_dir():
                            print(f"    - {model_dir.name}/")
        return 1

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Setting: {args.setting}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Generation path: {gen_path}")
    print(f"Output file: {output_file}")
    print()

    # Run tests
    test_edits(gen_path=gen_path, split=args.split, output_file=str(output_file))

    print(f"\n✓ Results saved to: {output_file}")
    return 0

if __name__ == "__main__":
    exit(main())
