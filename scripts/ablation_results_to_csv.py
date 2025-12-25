#!/usr/bin/env python3
"""
ablation_results_to_csv.py - Convert ablation study results to CSV format

Outputs a CSV where:
- Each row = a model
- Each column = a setting (no_highlight, with_highlight, with_cursor, with_cursor_and_highlight)
- Values = pass_rate and average_test_rate from JSON results
- Missing results = "---"

Usage:
    # Default: only pass_rate for all settings
    python scripts/ablation_results_to_csv.py

    # Both metrics (pass_rate and avg_test_rate)
    python scripts/ablation_results_to_csv.py --metric both

    # Only average test rate metric
    python scripts/ablation_results_to_csv.py --metric avg_test_rate

    # Custom output file
    python scripts/ablation_results_to_csv.py --output my_results.csv

    # Custom results directory
    python scripts/ablation_results_to_csv.py --results-dir path/to/results
"""

import json
import csv
from pathlib import Path
import argparse
from collections import defaultdict


def extract_metrics(json_file):
    """Extract pass_rate and average_test_rate from result JSON"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Check if metrics are already computed (updated-pipeline format)
        if 'pass_rate' in data and 'average_test_rate' in data:
            pass_rate = data['pass_rate']
            avg_test_rate = data['average_test_rate']
        else:
            # Calculate from raw data (main branch format)
            # Filter out non-numeric keys (like "pass_rate", "average_test_rate" if they exist)
            test_scores = []
            for key, value in data.items():
                # Only include numeric test results
                if isinstance(value, (int, float)) and key not in ['pass_rate', 'average_test_rate']:
                    test_scores.append(value)

            if not test_scores:
                return None

            # Calculate metrics
            pass_rate = sum(1 for score in test_scores if score == 1.0) / len(test_scores)
            avg_test_rate = sum(test_scores) / len(test_scores)

        # Return as percentages rounded to 2 decimal places
        return {
            'pass_rate': f"{pass_rate * 100:.2f}%",
            'avg_test_rate': f"{avg_test_rate * 100:.2f}%"
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def main():
    parser = argparse.ArgumentParser(description='Convert ablation study results to CSV')
    parser.add_argument('--output', '-o', default='ablation_results.csv',
                        help='Output CSV file (default: ablation_results.csv)')
    parser.add_argument('--results-dir', '-d', default='results',
                        help='Results directory (default: results)')
    parser.add_argument('--metric', '-m', choices=['pass_rate', 'avg_test_rate', 'both'],
                        default='pass_rate', help='Which metric to display (default: pass_rate)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return

    # Settings (columns) - directory names
    # Order: no_highlight, with_highlight, with_cursor, with_cursor_and_highlight
    prompt_types_map = {
        'no_highlight': 'no_highlight',
        'with_highlight': 'with_highlight',
        'with_cursor': 'with_cursor',
        'with_cursor_and_highlight': 'with_cursor_and_highlight'
    }
    # Use actual directory names as they exist in results/
    prompt_types = ['no_highlight', 'with_highlight', 'with_cursor', 'with_cursor_and_highlight']

    # Find all unique model names across all prompt types
    all_models = set()
    for prompt_type in prompt_types:
        prompt_dir = results_dir / prompt_type
        if prompt_dir.exists():
            for json_file in prompt_dir.glob('*.json'):
                model_name = json_file.stem
                all_models.add(model_name)

    # Sort models alphabetically
    all_models = sorted(all_models)

    # Build data structure
    results = defaultdict(dict)
    for model in all_models:
        for prompt_type in prompt_types:
            json_file = results_dir / prompt_type / f"{model}.json"
            metrics = extract_metrics(json_file)
            results[model][prompt_type] = metrics

    # Write CSV
    with open(args.output, 'w', newline='') as csvfile:
        if args.metric == 'both':
            # Two columns per prompt type (pass_rate and avg_test_rate)
            fieldnames = ['model']
            for pt in prompt_types:
                display_name = prompt_types_map[pt]
                fieldnames.append(f'{display_name}_pass_rate')
                fieldnames.append(f'{display_name}_avg_test_rate')
        else:
            # One column per prompt type
            fieldnames = ['model'] + [prompt_types_map[pt] for pt in prompt_types]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model in all_models:
            row = {'model': model}

            for prompt_type in prompt_types:
                display_name = prompt_types_map[prompt_type]
                metrics = results[model].get(prompt_type)

                if args.metric == 'both':
                    if metrics:
                        row[f'{display_name}_pass_rate'] = metrics['pass_rate']
                        row[f'{display_name}_avg_test_rate'] = metrics['avg_test_rate']
                    else:
                        row[f'{display_name}_pass_rate'] = '---'
                        row[f'{display_name}_avg_test_rate'] = '---'
                elif args.metric == 'pass_rate':
                    row[display_name] = metrics['pass_rate'] if metrics else '---'
                elif args.metric == 'avg_test_rate':
                    row[display_name] = metrics['avg_test_rate'] if metrics else '---'

            writer.writerow(row)

    print(f"✓ CSV written to: {args.output}")
    print(f"  Models: {len(all_models)}")
    print(f"  Prompt types: {', '.join([prompt_types_map[pt] for pt in prompt_types])}")

    # Print summary of missing results
    missing_count = 0
    for model in all_models:
        for prompt_type in prompt_types:
            if not results[model].get(prompt_type):
                missing_count += 1

    if missing_count > 0:
        print(f"  Missing results: {missing_count}/{len(all_models) * len(prompt_types)}")
    else:
        print(f"  ✓ All results found!")


if __name__ == '__main__':
    main()
