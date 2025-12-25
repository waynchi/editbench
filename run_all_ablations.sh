#!/bin/bash

# Script to run tests for all ablation settings and models
# This script calls run_experiment.sh for each combination
#
# Usage:
#   ./run_all_ablations.sh                          # Run all settings and models
#   ./run_all_ablations.sh --setting no_highlight   # Run only one setting
#   ./run_all_ablations.sh --model claude-sonnet-4  # Run one model across all settings
#   SPLIT=complete ./run_all_ablations.sh           # Use complete split instead of test

set -e

# Configuration
SPLIT="${SPLIT:-test}"
SPECIFIC_SETTING=""
SPECIFIC_MODEL=""
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --setting)
            SPECIFIC_SETTING="$2"
            shift 2
            ;;
        --model)
            SPECIFIC_MODEL="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --setting SETTING    Run only a specific setting"
            echo "  --model MODEL        Run only a specific model"
            echo "  --split SPLIT        Dataset split (default: test)"
            echo "  --help, -h           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --setting no_highlight"
            echo "  $0 --model claude-sonnet-4"
            echo "  $0 --setting with_cursor --model gemini-2.5-pro"
            echo "  SPLIT=complete $0"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}EDITBench Ablation Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Split: ${YELLOW}${SPLIT}${NC}"
[ -n "$SPECIFIC_SETTING" ] && echo -e "Setting filter: ${YELLOW}${SPECIFIC_SETTING}${NC}"
[ -n "$SPECIFIC_MODEL" ] && echo -e "Model filter: ${YELLOW}${SPECIFIC_MODEL}${NC}"
echo ""

# Check if generations directory exists
GENERATIONS_DIR="${PROJECT_DIR}/generations"
if [ ! -d "$GENERATIONS_DIR" ]; then
    echo -e "${RED}Error: generations/ directory not found${NC}"
    exit 1
fi

# Collect all test configurations
declare -a configs
total_tests=0

for setting_dir in "$GENERATIONS_DIR"/*; do
    [ ! -d "$setting_dir" ] && continue

    setting=$(basename "$setting_dir")

    # Filter by setting if specified
    [ -n "$SPECIFIC_SETTING" ] && [ "$setting" != "$SPECIFIC_SETTING" ] && continue

    for model_dir in "$setting_dir"/*; do
        [ ! -d "$model_dir" ] && continue

        model=$(basename "$model_dir")

        # Filter by model if specified
        [ -n "$SPECIFIC_MODEL" ] && [ "$model" != "$SPECIFIC_MODEL" ] && continue

        configs+=("${setting}|${model}")
        ((total_tests++))
    done
done

if [ $total_tests -eq 0 ]; then
    echo -e "${RED}No configurations found matching criteria${NC}"
    exit 1
fi

echo -e "${GREEN}Found ${total_tests} configurations to test${NC}"
echo ""

# Run tests for each configuration
completed=0
failed=0

for config in "${configs[@]}"; do
    IFS='|' read -r setting model <<< "$config"
    ((completed++))

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[${completed}/${total_tests}] ${setting}/${model}${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Check if results already exist
    result_file="${PROJECT_DIR}/results/${setting}/${model}.json"
    if [ -f "$result_file" ]; then
        echo -e "${YELLOW}Results file exists, will check if complete...${NC}"
    fi

    # Run the test via run_experiment.sh
    if bash "${PROJECT_DIR}/run_experiment.sh" \
        examples/run_ablation_tests.py \
        "$setting" \
        "$model" \
        --split "$SPLIT"; then
        echo -e "${GREEN}✓ Completed ${setting}/${model}${NC}"
    else
        echo -e "${RED}✗ Failed ${setting}/${model}${NC}"
        ((failed++))
    fi

    echo ""
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total: ${YELLOW}${total_tests}${NC}"
echo -e "Successful: ${GREEN}$((total_tests - failed))${NC}"
[ $failed -gt 0 ] && echo -e "Failed: ${RED}${failed}${NC}"
echo ""
echo -e "Results directory: ${YELLOW}${PROJECT_DIR}/results/${NC}"
echo ""

# Show how to view results
if [ -z "$SPECIFIC_SETTING" ]; then
    echo -e "${BLUE}View results for each setting:${NC}"
    for setting_dir in "$GENERATIONS_DIR"/*; do
        [ ! -d "$setting_dir" ] && continue
        setting=$(basename "$setting_dir")
        results_dir="${PROJECT_DIR}/results/${setting}"
        [ -d "$results_dir" ] && echo -e "  ${YELLOW}python scripts/display_results_csv.py results/${setting}${NC}"
    done
else
    echo -e "${BLUE}View results:${NC}"
    echo -e "  ${YELLOW}python scripts/display_results_csv.py results/${SPECIFIC_SETTING}${NC}"
    echo -e "  ${YELLOW}python scripts/display_results_csv.py results/${SPECIFIC_SETTING} --csv${NC}"
fi
echo ""

[ $failed -gt 0 ] && exit 1
exit 0
