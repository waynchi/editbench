from edit_bench.evaluation import test_edits
from os import getenv
from pathlib import Path


# Path to the generations 
o3_generation_path = Path(getenv("WORKDIR"), "generations", "whole_file", "gpt-o3-mini")

# Use our testing function to run tests on the generated files
test_edits(gen_path=o3_generation_path, split="test", output_file=f"example_results/gpt-o3-mini.json")