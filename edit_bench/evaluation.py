import json
import subprocess
import threading
import time
import shutil
import sys
import re

import traceback
from os import getenv
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from pathlib import Path
from tqdm import tqdm

# path inside the docker container
TEST_DIR = Path("/root/editbench_sandboxes")


def generate_single_file(generation_function, prompt_template, question, output_dir):
    """Generate a single file for a given question"""
    id = question["problem_id"]
    file_name = output_dir / str(id)
    
    # Skip if file already exists
    if file_name.exists():
        return {"status": "skipped", "problem_id": id}
    
    prompt = prompt_template.format(
        original_code=question["original_code"],
        highlighted_code=question["highlighted_code"],
        instruction=question["instruction"],
        lang=question["programming_language"],
        cursor_pos=question["cursor_position"]
    )
    
    try:
        generated_code = generation_function(prompt, question["programming_language"])
        
        # Write file based on programming language
        if question["programming_language"] in ["python", "javascript", "javascript/react"]:
            with open(file_name, "w") as f:
                f.write(generated_code)
            return {"status": "success", "problem_id": id}
        else:
            return {"status": "unsupported_language", "problem_id": id}
            
    except Exception as e:
        return {"status": "error", "problem_id": id, "error": str(e)}


def generate_files(generation_function, prompt_file, gen_path, split, n_samples=None, js_only=False, max_workers=8):
    """Generate files in parallel using ThreadPoolExecutor"""
    
    data = load_dataset("copilot-arena/editbench", split=split)

    if n_samples:
        data = data.select(range(n_samples))

    with open(prompt_file, "r") as f:
        prompt_template = f.read()
    
    futures_dict = {}
    errored_out = []
    err_reason = []
    skipped = []
    successful = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs to the executor
        for question in tqdm(data, desc="Creating generation threads"):
            # Skip Python files if js_only is True
            if js_only and question["programming_language"] == "python":
                continue
                
            future = executor.submit(
                generate_single_file,
                generation_function,
                prompt_template,
                question,
                gen_path
            )
            futures_dict[future] = question["problem_id"]
        
        # Process results as they complete
        for future in tqdm(
            as_completed(futures_dict),
            total=len(futures_dict),
            desc="Generating files",
            unit="file",
        ):
            problem_id = futures_dict[future]
            try:
                result = future.result()
                if result["status"] == "error":
                    errored_out.append(problem_id)
                    err_reason.append(result.get("error", "CANT GET REASON"))
                elif result["status"] == "skipped":
                    skipped.append(problem_id)
                elif result["status"] == "success":
                    successful.append(problem_id)
            except Exception as exc:
                errored_out.append(problem_id)
                err_reason.append(traceback.format_exc(exc))
    
    # Print summary
    print(f"Generation complete:")
    print(f"  - Successful: {len(successful)}")
    print(f"  - Skipped (already exist): {len(skipped)}")
    if errored_out:
        print(f"  - Errored out: {len(errored_out)}")
        for problem_id in errored_out[:5]:  # Show first 5 errors
            print(f"    - {problem_id}")
        if len(errored_out) > 5:
            print(f"    - ... and {len(errored_out) - 5} more")
        for reason in err_reason[:5]:  # Show first 5 errors
            print(f"    - {reason}")  # Print first 100 chars of error

#########################################################################################


def test_edits(gen_path, split, output_file, js_only=False):
    questions = load_dataset("copilot-arena/editbench", split=split)
    if Path(output_file).exists():
        with open(output_file, "r") as f:
            data = json.load(f)

        if len(data) > 0.9 * len(questions):
            print("Tests already run, skipping...")
            return
    create_question_folders(gen_path, split, js_only=js_only)
    run_tests(output_file, split, js_only=js_only)
    parse_results(output_file)

def create_question_folders(gen_path, split, js_only=False):
    data = load_dataset("copilot-arena/editbench", split=split)

    gen_path = Path(gen_path)
    num_generations = len(list(gen_path.iterdir()))

    for question in tqdm(data, desc="Creating testing sandboxes"):
        if question["programming_language"] == "python" and js_only:
            continue

        curr_dir = TEST_DIR / f"question_{str(question["problem_id"])}"
        curr_dir.mkdir(parents=True, exist_ok=True)
        with open(curr_dir / "instruction.txt", "w") as f:
            f.write(question["instruction"])

        qid_content = gen_path / str(question["problem_id"])
        if not qid_content.exists():
            raise FileNotFoundError(
                f"Generation for {qid_content} does not exist. Please run the generation function first."
            )

        with open(qid_content, "r") as f:
            generated_code = f.read()

        if question["programming_language"] == "python":
            with open(curr_dir / "requirements.txt", "w") as f:
                f.write(question["requirements"])
            with open(curr_dir / "test_code.py", "w") as f:
                f.write(question["test_code"])
            with open(curr_dir / "original_code.py", "w") as f:
                f.write(question["original_code"])
            with open(curr_dir / "implementation1.py", "w") as f:
                f.write(generated_code.split("```python")[-1].split("```")[0].strip())

        elif question["programming_language"] == "javascript":
            with open(curr_dir / "original_code.js", "w") as f:
                f.write(question["original_code"])
            with open(curr_dir / "implementation1.js", "w") as f:
                f.write(generated_code.split("```javascript")[-1].split("```")[0].strip())

            test_folder = curr_dir / "tests"
            test_folder.mkdir(exist_ok=True)
            with open(test_folder / "test_code.test.js", "w") as f:
                f.write(question["test_code"])

        elif question["programming_language"] == "javascript/react":
            with open(curr_dir / "original_code.jsx", "w") as f:
                f.write(question["original_code"])
            with open(curr_dir / "implementation1.jsx", "w") as f:
                f.write(generated_code.split("```javascript")[-1].split("```")[0].strip())

            test_folder = curr_dir / "tests"
            test_folder.mkdir(exist_ok=True)
            with open(test_folder / "test_code.test.js", "w") as f:
                f.write(question["test_code"])
        else:
            print(
                f"Unsupported programming language: {question['programming_language']}"
            )
            continue

        for file_name, file_content in question["test_harness"].items():
            if file_content is None:
                continue
            other_file = curr_dir / file_name
            other_file.parent.mkdir(parents=True, exist_ok=True)
            with open(other_file, "w") as f:
                f.write(file_content)
        
        # hardcoding a missing file
        if question["pair_id"] == "b8451da4-d914-442a-9eb5-6982148c1cab":
            with open(curr_dir / "app.py", "w") as f:
                f.write("fastapp = {}")

def parse_results(output_file):
    dir = TEST_DIR
    results = {}
    model = "1"  # since we just call the test file "implementation1.py"
    for q_dir in dir.glob("*"):
        # id = int(q_dir.name.split("_")[-1])
        id = q_dir.name
        try:
            with open(q_dir / "test_results.json", "r") as f:
                file_data = json.load(f)
                results_dict = file_data["results"][f"implementation{model}"]
                # fields = "passed", "failed", "skipped", "total"
                results[id] = results_dict["passed"] / (
                    results_dict["total"] + results_dict["skipped"]
                )

        except FileNotFoundError as e:
            print(f"No results in {q_dir}")
            results[id] = 0.0
            continue
        except KeyError as e:
            print(f"No results in {q_dir}")
            results[id] = 0.0
            continue

    n_tests = len(results)
    results_floats = [v for k, v in results.items()]
    num_perfect = sum(1 for item in results_floats if item == 1.0)
    average = sum(results_floats) / n_tests
    results["pass_rate"] = num_perfect / n_tests
    results["average_test_rate"] = average
    print("======== Results ========")
    print(f"Number of tests: {n_tests}")
    print(f"{num_perfect} perfect")
    print(f"{num_perfect / n_tests * 100:.2f}% pass rate")

    out_path = Path(output_file).parent
    out_path.mkdir(parents=True, exist_ok=True)
    # print(results)
    # print(results.keys())
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)


def get_python_commands(dir, python_version):
    """Generate commands with customizable arguments"""

    venv_path = str(dir / ".venv/bin/python")
    test_path = str(dir / "test_code.py")
    req_path = str(dir / "requirements.txt")

    setup_venv_cmd = ["uv", "venv", "--python", python_version]
    install_deps_cmd = ["uv", "pip", "install", "--python", venv_path, "-r", req_path]
    run_tests_cmd = [venv_path, "-m", "pytest", test_path, "-v", "-s"]
    remove_venv_cmd = ["rm", "-rf", ".venv"]

    return [setup_venv_cmd, install_deps_cmd, run_tests_cmd, remove_venv_cmd]


def get_javascript_commands(dir):
    install_cmd = ["npm", "install"]
    sleep_cmd = ["sleep", "1"]
    test_cmd = ["npm", "test"]

    return [install_cmd, sleep_cmd, test_cmd]


def run_sandbox_test(dir, lang, python_version, print_output=False, timeout=600):
    """Run tests for a single sandbox"""
    # Log start of test
    test_log_file = dir / "test_execution.log"

    try:
        with open(test_log_file, "w") as log:
            log.write(f"Starting test for {dir}\n")
            log.write(f"Language: {lang}\n")
            log.write(f"Python version: {python_version}\n")
            log.write(f"Timeout: {timeout}s\n")
            log.flush()

            if lang == "python":
                commands = get_python_commands(dir, python_version)
            elif lang == "javascript" or lang == "javascript/react":
                commands = get_javascript_commands(dir)

            log.write(f"Commands to run: {len(commands)}\n")
            log.flush()

            # Run each command in sequence
            command_outputs = []
            for i, command in enumerate(commands):
                log.write(f"Running command {i+1}/{len(commands)}: {' '.join(command)}\n")
                log.flush()

                result = subprocess.run(
                    command,
                    cwd=dir,
                    check=False,  # Don't raise an exception on error
                    stdout=subprocess.PIPE,  # Capture stdout
                    stderr=subprocess.PIPE,  # Capture stderr
                    text=True,  # Return strings rather than bytes
                    timeout=timeout,
                )
                command_outputs.append(result)

                log.write(f"Command {i+1} completed with return code: {result.returncode}\n")
                log.flush()

            with open(dir / "test_stdout.txt", "w") as f:
                for output in command_outputs:
                    f.write(f"=== Command: {' '.join(output.args)} ===\n")
                    f.write(f"=== Command output ===\n{output.stdout}\n")
                    if output.stderr:
                        f.write(f"=== Command error ===\n{output.stderr}\n")

            if print_output:
                print(f"=========== {str(dir)} ===========")
                for output in command_outputs:
                    print("=== Command: ", " ".join(output.args), " ===")
                    print(f"=== Command output ===\n{output.stdout}")
                    if output.stderr:
                        print(f"=== Command error ===\n{output.stderr}")

            log.write(f"Test completed successfully\n")
            return f"Ran tests for {str(dir)}"

    except Exception as e:
        error_msg = f"Error in {dir}: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        # Write error to log file
        try:
            with open(test_log_file, "a") as log:
                log.write(f"EXCEPTION OCCURRED:\n{error_msg}\n")
        except:
            pass

        if "install" in str(e):
            return f"Error installing dependencies in {str(dir)}: {e}"
        else:
            return f"failed running sandbox {str(dir)}: {e}"


def run_tests(output_file, split, max_workers=4, js_only=False):
    """Run tests in parallel using ThreadPoolExecutor"""
    futures_dict = {}
    errored_out = []

    questions = load_dataset("copilot-arena/editbench", split=split)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs to the executor
        for question in tqdm(questions, desc="Creating test threads"):
            dir = TEST_DIR / f"question_{str(question["problem_id"])}"

            if js_only and question["programming_language"] == "python":
                continue

            future = executor.submit(
                run_sandbox_test,
                dir,
                question["programming_language"],
                question["python_version"],
                print_output=False,
                # timeout=630,
                timeout=180,
            )
            futures_dict[future] = dir

        # Process results as they complete
        for future in tqdm(
            as_completed(futures_dict),
            total=len(questions),
            desc="Running tests",
            unit="sandbox",
        ):
            sandbox_id = futures_dict[future]
            try:
                result = future.result()
                # print(result)
            except Exception as exc:
                errored_out.append(sandbox_id)


    if errored_out:
        print(f"Errored out sandboxes: {len(errored_out)}")
        for sandbox in errored_out:
            print(f"  - {sandbox}")
