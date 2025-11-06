# EditBench

[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/iamwaynechi?style=flat-square&logo=x&label=Wayne%20Chi)](https://twitter.com/iamwaynechi)
[![GitHub](https://img.shields.io/badge/waynchi-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/waynchi)
[![Website](https://img.shields.io/badge/waynechi.com-4285F4?style=flat-square&logo=google-chrome&logoColor=white)](https://www.waynechi.com/)

[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/valeriechen_?style=flat-square&logo=x&label=Valerie%20Chen)](https://twitter.com/valeriechen_)
[![GitHub](https://img.shields.io/badge/valeriechen-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/valeriechen)
[![Website](https://img.shields.io/badge/valeriechen.github.io-4285F4?style=flat-square&logo=google-chrome&logoColor=white)](https://valeriechen.github.io/)

[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/RyanShar01?style=flat-square&logo=x&label=Ryan%20Shar)](https://twitter.com/RyanShar01)
[![GitHub](https://img.shields.io/badge/rShar01-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/rShar01)
[![Website](https://img.shields.io/badge/rShar01.github.io-4285F4?style=flat-square&logo=google-chrome&logoColor=white)](https://rShar01.github.io/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

EditBench is a code editing benchmark built on real code edits from [Copilot Arena](https://github.com/lmarena/copilot-arena).  
The dataset can be found in [HuggingFace](https://huggingface.co/datasets/copilot-arena/EditBench).

## Overview

The EditBench repository provides a simple method for generating code snippets and evaluating them in an isolated Docker container. 



## Benchmark Results

TODO


## Running Experiments

All experiments are executed using the `run_experiment.sh` shell script, which serves as the main command-line interface for the framework. This script handles building docker containers and running experiments inside the container.

All environment variables to be used in the docker container are defined in the `EditBench.config` file.

### Quick Start
We provide a few sample scripts for generating and running experiments.
To run the tests with pre-generated code edits for `gpt-o3-mini`:

```bash
bash run_experiment.sh examples/run_gpt_o3_mini_tests.py
```
You should see the results in `example_results/gpt-o3-mini.json`

To generate gpt-4o-mini code solutions using the `prompts/whole_file.txt` prompt and run the tests in one command, use
```bash
bash run_experiment.sh examples/run_gpt_4o_mini_experiment.py --should_generate
```
The generated code will be stored in "generations/whole_file/gpt-4o-mini" and the results will be in `example_results/gpt-4o-mini.json`

For a complete end-to-end generation and testing script using OpenRouter and OpenAI, see `examples/openrouter_experiment.py` and `examples/openai_experiment.py`. These scripts take a YAML file as the first argument and runs the experiment with the configuration inside the YAML. For example:
```bash
bash run_experiment.sh examples/openai_experiment.py configs/gpt-5-high.yaml
```

### Commands in run_experiment

By default, the bash script will built and run the docker container, then execute the given python file along with the command line arguments inside the docker container. 
```bash
bash ./run_experiment <path to python file> [args for python file]
```

To help with debugging, we provide the `build` and `shell` commands
```
# Force rebuild the Docker container
bash ./run_experiment build

# Create an interactive session (useful for debugging)
bash ./run_experiment shell
```

## Writing Your Own Inference & Testing Script
Experiments run inside Docker containers, and the `edit_bench` package provides convenient functions for running experiments. The docker container is an isolated execution environment and mounts this repo inside the container as `/projects` (can be accessed using the WORKDIR env variable). Edits made in this repo are synced with the repo inside docker.

The two function you need from `edit_bench.evaluation` are:

- **`generate_files`** - Generates code files for the specified model
- **`test_heb`** - Runs tests for the specified model's generations

The end-to-end examples (e.g. `examples/openai_experiment.py`) provide practical uses for these function. The spec for these functions:

`generate_files(fn, prompt_path, generations_path, split)`
- This function loads data from HF and uses `fn` in multiple threads to generate solutions to each problem. The function ignores problem_ids that already exist in `generations_path`
- `fn(prompt, lang)` is a function that takes a prompt string and programming language string and returns the model's generation for that prompt. The lang string makes parsing the output easier
- `prompt_path` is the path to the prompt f-string. See `prompts/` for examples. The f-string has access to variables: lang (programming language), original_code, instruction (user instruction), and highlighted_code
- `generation_path` is the directory for generated outputs. The generations are stored by problem_id name. Set the path prefix to  `/projects` (can be accessed using the WORKDIR env variable) for the generations to persist outside of docker. 
- `split` the set of questions to use from HF

`test_edits(gen_path, split, output_file)`
- This function tests the generations in the `gen_path` directory and returns the results (as json) to `output_file`. The tests will not run if > 90% of results are already present in the output file (strongly indicates that tests were already run).
- `gen_path` where the generations are located
- `split` the HF split to use
- `output_file` the location of outputs. Use `/projects` (can be accessed using the WORKDIR env variable) to ensure results persists between docker runs.


## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the Apache 2.0 License.

## Acknowledgments

- Thanks to all contributors who have helped shape HumanEditBench
- Special thanks to the open source community for continuous support

## Contact

For questions and feedback, please open an issue or feel free to reach out directly!

## Citation

