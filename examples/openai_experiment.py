from edit_bench.evaluation import generate_files, test_edits
from openai import OpenAI
from os import getenv
import time
import sys
import yaml
import os

from pathlib import Path

"""
An end-to-end experiment script for running code generation experiments using OpenAI models.
Takes a config YAML file as input specifying the model, prompt file, and other parameters.
"""

# Increase HuggingFace dataset timeout to avoid timeouts in Docker
os.environ['HF_DATASETS_TIMEOUT'] = '60'

# Set HuggingFace cache directory to persist between Docker runs
HF_CACHE_DIR = Path(getenv("WORKDIR"), ".hf_cache")
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(HF_CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = str(HF_CACHE_DIR / "datasets")


# Map between OpenAI model names and generation folder names for ease of use
GPT_MAP = {
    "o4-mini-2025-04-16": "gpt-o4-mini",
    "o3-mini-2025-01-31": "gpt-o3-mini",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-5-mini-2025-08-07": "gpt-5-mini",
    "gpt-5-2025-08-07": "gpt-5",
    "gpt-5-nano-2025-08-07": "gpt-5-nano"
}

def generate_openai(model, prompt, reasoning=None):
    client = OpenAI(
        api_key=getenv("OPENAI_API_KEY"),
    )
    max_retries = 5
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            if reasoning:
                response = client.responses.create(
                    model=model,
                    reasoning={"effort": reasoning},
                    input=[
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ]
                )
                return response.output_text
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                        "role": "user",
                        "content": prompt
                        }
                    ]
                )
                return completion.choices[0].message.content
            
        except Exception as e:
            last_exception = e
            
            # Don't sleep on the last attempt
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                delay = 2 ** attempt
                time.sleep(delay)
    
    # If we get here, all retries failed
    print(last_exception)
    raise last_exception

def parser(generation, lang):
    """
    Parses the code output using the ```lang markers.
    """
    return generation.split(f"```{lang}")[-1].split("```")[0].strip()

def make_generator(model, reasoning):
    """
    Returns a function that generates code using the specified model + reasoning level
    The generate_files function expects a function of the form: fn(prompt, lang) -> str
    """
    def generate(prompt, lang):
        generation = generate_openai(model, prompt, reasoning)
        return parser(generation, lang)

    return generate


if __name__ == "__main__":
    # Load configuration from YAML file
    config_file = sys.argv[1] 
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model = config["model"]
    reasoning = config.get("reasoning", None)
    prompt_file = Path(config["prompt_file"])

    # To differentiate outputs from the same model under different reasoning levels
    model_name_path = GPT_MAP[model]
    if reasoning:
        model_name_path += "-" + reasoning

    # If a generation path is specified in the config, use it
    if "generate_path" in config:
        gen_path = config["generate_path"]
    else:
        # default generation path convention
        gen_path = Path(getenv("WORKDIR"), "generations", prompt_file.stem, model_name_path)

    gen_path.mkdir(parents=True, exist_ok=True)

    # retry k times in case generations timeout
    for i in range(1):
        generate_files(make_generator(model, reasoning), prompt_file, gen_path, split=config['split'], max_workers=32)
        time.sleep(2)  # avoid rate limits

    # Test the generated files
    test_edits(gen_path=gen_path, split=config['split'], output_file=f"results/{prompt_file.stem}/{model_name_path}.json")