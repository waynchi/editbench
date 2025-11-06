from edit_bench.evaluation import generate_files, test_edits
from openai import OpenAI
from os import getenv
from pathlib import Path
import time
import re
import yaml
import os
import sys

"""
An end-to-end experiment script for running code generation experiments using OpenRouter models.
Takes a config YAML file as input specifying the model, prompt file, and other parameters.
"""

# Increase HuggingFace dataset timeout to avoid timeouts in Docker
os.environ['HF_DATASETS_TIMEOUT'] = '60'

# Optional: set HuggingFace cache directory to persist between Docker runs
HF_CACHE_DIR = Path(getenv("WORKDIR"), ".hf_cache")
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(HF_CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = str(HF_CACHE_DIR / "datasets")

# Mapping between the OpenRouter model names and the generation folder names for ease of use
OPENROUTER_NAME_MAP = {
    # DeepSeek models
    "deepseek/deepseek-r1-0528": "deepseek-r1-0528",
    "deepseek/deepseek-chat-v3.1": "deepseek-chat-v3.1",
    
    # Meta Llama models
    "meta-llama/llama-3.1-405b-instruct": "llama-3.1-405b-instruct",
    "meta-llama/llama-4-scout": "llama-4-scout",
    "meta-llama/llama-3.3-8b-instruct:free": "llama-3.3-8b-instruct",  
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b-instruct",  
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b-instruct",  
    "meta-llama/llama-4-maverick": "llama-4-maverick",
    
    # Qwen models
    "qwen/qwen-2.5-72b-instruct": "qwen-2.5-72b-instruct",
    "qwen/qwen3-coder": "qwen3-coder",
    "qwen/qwen-2.5-coder-32b-instruct": "qwen-2.5-coder-32b-instruct",
    "qwen/qwen3-8b": "qwen/qwen3-8b:free",
    "qwen/qwen3-14b": "qwen3-14b", 
    "qwen/qwen3-30b-a3b": "qwen3-30b-a3b",  
    "qwen/qwen3-4b:free": "qwen3-4b", 
    "qwen/qwen3-coder-flash": "qwen3-coder-flash", 

    # Google models
    "google/gemma-3-27b-it": "gemma-3-27b-it",
    "google/gemma-3-12b-it": "gemma-3-12b-it",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemma-3n-e4b-it": "gemma-3n-e4b-it",
    
    # OpenAI models (verified to exist)
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    
    # Mistral models
    "mistralai/mistral-small-3.2-24b-instruct": "mistral-small-3.2-24b-instruct",
    "mistralai/devstral-medium": "devstral-medium",  
    "mistralai/devstral-small": "devstral-small",  
    "mistralai/codestral-2508": "mistralai-codestral-2508", 

    # Anthropic
    "anthropic/claude-3.5-sonnet": "claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet": "claude-3.7-sonnet",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
    
    # Grok
    "x-ai/grok-code-fast-1": "grok-code-fast-1",
    "x-ai/grok-4-fast": "grok-4-fast",  

    # Z-AI
    "z-ai/glm-4.5": "glm-4.5",
    "z-ai/glm-4.6": "glm-4.6",
    
    # Moonshot
    "moonshotai/kimi-k2-0905": "kimi-k2-0905",
    "moonshotai/kimi-dev-72b": "kimi-dev-72b",

}

def parse_code_r1_format(generated_code):
    """
    More specific code to DeepSeek R1 outputs where code blocks may not have proper markdown fences
    Due to the thinking blocks in output
    """
    # Strategy 1: Try standard markdown code fences first (any language)
    if "```" in generated_code:
        # Match any code fence
        match = re.search(r'```[\w/\-+.]*\n(.*?)```', generated_code, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code

    # Strategy 2: Find code after language indicator on its own line or at end of line
    lines = generated_code.split("\n")
    code_start_idx = None

    for language in ["python", "javascript", "jsx", "typescript", "verilog"]:
        for i, line in enumerate(lines):
            # Check if line ends with language name (like "### Solution Codepython")
            if line.strip().endswith(language):
                code_start_idx = i + 1
                break
            # Check if line is just the language name
            if line.strip() == language:
                code_start_idx = i + 1
                break
        if code_start_idx is not None:
            break

    if code_start_idx is not None:
        # Extract code until we hit a markdown heading or end
        code_lines = []
        for i in range(code_start_idx, len(lines)):
            line = lines[i]
            # Stop at markdown headings (### or ##) but not comments
            if line.startswith("###") or line.startswith("##"):
                break
            code_lines.append(line)

        code = "\n".join(code_lines).strip()
        if code:
            return code

    # Strategy 3: Fallback - return everything after the first import/def/class statement
    for keyword in ["import ", "from ", "def ", "class ", "function ", "const ", "let ", "var ", "module "]:
        if keyword in generated_code:
            idx = generated_code.index(keyword)
            # Find the start of the line containing this keyword
            line_start = generated_code.rfind("\n", 0, idx)
            if line_start == -1:
                line_start = 0
            else:
                line_start += 1

            code = generated_code[line_start:].strip()
            # Try to cut off at markdown headings (but not python comments)
            for heading_marker in ["\n### ", "\n## "]:
                if heading_marker in code:
                    code = code.split(heading_marker)[0].strip()
                    break
            if code:
                return code

    # Last resort: return the whole thing stripped
    return generated_code.strip()



def generate_openrouter(model, prompt):
    """
    Query the OpenRouter API with retry logic.
    Retries up to 5 times on any failure with exponential backoff.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=getenv("OPENROUTER_KEY"),
    )
    max_retries = 5
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
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

def make_generator(model):
    """
    Returns a function that generates code using the specified model
    The generate_files function expects a function of the form: fn(prompt, lang) -> str
    """
    def generate(prompt, lang):
        generation = generate_openrouter(model, prompt)
        return parser(generation, lang)
    return generate


if __name__ == "__main__":
    # Load configuration from YAML file
    config_file = sys.argv[1] 
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model = config["model"]
    prompt_file = Path(config["prompt_file"])

    # If a generation path is specified in the config, use it
    if "generate_path" in config:
        gen_path = Path(getenv("WORKDIR"), config["generate_path"])
    else:
        # default generation path convention
        gen_path = Path(getenv("WORKDIR"), "generations", prompt_file.stem, OPENROUTER_NAME_MAP[model])

    gen_path.mkdir(parents=True, exist_ok=True)

    # retry k times in case generations timeout
    for i in range(4):
        generate_files(make_generator(model), prompt_file, gen_path, split=config['split'], max_workers=8)
        time.sleep(2)  # avoid rate limits

    # Test the generated files
    test_edits(gen_path=gen_path, split=config['split'], output_file=f"results/{prompt_file.stem}/{OPENROUTER_NAME_MAP[model]}.json")