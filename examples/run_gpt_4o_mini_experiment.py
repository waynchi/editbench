from edit_bench.evaluation import generate_files, test_edits
from os import getenv
from pathlib import Path
from openai import OpenAI
import time
import argparse

def gpt_4o_mini_gen_function(prompt, lang):
    client = OpenAI(
        api_key=getenv("OPENAI_API_KEY"),
    )
    max_retries = 5
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                    "role": "user",
                    "content": prompt
                    }
                ]
            )
            generation =  completion.choices[0].message.content
            return generation.split(f"```{lang}")[-1].split("```")[0].strip()
            
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--should_generate", action="store_true", help="Whether to generate new files or just test existing ones")
    args = parser.parse_args()
    # Path to the generations 
    gpt_4o_mini_gen_path = Path(getenv("WORKDIR"), "generations", "whole_file", "gpt-4o-mini")

    if args.should_generate:
        generate_files(gpt_4o_mini_gen_function, "prompts/whole_file.txt", gpt_4o_mini_gen_path, split="test", max_workers=32)

    # Use our testing function to run tests on the generated files
    test_edits(gen_path=gpt_4o_mini_gen_path, split="test", output_file=f"example_results/gpt-4o-mini.json")