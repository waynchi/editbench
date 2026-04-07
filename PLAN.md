# Run EDIT-Bench on 3 New Models

## Context
Run EDIT-Bench on GPT-5.4, Claude Sonnet 4.6, and Gemini 3.1 Flash Lite using **two prompt configs** on the `test` split (108 problems). Each model uses its own API provider directly.

**Prompt configs:**
- **+highlight** = `prompts/whole_file.txt` (shows highlighted section to change)
- **+cursor** = `prompts/cursor_position.txt` (shows highlighted section + cursor position)

Total: 3 models x 2 prompts = **6 experiment runs**.

## Confirmed Model IDs
| User Name | API Model ID | Provider | Script |
|---|---|---|---|
| ChatGPT 5.4 | `gpt-5.4` | OpenAI | `openai_experiment.py` (existing) |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | Anthropic | `anthropic_experiment.py` (new) |
| Gemini 3.1 Flash Lite | `gemini-3.1-flash-lite-preview` | Google AI | `google_experiment.py` (new) |

## Answers to Your Questions

### a) Resume on failure
**YES.** Generation is fully resumable. Each problem is saved as an individual file (`generations/cursor_position/{model}/{problem_id}`). On re-run, `generate_single_file()` in `edit_bench/evaluation.py:27` skips any file that already exists. Just re-run the same command and it picks up where it left off.

### b) Reuse test results when running complete
**YES.** The generation directory path is `generations/{prompt_name}/{model_name}/` — does NOT include the split. So:
1. Run on `test` → generates 108 files in `generations/cursor_position/{model}/`
2. Later run on `complete` → sees 108 files already exist, skips them, only generates the remaining 432
3. Testing re-runs on all 540 questions (reusing those generation files), writes complete results

One caveat: the result JSON (`results/cursor_position/{model}.json`) gets overwritten on the complete run. Back up your test results first if you want to keep them separately.

## Implementation Steps

### 1. Add GPT-5.4 to `examples/openai_experiment.py`
Add `"gpt-5.4": "gpt-5.4"` to `GPT_MAP` dict (line 27).

### 2. Create `examples/google_experiment.py`
Google's Gemini API has an [OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai), so no new dependencies needed:
```python
client = OpenAI(
    api_key=getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
```
Follow the same pattern as `openrouter_experiment.py`: name map, retry logic with exponential backoff, `generate_files`/`test_edits` from `edit_bench.evaluation`.

### 3. Create `examples/anthropic_experiment.py`
Uses the `anthropic` SDK (already a dependency in `pyproject.toml`):
```python
client = anthropic.Anthropic(api_key=getenv("ANTHROPIC_API_KEY"))
response = client.messages.create(model=model, max_tokens=8192, messages=[...])
```
Same pattern: name map, retry logic, `generate_files`/`test_edits`.

### 4. Create 6 config YAML files (3 models x 2 prompts)

**GPT-5.4:**
- `configs/gpt-5.4-highlight.yaml` → `prompt_file: prompts/whole_file.txt`, `model: gpt-5.4`, `split: test`
- `configs/gpt-5.4-cursor.yaml` → `prompt_file: prompts/cursor_position.txt`, `model: gpt-5.4`, `split: test`

**Claude Sonnet 4.6:**
- `configs/claude-sonnet-4.6-highlight.yaml` → `prompt_file: prompts/whole_file.txt`, `model: claude-sonnet-4-6`, `split: test`
- `configs/claude-sonnet-4.6-cursor.yaml` → `prompt_file: prompts/cursor_position.txt`, `model: claude-sonnet-4-6`, `split: test`

**Gemini 3.1 Flash Lite:**
- `configs/gemini-3.1-flash-lite-highlight.yaml` → `prompt_file: prompts/whole_file.txt`, `model: gemini-3.1-flash-lite-preview`, `split: test`
- `configs/gemini-3.1-flash-lite-cursor.yaml` → `prompt_file: prompts/cursor_position.txt`, `model: gemini-3.1-flash-lite-preview`, `split: test`

### 5. API Keys — add to `EditBench.config`
```
GOOGLE_API_KEY="your-google-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
```
`OPENAI_API_KEY` already exists. `run_experiment.sh` automatically reads ALL variables from `EditBench.config` and passes them to Docker — no script changes needed.

## Running the Experiments (6 runs)
```bash
# +highlight runs (whole_file prompt)
bash run_experiment.sh examples/openai_experiment.py configs/gpt-5.4-highlight.yaml
bash run_experiment.sh examples/anthropic_experiment.py configs/claude-sonnet-4.6-highlight.yaml
bash run_experiment.sh examples/google_experiment.py configs/gemini-3.1-flash-lite-highlight.yaml

# +cursor runs (cursor_position prompt)
bash run_experiment.sh examples/openai_experiment.py configs/gpt-5.4-cursor.yaml
bash run_experiment.sh examples/anthropic_experiment.py configs/claude-sonnet-4.6-cursor.yaml
bash run_experiment.sh examples/google_experiment.py configs/gemini-3.1-flash-lite-cursor.yaml
```

## Output
- Generations: `generations/{whole_file,cursor_position}/{model}/`
- Results: `results/{whole_file,cursor_position}/{model}.json`
- Compare highlight: `python3 scripts/display_results_csv.py results/whole_file`
- Compare cursor: `python3 scripts/display_results_csv.py results/cursor_position`

## Verification
1. Check generation count per model/prompt: `ls generations/{whole_file,cursor_position}/{model}/ | wc -l` → should be 108 each
2. Check each result JSON for `pass_rate` and `average_test_rate`

## Files to Create/Modify
| Action | File |
|---|---|
| Modify | `examples/openai_experiment.py` — add `gpt-5.4` to `GPT_MAP` |
| Modify | `EditBench.config` — add `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY` |
| Create | `examples/anthropic_experiment.py` |
| Create | `examples/google_experiment.py` |
| Create | `configs/gpt-5.4-highlight.yaml` |
| Create | `configs/gpt-5.4-cursor.yaml` |
| Create | `configs/claude-sonnet-4.6-highlight.yaml` |
| Create | `configs/claude-sonnet-4.6-cursor.yaml` |
| Create | `configs/gemini-3.1-flash-lite-highlight.yaml` |
| Create | `configs/gemini-3.1-flash-lite-cursor.yaml` |
