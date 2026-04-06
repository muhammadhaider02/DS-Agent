""" This file contains the code for calling all LLM APIs. """

import os
from functools import partial
import tiktoken
from dotenv import load_dotenv
from .schema import TooLongPromptError, LLMError

load_dotenv()

enc = tiktoken.get_encoding("cl100k_base")
STATISTICAL_DIR = None
FINETUNE_STEP = 0

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

try:   
    import anthropic
    claude_key = os.getenv("CLAUDE_API_KEY")
    if not claude_key:
        raise ValueError("No API key found. Set CLAUDE_API_KEY in .env")
    anthropic_client = anthropic.Anthropic(api_key=claude_key)
except Exception as e:
    print(e)
    print("Could not load Anthropic API key.")
    
client = None  # Will be set if OpenAI/GitHub key is available
try:
    import openai
    github_token = os.getenv("GITHUB_OPENAI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if github_token:
        github_base_url = os.getenv("GITHUB_OPENAI_API_BASE_URL", "https://models.inference.ai.azure.com")
        client = openai.OpenAI(
            api_key=github_token,
            base_url=github_base_url
        )
    elif openai_key:
        client = openai.OpenAI(api_key=openai_key)
    else:
        raise ValueError("No API key found. Set GITHUB_OPENAI_API_KEY or OPENAI_API_KEY in .env")

except Exception as e:
    print(e)
    print("Could not load OpenAI API key.")

try:
    import openai as _openai
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not deepseek_key:
        raise ValueError("No API key found. Set DEEPSEEK_API_KEY in .env")
    deepseek_client = _openai.OpenAI(
        api_key=deepseek_key,
        base_url=deepseek_base_url
    )
except Exception as e:
    print(e)
    print("Could not load DeepSeek API key.")
    deepseek_client = None


def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """ Log the prompt and completion to a file."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")
    
    finetune_directory = os.path.join(STATISTICAL_DIR, "finetune_log")
    if not os.path.exists(finetune_directory):
        os.mkdir(finetune_directory)
    global FINETUNE_STEP
    FINETUNE_STEP += 1
    with open(os.path.join(finetune_directory, f"step_{FINETUNE_STEP}.txt"), "wt") as f:
        f.write(prompt)
        f.write("\n[This is a split string for finetuning]\n")
        f.write(completion)


def complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], model="claude-v1", max_tokens_to_sample=2000, temperature=0.5, log_file=None, **kwargs):
    """ Call the Claude API to complete a prompt."""
    ai_prompt = anthropic.AI_PROMPT
    if "ai_prompt" in kwargs is not None:
        ai_prompt = kwargs["ai_prompt"]

    try:
        rsp = anthropic_client.completions.create(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {ai_prompt}",
            stop_sequences=stop_sequences,
            model=model,
            temperature=temperature,
            max_tokens_to_sample=max_tokens_to_sample,
            **kwargs
        )
    except anthropic.APIStatusError as e:
        print(e)
        raise TooLongPromptError()
    except Exception as e:
        raise LLMError(e)

    completion = rsp.completion
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def complete_text_openai(prompt, stop_sequences=[], model="gpt-4o-mini", max_tokens_to_sample=1000, temperature=0.5, log_file=None, _client=None, **kwargs):
    """ Call the OpenAI-compatible API to complete a prompt."""
    _client = _client or client
    raw_request = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens_to_sample,
        "stop": stop_sequences or None,
        **kwargs
    }

    iteration = 0
    completion = None
    response = None
    while iteration < 10:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = _client.chat.completions.create(messages=messages, timeout=600, **raw_request)
            completion = response.choices[0].message.content
            break
        except Exception as e:
            iteration += 1
            print(f"===== Retry: {iteration} =====")
            print(f"Error occurs when calling API: {e}")
            continue

    if STATISTICAL_DIR and completion is not None:
        # Route to per-model token file
        model_lower = model.lower()
        if "reasoner" in model_lower:
            token_file = STATISTICAL_DIR + "/count_tokens_deepseek_reasoner.txt"
        elif "deepseek" in model_lower:
            token_file = STATISTICAL_DIR + "/count_tokens_deepseek_chat.txt"
        else:
            token_file = STATISTICAL_DIR + "/count_tokens_other.txt"

        # Use actual API usage counts; fall back to tiktoken if unavailable
        if response is not None and hasattr(response, 'usage') and response.usage is not None:
            num_prompt_tokens = response.usage.prompt_tokens
            num_sample_tokens = response.usage.completion_tokens
        else:
            num_prompt_tokens = len(enc.encode(prompt))
            num_sample_tokens = len(enc.encode(completion))

        # Read existing per-model input/output counts
        try:
            with open(token_file, 'r') as f:
                lines = f.read().strip().splitlines()
                existing_input = int(lines[0].split('=')[1])
                existing_output = int(lines[1].split('=')[1])
        except (FileNotFoundError, IndexError, ValueError):
            existing_input = 0
            existing_output = 0

        # Write updated counts
        with open(token_file, 'w') as f:
            f.write(f"input={existing_input + num_prompt_tokens}\n")
            f.write(f"output={existing_output + num_sample_tokens}\n")

    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def complete_text(prompt, log_file, model, **kwargs):
    """ Complete text using the specified model with appropriate API. """
    if model.startswith("claude"):
        completion = complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT, "Observation:"], log_file=log_file, model=model, **kwargs)
    elif model.lower().startswith("deepseek"):
        completion = complete_text_openai(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, _client=deepseek_client, **kwargs)
    else:
        completion = complete_text_openai(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    return completion


FAST_MODEL = "gpt-4o-mini"
def complete_text_fast(prompt, **kwargs):
    return complete_text(prompt=prompt, model=FAST_MODEL, temperature=0.01, **kwargs)