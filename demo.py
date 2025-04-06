import mlx_parallm
import importlib
importlib.reload(mlx_parallm)

from mlx_parallm.utils import load, generate, batch_generate
import string
import random
import sys

# --- Configuration ---
MODEL_NAME = "google/gemma-1.1-2b-it"
NUM_PROMPTS = 5
MAX_TOKENS = 50
TEMPERATURE = 0.6 
VERBOSE_TIMING = True

# --- Load Model ---
print(f"Loading model: {MODEL_NAME}...")
# Ensure tokenizer is TokenizerWrapper for detokenizer access
model, tokenizer_wrapper = load(MODEL_NAME)
print("Model loaded.")

# --- Prepare Prompts ---
capital_letters = string.ascii_uppercase
distinct_pairs = [(a, b) for i, a in enumerate(capital_letters) for b in capital_letters[i + 1:]]

# Sample some distinct letter pairs
if len(distinct_pairs) < NUM_PROMPTS:
    print(f"Warning: Requested {NUM_PROMPTS} prompts, but only {len(distinct_pairs)} distinct pairs available.")
    num_actual_prompts = len(distinct_pairs)
else:
    num_actual_prompts = NUM_PROMPTS

sampled_pairs = random.sample(distinct_pairs, num_actual_prompts)

# Create prompts using different templates
prompt_template_1 = "Think of a real word containing both the letters {l1} and {l2}. Then, say 3 sentences which use the word."
prompt_template_2 = "Come up with a real English word containing both the letters {l1} and {l2}. No acronyms. Then, give 3 complete sentences which use the word."

prompts_raw = []
for i, p in enumerate(sampled_pairs):
    if i % 2 == 0:
        prompts_raw.append(prompt_template_1.format(l1=p[0], l2=p[1]))
    else:
        prompts_raw.append(prompt_template_2.format(l1=p[0], l2=p[1]))

print(f"\n--- Starting Generation for {len(prompts_raw)} prompts ---")

# --- Streaming Batch Generation ---
# Store the accumulated responses for each prompt
accumulated_responses = [""] * len(prompts_raw)

# Iterate through the generator provided by batch_generate
# Each `step_segments` is a list [str|None, str|None, ...]
for step_segments in batch_generate(
    model,
    tokenizer_wrapper, # Pass the wrapper
    prompts=prompts_raw,
    max_tokens=MAX_TOKENS,
    verbose=VERBOSE_TIMING, # Pass verbose flag for timing info
    temp=TEMPERATURE,
    # Add other generation parameters if needed (e.g., repetition_penalty=1.1)
):
    # Process the segments for each prompt in the batch
    for i, segment in enumerate(step_segments):
        if segment is not None:
            accumulated_responses[i] += segment
            # Print the segment immediately to show streaming
            # Use end='' and flush=True for real-time effect in terminal
            print(f"[Prompt {i+1}]: {segment}", end='', flush=True)
    # Add a newline after processing a full step across the batch for clarity
    # Only print newline if any segment was printed in this step
    if any(s is not None for s in step_segments):
        print() # Newline after a step yield

# --- Print Final Results ---
print("\n\n--- Final Accumulated Responses ---")
for i, response in enumerate(accumulated_responses):
    print(f"\n--- Prompt {i+1} ---")
    print(prompts_raw[i])
    print(f"--- Response {i+1} ---")
    print(response.strip()) # Strip leading/trailing whitespace
    print("=" * 20)

print("\nDemo finished.")

