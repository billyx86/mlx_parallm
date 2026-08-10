"""
Improved demo for mlx_parallm with better streaming display.
"""

from mlx_parallm import load, batch_generate
import sys

MODEL_NAME = "google/gemma-1.1-2b-it"
PROMPTS = [
    "Explain quantum computing in one paragraph.",
    "Write a haiku about rain.",
    "List three benefits of batch inference.",
]

print(f"Loading model: {MODEL_NAME}...")
model, tokenizer = load(MODEL_NAME)
print("Model loaded.\n")

print("Starting batch generation...")
accumulated = ["" for _ in PROMPTS]

for step in batch_generate(model, tokenizer, PROMPTS, max_tokens=64, verbose=True, temp=0.7):
    for i, seg in enumerate(step):
        if seg:
            accumulated[i] += seg
            # Simple streaming display
            sys.stdout.write(seg)
            sys.stdout.flush()
    # Newline per step
    if any(s for s in step if s):
        print()

print("\n\n=== Final Results ===")
for i, (p, r) in enumerate(zip(PROMPTS, accumulated)):
    print(f"\nPrompt {i+1}: {p}")
    print(f"Response: {r.strip()}")
