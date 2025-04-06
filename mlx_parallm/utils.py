# Copyright © 2023-2024 Apple Inc.

import copy
import glob
import importlib
import json
import logging
import shutil
import time
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, Type

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from mlx.utils import tree_flatten
from transformers import PreTrainedTokenizer

# mlx_lm
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
try:
    from mlx_lm.lora import apply_lora_layers
except ImportError:
    logging.warning("Could not import 'apply_lora_layers' from 'mlx_lm.lora'. LoRA adapter loading might fail.")
    def apply_lora_layers(model, adapter_path):
        raise NotImplementedError("LoRA adapter application function not found in installed mlx_lm version.")

from mlx_lm.tuner.utils import dequantize as dequantize_model

# Local imports
from mlx_parallm.sample_utils import top_p_sampling
from mlx_parallm.models.base import BaseModelArgs, BatchedKVCache

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_REMAPPING = {
    "mistral": "llama",
    "phi-msft": "phixtral",
}
MAX_FILE_SIZE_GB = 5


class ModelNotFoundError(Exception):
    """Custom exception for when a model is not found."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _get_classes(config: dict) -> Tuple[Type[nn.Module], Type[BaseModelArgs]]:
    """Retrieve the model and model args classes based on the configuration."""
    model_type = config.get("model_type")
    if not model_type:
        raise ValueError("Model configuration must contain 'model_type'.")

    model_type = MODEL_REMAPPING.get(model_type, model_type)
    module_name = f"mlx_parallm.models.{model_type}"
    try:
        arch = importlib.import_module(module_name)
        model_class = getattr(arch, "Model", None)
        model_args_class = getattr(arch, "ModelArgs", None)
        if model_class is None or model_args_class is None:
             raise AttributeError(f"Module {module_name} does not contain Model or ModelArgs class.")
    except ImportError:
        msg = f"Model type '{model_type}' not supported or module '{module_name}' not found."
        logging.error(msg)
        raise ValueError(msg) from None
    except AttributeError as e:
        msg = f"Error retrieving classes from module '{module_name}': {e}"
        logging.error(msg)
        raise ValueError(msg) from e

    if not issubclass(model_args_class, BaseModelArgs):
         logging.warning(f"ModelArgs class in {module_name} does not inherit from BaseModelArgs.")

    return model_class, model_args_class


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """Ensures the model is available locally, downloading if necessary."""
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        logging.info(f"Local path '{model_path}' not found. Attempting download from Hugging Face Hub: '{path_or_hf_repo}'")
        try:
            download_path = snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json", "*.safetensors", "*.py", "tokenizer.model",
                    "*.tiktoken", "*.txt", "tokenizer_config.json",
                    "vocab.*", "merges.txt",
                ],
                local_dir_use_symlinks=False,
            )
            model_path = Path(download_path)
            logging.info(f"Model downloaded successfully to: {model_path}")
        except RepositoryNotFoundError:
            raise ModelNotFoundError(
                f"Model not found for path or HF repo: '{path_or_hf_repo}'.\n"
                "1. Check spelling.\n2. Ensure repo exists.\n3. Check authentication for private repos."
            ) from None
        except Exception as e:
             logging.error(f"An unexpected error occurred during download: {e}")
             raise RuntimeError(f"Failed to download model '{path_or_hf_repo}'") from e

    if not model_path.is_dir():
         raise ModelNotFoundError(f"Path '{model_path}' exists but is not a directory.")

    return model_path


def apply_repetition_penalty(
    logits: mx.array, generated_tokens: mx.array, penalty: float
) -> mx.array:
    """Apply repetition penalty to logits for a single sequence."""
    if generated_tokens.size == 0 or penalty == 1.0:
        return logits

    if logits.ndim > 1 and logits.shape[0] != 1:
        raise ValueError("apply_repetition_penalty expects logits for a single sequence.")

    unique_tokens = mx.unique(generated_tokens)
    selected_logits = logits[..., unique_tokens]
    penalized_logits = mx.where(
        selected_logits > 0, selected_logits / penalty, selected_logits * penalty
    )
    logits[..., unique_tokens] = penalized_logits
    return logits


def generate_step(
    prompts: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """Generator producing token IDs based on the prompt batch."""
    if repetition_penalty is not None and repetition_penalty < 0:
        raise ValueError("repetition_penalty must be non-negative.")

    batch_size = prompts.shape[0]

    def sample(logits: mx.array) -> Tuple[mx.array, mx.array]:
        """Samples tokens from logits."""
        if logit_bias:
            bias_indices = mx.array(list(logit_bias.keys()), dtype=mx.int32)
            bias_values = mx.array(list(logit_bias.values()), dtype=logits.dtype)
            logits[:, bias_indices] += bias_values

        softmax_probs = mx.softmax(logits, axis=-1)

        if temp == 0:
            tokens = mx.argmax(logits, axis=-1, keepdims=True)
        else:
            if top_p > 0 and top_p < 1.0:
                tokens = top_p_sampling(logits, top_p, temp)
            else:
                scaled_logits = logits / temp
                tokens = mx.random.categorical(scaled_logits, axis=-1)
                if tokens.ndim == 1:
                   tokens = mx.expand_dims(tokens, axis=-1)

        sampled_probs = mx.take_along_axis(softmax_probs, tokens, axis=-1)
        return tokens, sampled_probs

    # Initialize KV cache
    if hasattr(model, "n_kv_heads"):
        kv_heads = ([model.n_kv_heads] * len(model.layers)
                    if isinstance(model.n_kv_heads, int) else model.n_kv_heads)
    elif hasattr(model, "args") and hasattr(model.args, "num_key_value_heads"):
         kv_heads = [model.args.num_key_value_heads] * len(model.layers)
    else:
         logging.warning("Cannot determine n_kv_heads, assuming num_attention_heads.")
         if hasattr(model, "args") and hasattr(model.args, "num_attention_heads"):
              kv_heads = [model.args.num_attention_heads] * len(model.layers)
         else:
              raise AttributeError("Model lacks required 'n_kv_heads' or 'num_attention_heads'.")

    if not hasattr(model, "head_dim"):
         if hasattr(model, "args") and hasattr(model.args, "head_dim"):
             model_head_dim = model.args.head_dim
         elif hasattr(model, "args") and hasattr(model.args, "hidden_size") and hasattr(model.args, "num_attention_heads"):
             model_head_dim = model.args.hidden_size // model.args.num_attention_heads
             logging.info(f"Calculated head_dim: {model_head_dim}")
         else:
            raise AttributeError("Model lacks required 'head_dim'.")
    else:
        model_head_dim = model.head_dim

    cache = [BatchedKVCache(model_head_dim, n, batch_size) for n in kv_heads]
    generated_sequences = [[token.item() for token in prompt] for prompt in prompts]

    # Initial pre-fill step
    y = prompts
    logits = model(y, cache=cache)[:, -1, :]

    if repetition_penalty is not None:
        for i in range(batch_size):
            context_size = len(generated_sequences[i])
            start_index = max(0, context_size - repetition_context_size) if repetition_context_size is not None else 0
            context_tokens = mx.array(generated_sequences[i][start_index:])
            logits[i:i+1] = apply_repetition_penalty(logits[i:i+1], context_tokens, repetition_penalty)

    y, p = sample(logits)
    for i in range(batch_size):
        generated_sequences[i].append(y[i, 0].item())
    mx.async_eval(y, p)
    yield y, p

    # Generation loop
    while True:
        logits = model(y, cache=cache)[:, -1, :]
        if repetition_penalty is not None:
            for i in range(batch_size):
                context_size = len(generated_sequences[i])
                start_index = max(0, context_size - repetition_context_size) if repetition_context_size is not None else 0
                context_tokens = mx.array(generated_sequences[i][start_index:])
                logits[i:i+1] = apply_repetition_penalty(logits[i:i+1], context_tokens, repetition_penalty)

        next_y, next_p = sample(logits)
        mx.async_eval(next_y, next_p)
        for i in range(batch_size):
            generated_sequences[i].append(next_y[i, 0].item())

        mx.eval(y, p)
        yield y, p
        y, p = next_y, next_p


def stream_generate(
    model: nn.Module,
    tokenizer: TokenizerWrapper, # Expect TokenizerWrapper directly
    prompt: str,
    max_tokens: int = 100,
    **kwargs,
) -> Generator[str, None, None]:
    """Generates text for a single prompt, yielding decoded segments."""
    prompt_tokens = mx.array(tokenizer.encode(prompt))[None]

    generated_token_ids = []
    previous_text = ""
    token_count = 0

    for (tokens_batch, _), _ in zip(
        generate_step(prompt_tokens, model, **kwargs),
        range(max_tokens),
    ):
        token_id = tokens_batch[0, 0].item()
        token_count += 1

        # Stop before adding EOS token to the list for decoding
        if token_id == tokenizer.eos_token_id:
            break

        generated_token_ids.append(token_id)
        # Decode the full sequence generated so far, skipping special tokens
        current_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        # Calculate the new segment
        # Check if current_text starts with previous_text to handle potential decoding variations
        if current_text.startswith(previous_text):
            segment = current_text[len(previous_text):]
        else:
            # Fallback if decoding changed previous parts (less likely with skip_special_tokens=True)
            segment = current_text # Or log a warning
            logging.debug(f"Decoding changed previous text. Previous: '{previous_text}', Current: '{current_text}'")

        previous_text = current_text # Update for next step

        if segment: # Yield if non-empty
            yield segment

    if token_count == 0 and max_tokens > 0:
        logging.warning("stream_generate yielded no tokens.")


def batch_generate(
    model: nn.Module,
    tokenizer: TokenizerWrapper, # Expect TokenizerWrapper directly
    prompts: List[str],
    max_tokens: int = 100,
    verbose: bool = False,
    format_prompts: bool = True,
    **kwargs,
) -> Generator[List[Optional[str]], None, None]:
    """Generate responses for a batch of prompts, yielding results in a streaming manner using manual decoding."""
    batch_size = len(prompts)
    if batch_size == 0:
        return

    if verbose:
        logging.info(f"Starting streaming batch generation for {batch_size} prompts...")

    # Apply chat template
    if format_prompts:
        prompts_formatted = []
        for i, p in enumerate(prompts):
            message = [{"role": "user", "content": p}]
            try:
                formatted_p = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
                prompts_formatted.append(formatted_p)
            except Exception as e:
                logging.warning(f"Failed chat template for prompt {i}: {e}. Using raw prompt.")
                prompts_formatted.append(p)
        prompts_to_encode = prompts_formatted
    else:
        prompts_to_encode = prompts

    # Configure tokenizer for left-padding
    underlying_tokenizer = tokenizer._tokenizer
    original_padding_side = underlying_tokenizer.padding_side
    underlying_tokenizer.padding_side = 'left'
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        if hasattr(underlying_tokenizer, 'pad_token') and underlying_tokenizer.pad_token is None:
            underlying_tokenizer.pad_token = tokenizer.eos_token
        if hasattr(underlying_tokenizer, 'pad_token_id') and underlying_tokenizer.pad_token_id is None:
            underlying_tokenizer.pad_token_id = pad_token_id
        logging.info(f"Using eos_token_id ({pad_token_id}) for padding.")

    # Tokenize batch
    try:
        prompt_encoding = underlying_tokenizer(prompts_to_encode, padding='longest', return_tensors="np", truncation=False)
        prompts_toks = mx.array(prompt_encoding['input_ids'])
    except Exception as e:
        logging.error(f"Error during batch tokenization: {e}")
        underlying_tokenizer.padding_side = original_padding_side
        raise RuntimeError("Failed to tokenize prompts") from e
    underlying_tokenizer.padding_side = original_padding_side # Restore

    tic = time.perf_counter()
    prompt_time = 0.0
    gen_time = 0.0

    # Track full token sequences and previously decoded text for manual streaming
    full_token_ids = [[] for _ in range(batch_size)]
    previous_texts = ["" for _ in range(batch_size)]
    eos_reached = [False] * batch_size

    # --- Generation Loop ---
    step_count = 0
    generation_generator = generate_step(prompts_toks, model, **kwargs)

    for n in range(max_tokens):
        try:
            step_tokens_batch, _ = next(generation_generator)
            step_count = n + 1
            if n == 0:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()

            current_step_segments = [None] * batch_size
            any_active = False

            for i in range(batch_size):
                if not eos_reached[i]:
                    any_active = True
                    token_id = step_tokens_batch[i, 0].item()

                    # Stop before adding EOS token to list for decoding
                    if token_id == tokenizer.eos_token_id:
                        eos_reached[i] = True
                        # Decode final segment if needed (text since last yield)
                        current_text = tokenizer.decode(full_token_ids[i], skip_special_tokens=True)
                        segment = current_text[len(previous_texts[i]):]
                        previous_texts[i] = current_text # Update previous text even on EOS
                        if segment:
                            current_step_segments[i] = segment
                    else:
                        full_token_ids[i].append(token_id) # Add new token ID
                        # Decode full sequence and get new segment, skipping special tokens
                        current_text = tokenizer.decode(full_token_ids[i], skip_special_tokens=True)

                        # Calculate segment, checking prefix for safety
                        if current_text.startswith(previous_texts[i]):
                             segment = current_text[len(previous_texts[i]):]
                        else:
                             segment = current_text # Fallback
                             logging.debug(f"Decoding changed previous text for batch item {i}.")

                        previous_texts[i] = current_text # Update previous text
                        if segment:
                            current_step_segments[i] = segment

            yield current_step_segments

            if not any_active:
                logging.info(f"All sequences reached EOS after {step_count} steps.")
                break
        except StopIteration:
             logging.warning(f"Generation stopped unexpectedly after {step_count} steps.")
             break
    else: # max_tokens reached
        logging.info(f"Generation stopped after reaching max_tokens ({max_tokens}).")

    gen_time = time.perf_counter() - tic

    # --- Finalization Step (Ensure everything decoded) ---
    # The loop above should handle the text up to the last token before EOS or max_tokens
    # No explicit finalize needed for this manual decoding approach

    # --- Verbose Output ---
    if verbose:
        total_prompt_tokens = prompts_toks.size
        total_generated_tokens = sum(len(seq) for seq in full_token_ids) # More accurate count now
        prompt_tps = total_prompt_tokens / prompt_time if prompt_time > 0 else 0
        gen_tps = total_generated_tokens / gen_time if gen_time > 0 else 0

        logging.info("-" * 10)
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Max tokens: {max_tokens}")
        logging.info(f"Steps taken: {step_count}")
        logging.info(f"Prompt processing time: {prompt_time:.3f} s")
        logging.info(f"Generation time: {gen_time:.3f} s")
        logging.info(f"Prompt TPS: {prompt_tps:.2f} tokens/sec")
        logging.info(f"Generation TPS: {gen_tps:.2f} tokens/sec") # Now based on actual token count
        logging.info(f"Total generated tokens: {total_generated_tokens}")
        logging.info("-" * 10)


def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Optional[Callable[[str, float], None]] = None,
    **kwargs,
) -> str:
    """Generate a complete response string using stream_generate internally."""
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if verbose:
        print("=" * 10)
        print("Prompt:", prompt)
        if formatter is None:
            print("Response: ", end="", flush=True)

    tic = time.perf_counter()
    response_pieces = []
    segment_count = 0

    try:
        for token_segment in stream_generate(model, tokenizer, prompt, max_tokens, **kwargs):
            response_pieces.append(token_segment)
            segment_count += 1
            if verbose:
                if formatter:
                     formatter(token_segment, -1.0) # Placeholder prob
                else:
                     print(token_segment, end="", flush=True)
    except Exception as e:
         logging.error(f"Error during generation stream: {e}")

    gen_time = time.perf_counter() - tic
    final_text = "".join(response_pieces)

    if verbose:
        if formatter is None: print() # Newline
        print("\n" + "=" * 10)
        if segment_count == 0: print("No tokens generated.")
        else:
            gen_tps = segment_count / gen_time if gen_time > 0 else 0
            print(f"Generation time: {gen_time:.3f} s")
            print(f"Generation TPS (segments/sec): {gen_tps:.2f}")

    return final_text


def load_config(model_path: Path) -> dict:
    """Loads the model configuration file (config.json)."""
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found in {model_path}")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from config file: {config_path}. Error: {e}")
        raise ValueError(f"Invalid JSON in config file: {config_path}") from e
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    model_config_overrides: dict = {},
) -> nn.Module:
    """Load and initialize the MLX model from a local path."""
    config = load_config(model_path)
    config.update(model_config_overrides)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files: weight_files = glob.glob(str(model_path / "weight*.safetensors"))
    if not weight_files:
        single_model_file = model_path / "model.safetensors"
        single_weights_file = model_path / "weights.safetensors"
        if single_model_file.is_file(): weight_files = [str(single_model_file)]
        elif single_weights_file.is_file(): weight_files = [str(single_weights_file)]
        else:
            index_file = model_path / "model.safetensors.index.json"
            if index_file.is_file():
                 logging.warning(f"Index file found ({index_file}), but no shards detected.")
            raise FileNotFoundError(f"No weight files (*.safetensors) found in {model_path}")

    logging.info(f"Loading weights from: {weight_files}")
    weights = {}
    for wf_path in weight_files:
        try:
            weights.update(mx.load(wf_path))
        except Exception as e:
             logging.error(f"Error loading weights from {wf_path}: {e}")
             raise RuntimeError(f"Failed to load weights from {wf_path}") from e

    model_class, model_args_class = _get_classes(config=config)

    try:
        model_args = model_args_class.from_dict(config)
        model = model_class(model_args)
    except Exception as e:
        logging.error(f"Error initializing model {model_class.__name__}: {e}")
        raise ValueError("Failed to initialize model from config") from e

    if hasattr(model, "sanitize"): weights = model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:
        logging.info(f"Applying quantization config: {quantization}")
        def class_predicate(p, m): return hasattr(m, "to_quantized") and f"{p}.scales" in weights
        try:
            nn.quantize(model, class_predicate=class_predicate, **quantization)
        except Exception as e:
             logging.error(f"Failed during quantization: {e}")

    try:
        model.load_weights(list(weights.items()))
    except Exception as e:
        logging.error(f"Error loading weights into model structure: {e}")
        raise RuntimeError("Failed to load weights into model structure") from e

    if not lazy:
        logging.info("Evaluating model parameters...")
        try:
            mx.eval(model.parameters())
            logging.info("Parameters evaluated.")
        except Exception as e:
            logging.error(f"Error during mx.eval: {e}")

    model.eval()
    return model


def load(
    path_or_hf_repo: str,
    model_config_overrides: dict = {},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """Load model and tokenizer, optionally applying LoRA adapters."""
    logging.info(f"Loading model and tokenizer from: {path_or_hf_repo}")
    model_path = get_model_path(path_or_hf_repo)
    logging.info(f"Base model files located at: {model_path}")

    model = load_model(model_path, lazy, model_config_overrides)

    if adapter_path is not None:
        adapter_file = Path(adapter_path)
        if not adapter_file.is_file():
             raise FileNotFoundError(f"LoRA adapter file not found: {adapter_path}")
        logging.info(f"Applying LoRA adapters from: {adapter_path}")
        try:
            model = apply_lora_layers(model, str(adapter_file))
            model.eval()
            if not lazy:
                 logging.info("Evaluating parameters after applying LoRA...")
                 mx.eval(model.parameters())
                 logging.info("Parameters evaluated.")
        except NotImplementedError as e:
             logging.error(f"LoRA adapter application failed: {e}")
             raise ValueError("LoRA function not available in installed mlx_lm.") from e
        except Exception as e:
             logging.error(f"Failed to apply LoRA adapters: {e}")
             raise ValueError(f"Failed to apply LoRA adapters from {adapter_path}") from e

    try:
        tokenizer = load_tokenizer(model_path) # Load wrapper
    except Exception as e:
         logging.error(f"Failed to load tokenizer from {model_path}: {e}")
         raise ValueError(f"Failed to load tokenizer from {model_path}") from e

    logging.info("Model and tokenizer loaded successfully.")
    return model, tokenizer


# --- Utility functions for saving/uploading ---

def fetch_from_hub(
    model_path: Path, lazy: bool = False
) -> Tuple[nn.Module, dict, TokenizerWrapper]:
    """Fetches model, config, and tokenizer from a local path."""
    if not model_path.is_dir():
        raise FileNotFoundError(f"Provided model path is not a directory: {model_path}")
    model = load_model(model_path, lazy)
    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path)
    return model, config, tokenizer


def make_shards(weights: Dict[str, mx.array], max_file_size_gb: int = MAX_FILE_SIZE_GB) -> List[Dict[str, mx.array]]:
    """Splits model weights into shards based on size."""
    max_bytes = max_file_size_gb << 30
    shards = []
    current_shard, current_size = {}, 0
    for k, v in sorted(weights.items()):
        v_size = v.nbytes
        if current_size + v_size > max_bytes and current_size > 0:
            shards.append(current_shard)
            current_shard, current_size = {}, 0
        current_shard[k] = v
        current_size += v_size
    if current_shard: shards.append(current_shard)
    if len(shards) > 1: logging.info(f"Split weights into {len(shards)} shards.")
    return shards


def upload_to_hub(path: str, upload_repo: str, hf_path: Optional[str] = None):
    """Uploads model files from local path to Hugging Face Hub."""
    import os
    from huggingface_hub import ModelCard, logging as hf_logging

    local_path = Path(path)
    if not local_path.is_dir():
        raise FileNotFoundError(f"Local path '{path}' not found or not a directory.")

    readme_path = local_path / "README.md"
    if hf_path and not readme_path.exists():
         logging.info("Generating basic README.md for Hub.")
         try: card = ModelCard.load(hf_path)
         except Exception:
             logging.warning(f"Could not load card from '{hf_path}'. Creating generic card.")
             card = ModelCard.from_template(card_data={"library_name": "mlx", "tags": ["mlx"]}, model_id=upload_repo)
         card.data.tags = list(set((card.data.tags or []) + ["mlx"]))

         try: from mlx_lm import __version__ as mlx_lm_version
         except ImportError: mlx_lm_version = "[unknown]"

         card_content = dedent(f"""
             ---
             library_name: mlx
             tags: [- mlx]
             ---
             # {upload_repo}
             MLX format model converted{f' from [{hf_path}](https://huggingface.co/{hf_path})' if hf_path else ''} using [mlx-lm](https://github.com/ml-explore/mlx-lm) v{mlx_lm_version}.
             ## Usage with `mlx-lm`
             ```bash
             pip install mlx-lm
             ```python
             from mlx_lm import load, generate
             model, tokenizer = load("{upload_repo}")
             response = generate(model, tokenizer, prompt="hello", verbose=True)
             ```""")
         card.text = card_content + ("\n" + card.text if hasattr(card, 'text') and card.text else "")
         try: card.save(readme_path)
         except Exception as e: logging.warning(f"Could not save generated README.md: {e}")

    hf_logging.set_verbosity_info()
    api = HfApi()
    try:
        logging.info(f"Creating/checking Hub repository: '{upload_repo}'")
        repo_url_obj = api.create_repo(repo_id=upload_repo, repo_type="model", exist_ok=True)
        logging.info(f"Uploading files from '{local_path}' to '{repo_url_obj.repo_id}'...")
        api.upload_folder(
            folder_path=str(local_path), repo_id=upload_repo, repo_type="model",
            commit_message="Upload MLX model", multi_commits=True, multi_commits_verbose=True,
        )
        logging.info(f"Upload complete: [https://huggingface.co/](https://huggingface.co/){upload_repo}")
    except Exception as e:
        logging.error(f"Failed to upload to Hub repository '{upload_repo}': {e}")
        raise RuntimeError("Hub upload failed") from e


def save_weights(
    save_path: Union[str, Path], weights: Dict[str, mx.array], *, donate_weights: bool = False
) -> None:
    """Saves model weights, potentially sharding them."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_format = "model-{:05d}-of-{:05d}.safetensors" if shards_count > 1 else "model.safetensors"
    index_data = {"metadata": {"total_size": sum(v.nbytes for v in weights.values())}, "weight_map": {}}

    logging.info(f"Saving {shards_count} weight shard(s) to {save_path}...")
    for i, shard_weights in enumerate(shards):
        shard_name = shard_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name
        try:
            mx.save_safetensors(str(shard_path), shard_weights, metadata={"format": "mlx"})
            for w_name in shard_weights: index_data["weight_map"][w_name] = shard_name
        except Exception as e:
            logging.error(f"Error saving shard {shard_name}: {e}")
            raise RuntimeError(f"Failed to save shard {shard_name}") from e
        if donate_weights: shards[i] = None; del shard_weights

    if shards_count > 1:
        index_path = save_path / "model.safetensors.index.json"
        index_data["weight_map"] = dict(sorted(index_data["weight_map"].items()))
        try:
            with open(index_path, "w") as f: json.dump(index_data, f, indent=4)
            logging.info(f"Saved weight index file: {index_path}")
        except Exception as e: logging.error(f"Error saving weight index file {index_path}: {e}")

    if donate_weights:
        weights.clear()
        import gc; gc.collect()
        logging.debug("Input weight dictionary cleared.")


def quantize_model(
    model: nn.Module, config: dict, q_group_size: int, q_bits: int
) -> Tuple[Dict[str, mx.array], dict]:
    """Applies quantization and returns quantized weights and updated config."""
    logging.info(f"Quantizing model: group_size={q_group_size}, bits={q_bits}")
    quantized_config = copy.deepcopy(config)
    try:
        nn.quantize(model, q_group_size, q_bits)
        quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
        quantized_weights = dict(tree_flatten(model.parameters()))
        logging.info("Quantization successful.")
        return quantized_weights, quantized_config
    except Exception as e:
        logging.error(f"Error during model quantization: {e}")
        raise ValueError("Model quantization failed") from e


def save_config(
    config: dict, config_path: Union[str, Path]
) -> None:
    """Saves the model configuration dictionary to config.json."""
    config_path = Path(config_path)
    config.pop("_name_or_path", None); config.pop("auto_map", None)
    config_sorted = dict(sorted(config.items()))
    try:
        with open(config_path, "w") as f: json.dump(config_sorted, f, indent=4)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Error saving configuration to {config_path}: {e}")
        raise RuntimeError("Failed to save configuration") from e


def convert(
    hf_path: str, mlx_path: str = "mlx_model", quantize: bool = False,
    q_group_size: int = 64, q_bits: int = 4, dtype: str = "float16",
    upload_repo: Optional[str] = None, revision: Optional[str] = None,
    dequantize: bool = False
):
    """Converts HF model to MLX, optionally processes, saves, and uploads."""
    if quantize and dequantize: raise ValueError("Cannot set both quantize and dequantize.")

    logging.info(f"Starting conversion for '{hf_path}'...")
    mlx_path = Path(mlx_path); mlx_path.mkdir(parents=True, exist_ok=True)

    logging.info("[1/5] Fetching config and tokenizer...")
    model_path = get_model_path(hf_path, revision=revision)
    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path)
    try:
        tokenizer.save_pretrained(mlx_path); logging.info(f"Tokenizer saved to {mlx_path}")
    except Exception as e: raise RuntimeError("Could not save tokenizer") from e

    logging.info("[2/5] Loading model architecture...")
    model = load_model(model_path, lazy=True)

    weights = {}
    if dequantize:
        logging.info("[3/5] Dequantizing model...")
        if "quantization" not in config: logging.warning(f"'{hf_path}' config lacks quantization info.")
        try:
            model = load_model(model_path, lazy=False) # Need evaluated params
            model = dequantize_model(model)
            weights = {k: v.astype(mx.float16) for k, v in tree_flatten(model.parameters())}
            config.pop("quantization", None)
            logging.info("Dequantization complete (float16).")
        except Exception as e: raise RuntimeError("Dequantization failed") from e
    elif quantize:
        logging.info("[3/5] Quantizing model...")
        try:
            model = load_model(model_path, lazy=False) # Need evaluated params
            weights, config = quantize_model(model, config, q_group_size, q_bits)
            logging.info("Quantization complete.")
        except Exception as e: raise # Error handled in quantize_model
    else:
        logging.info(f"[3/5] Converting weights to {dtype}...")
        try: target_dtype = getattr(mx, dtype)
        except AttributeError: raise ValueError(f"Invalid dtype: {dtype}")
        mx.eval(model.parameters()) # Ensure loaded
        weights = {k: v.astype(target_dtype) for k, v in tree_flatten(model.parameters())}
        logging.info(f"Weights converted to {dtype}.")

    del model; import gc; gc.collect(); logging.info("Model instance deleted.")

    logging.info("[4/5] Saving MLX weights and config...")
    save_weights(mlx_path, weights, donate_weights=True)
    save_config(config, config_path=mlx_path / "config.json")

    try:
        py_files = glob.glob(str(model_path / "*.py")); copied_files = []
        for file in py_files:
            dest = mlx_path / Path(file).name; shutil.copy2(file, dest); copied_files.append(Path(file).name)
        if copied_files: logging.info(f"Copied Python files: {copied_files}")
    except Exception as e: logging.warning(f"Could not copy Python files: {e}")

    if upload_repo:
        logging.info(f"[5/5] Uploading to Hugging Face Hub: '{upload_repo}'...")
        try: upload_to_hub(str(mlx_path), upload_repo, hf_path=hf_path)
        except Exception as e: logging.warning(f"Upload failed: {e}. Local conversion complete.")
    else: logging.info("[5/5] Skipping Hub upload.")

    logging.info(f"----- Conversion complete: {mlx_path} -----")

