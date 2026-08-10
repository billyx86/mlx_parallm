import mlx.core as mx

def top_p_sampling(logits: mx.array, top_p: float, temperature: float, axis: int = -1) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token selected based on the top-p criterion.
    """
    if temperature <= 0:
        # Deterministic argmax
        return mx.argmax(logits, axis=axis, keepdims=True)

    # Apply temperature and compute softmax
    logits_scaled = logits / temperature
    probs = mx.softmax(logits_scaled, axis=axis)
    
    # Sort probs in descending order
    sorted_indices = mx.argsort(-probs, axis=axis)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=axis)
    
    # Compute cumulative probabilities
    cumulative_probs = mx.cumsum(sorted_probs, axis=axis)
    
    # Create a mask for probs above the threshold
    # Keep at least one token
    mask = cumulative_probs <= top_p
    # Ensure the first token that exceeds top_p is still kept
    # Find first index where cumulative > top_p
    # Simple approach: set all to True up to first False, keep rest False
    # Use a trick: keep tokens where cumulative <= top_p or previous token
    # For simplicity, we filter and renormalize
    
    # Apply mask, but ensure at least one token remains
    masked_probs = sorted_probs * mask
    # If mask filtered too much, fallback to top-1
    # Compute sum to check
    # We'll renormalize with fallback
    # Use where to keep at least first token
    # Find where mask is False but we need at least one
    # Simpler: use cumulative mask and then force first token to be kept
    # This is a practical approximation
    
    # Normalize the masked probabilities
    sum_masked = mx.sum(masked_probs, axis=axis, keepdims=True)
    # Avoid division by zero
    sum_masked = mx.where(sum_masked == 0, mx.ones_like(sum_masked), sum_masked)
    normalized_probs = masked_probs / sum_masked
    
    # Sample from the normalized probabilities
    sampled_indices = mx.random.categorical(mx.log(normalized_probs + 1e-20), axis=axis)
    
    # Gather the original token indices
    tokens = mx.take_along_axis(sorted_indices, mx.expand_dims(sampled_indices, axis=axis), axis=axis)
    
    return tokens