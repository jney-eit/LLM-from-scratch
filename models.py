import torch
import torch.nn as nn
from torch.nn import functional as F

# Set a manual seed for reproducibility of results.
torch.manual_seed(1337)


class Head(nn.Module):
    """
    Represents a single head of self-attention.
    Self-attention allows the model to weigh the importance of different words in the input sequence
    when processing a particular word.
    """

    def __init__(self, context_length, d_model, head_size, dropout=0.0):
        """
        Initializes the single attention head.
        :param context_length: The maximum length of the input sequences.
        :param d_model: The embedding dimension for each token.
        :param head_size: The inner dimension of the key, query, and value vectors.
        :param dropout: The dropout rate for regularization.
        """
        super().__init__()
        self.head_size = head_size

        # Linear layers to project the input into key, query, and value vectors.
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

        # A triangular mask to prevent positions from attending to subsequent positions.
        # This is essential for a decoder-only (causal) transformer.
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Defines the forward pass for a single self-attention head.
        :param x: Input tensor of shape (batch_size, sequence_length, d_model).
        """
        batch_size, sequence_length, d_model = x.shape

        # Generate key, query, and value vectors from the input.
        k = self.key(x)  # (batch_size, sequence_length, head_size)
        q = self.query(x)  # (batch_size, sequence_length, head_size)
        v = self.value(x)  # (batch_size, sequence_length, head_size)

        # Compute attention scores ("wei").
        # This is the dot product between queries and keys, scaled by the square root of head_size.
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (batch_size, sequence_length, sequence_length)

        # Apply the causal mask to hide future tokens.
        wei = wei.masked_fill(self.tril[:sequence_length, :sequence_length] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (batch_size, sequence_length, sequence_length)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values to get the output.
        out = wei @ v  # (B, seq_len, seq_len) @ (B, seq_len, head_size) -> (B, seq_len, head_size)

        return out


class FeedForward(nn.Module):
    """
    A simple two-layer feed-forward neural network applied after the attention mechanism.
    This allows the model to process the information gathered by the attention heads.
    """

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        # The inner layer is typically 4 times the size of the model dimension.
        ffwd_size_fac = 4
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * ffwd_size_fac),
            nn.ReLU(),
            nn.Linear(ffwd_size_fac * d_model, d_model),  # Projection layer back to d_model
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """
    Combines multiple self-attention heads to allow the model to focus on different
    parts of the input sequence simultaneously.
    """

    def __init__(self, context_length, d_model, head_size, num_heads, dropout=0.0):
        super().__init__()
        # Create a list of attention heads.
        self.heads = nn.ModuleList([Head(context_length, d_model, head_size, dropout) for _ in range(num_heads)])
        # A projection layer to combine the outputs of all heads.
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the outputs from all heads along the last dimension.
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (batch_size, sequence_length, num_heads * head_size)
        # Project the concatenated output back to the original model dimension.
        out = self.proj(out)
        out = self.dropout(out)
        return out


class CombinedMultiHeadAttention(nn.Module):
    """
    An optimized implementation of Multi-Head Attention. Instead of calculating heads separately,
    it performs the key, query, and value projections for all heads in a single matrix multiplication,
    which is more efficient.
    """

    def __init__(self, context_length, d_model, head_size, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model must be divisible by num_heads, d_model: {d_model}, num_heads: {num_heads}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = head_size
        self.context_length = context_length

        # Single linear layers for all heads' queries, keys, and values.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Final projection layer.
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1),
            persistent=False
        )

        # Buffers for the Key-Value (KV) cache, used for fast generation.
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self, x, use_kv_cache=False):
        batch_size, sequence_length, d_model = x.shape

        # Project input to new keys, queries, and values for all heads at once.
        keys_new = self.W_k(x)
        queries_new = self.W_q(x)
        values_new = self.W_v(x)

        if use_kv_cache:
            # If caching is enabled, append new keys and values to the stored cache.
            if self.cache_k is None:  # First pass (e.g., processing a prompt)
                keys = keys_new
                values = values_new
            else:  # Subsequent passes (e.g., generating one token at a time)
                keys = torch.cat((self.cache_k, keys_new), dim=1)
                values = torch.cat((self.cache_v, values_new), dim=1)

                # Discard first keys and values if context length is exceeded
                if keys.shape[1] > self.context_length:
                    keys = keys[:, 1:, :]
                    values = values[:, 1:, :]

            # Update the cache for the next iteration.
            self.cache_k = keys
            self.cache_v = values
            queries = queries_new
        else:
            # Without cache, keys, values, and queries are from the current input.
            keys = keys_new
            values = values_new
            queries = queries_new

        q_len = queries.shape[1]
        kv_len = keys.shape[1]

        # Reshape and transpose to separate heads.
        queries = queries.view(batch_size, q_len, self.num_heads, self.head_size).transpose(1, 2)
        keys = keys.view(batch_size, kv_len, self.num_heads, self.head_size).transpose(1, 2)
        values = values.view(batch_size, kv_len, self.num_heads, self.head_size).transpose(1, 2)

        # Calculate attention scores.
        attn_scores = queries @ keys.transpose(-2, -1)

        #print(f"mask shape: {self.mask.shape}")
        #print(f"q_len: {q_len}")
        #print(f"kv_len: {kv_len}")

        # Apply causal mask.
        if q_len > 1:
            mask_slice = self.mask[:q_len, :kv_len]
            attn_scores = attn_scores.masked_fill(mask_slice, float('-inf'))

        # Apply softmax and scaling.
        attn_weights = torch.softmax(attn_scores / self.head_size ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Get the context vector by multiplying attention weights with values.
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads back into a single tensor.
        context_vec = context_vec.contiguous().view(batch_size, q_len, self.d_model)
        context_vec = self.proj(context_vec)

        return context_vec

    def reset_cache(self):
        """ Clears the KV cache. """
        self.cache_k, self.cache_v = None, None


class TransformerBlock(nn.Module):
    """
    A standard Transformer block, which consists of a self-attention mechanism
    followed by a feed-forward network. Layer normalization and residual connections
    are used to stabilize training.
    """

    def __init__(self, context_length, d_model, num_heads, dropout=0.0):
        super().__init__()
        head_size = d_model // num_heads
        self.sa = CombinedMultiHeadAttention(context_length, d_model, head_size, num_heads, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, use_kv_cache=False):
        # Self-attention with a residual connection (x + ...)
        x = x + self.sa(self.ln1(x), use_kv_cache=use_kv_cache)
        # Feed-forward network with a residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

    def reset_cache(self):
        """ Resets the cache in the self-attention layer. """
        self.sa.reset_cache()


class VanillaGPT(nn.Module):
    """
    A simple decoder-only Language Model (LLM) using multi-head self-attention.
    """

    def __init__(self, vocab_size, context_length, d_model, num_heads, n_layer, dropout=0.0, device='cpu'):
        super().__init__()
        self.device = device

        # Embedding layers for tokens and their positions in the sequence.
        self.token_embedding_layer = nn.Embedding(vocab_size, d_model)
        self.position_embedding_layer = nn.Embedding(context_length, d_model)

        # A stack of Transformer blocks.
        self.blocks = nn.ModuleList(
            [TransformerBlock(context_length, d_model, num_heads, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)  # Final layer normalization

        # The final linear layer that maps the model's output to vocabulary logits.
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.current_pos = 0
        self.context_length = context_length

    def forward(self, idx, use_kv_cache=False):
        batch_size, sequence_length = idx.shape

        # Get token embeddings.
        tok_emb = self.token_embedding_layer(idx)  # (B, T, C)

        # Get position embeddings.
        if use_kv_cache and sequence_length == 1:
            current_wrapped_position = self.current_pos % self.context_length

            # For KV cache, only generate embeddings for new positions.
            pos_ids = torch.tensor([current_wrapped_position], device=self.device, dtype=torch.long)
        else:
            pos_ids = torch.arange(0, sequence_length, device=self.device, dtype=torch.long)

            if self.current_pos >= self.context_length:
                shift_amount = (self.current_pos % self.context_length) + 1

                # Roll pos ids with current position to have absolute positional embedding
                pos_ids = torch.roll(pos_ids, shifts=-shift_amount, dims=0)

        # For first input (prompt) we need to increase by sequence length, then we assume one token is added each time
        if self.current_pos == 0:
            self.current_pos += sequence_length
        else:
            self.current_pos += 1

        # print(f"sequence_length: {sequence_length}")
        # print(f"context_length: {self.context_length}")
        # print(f"pos_ids: {pos_ids}")
        pos_emb = self.position_embedding_layer(pos_ids)  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        # Pass through all Transformer blocks.
        for blk in self.blocks:
            x = blk(x, use_kv_cache=use_kv_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    def reset_kv_cache(self):
        """ Resets the KV cache in all transformer blocks. """
        for blk in self.blocks:
            blk.reset_cache()
        self.current_pos = 0

    def generate(self, running_idx_sequence, context_length, num_tokens_to_generate):
        """
        Generates new tokens autoregressively without using a KV cache.
        This is less efficient as it recomputes the full context at each step.
        :param running_idx_sequence: The initial sequence of tokens (prompt).
        :param context_length: The maximum context length the model can handle.
        :param num_tokens_to_generate: The number of new tokens to generate.
        :return: The extended sequence of tokens.
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            self.reset_kv_cache()
            for _ in range(num_tokens_to_generate):
                # Crop the sequence to the last `context_length` tokens.
                idx_cond = running_idx_sequence[:, -context_length:]
                # Get model predictions (logits).
                logits = self.forward(idx_cond)
                # Focus on the logit for the very last token.
                logits = logits[:, -1, :]
                # Apply softmax to get probabilities.
                probs = F.softmax(logits, dim=-1)
                # Sample the next token from the probability distribution.
                idx_next = torch.multinomial(probs, num_samples=1)
                # Append the new token to the running sequence.
                running_idx_sequence = torch.cat((running_idx_sequence, idx_next), dim=1)
        return running_idx_sequence

    def generate_with_kv_cache(self, running_idx_sequence, context_length, num_tokens_to_generate):
        """
        Generates new tokens autoregressively using the KV cache for efficiency.
        Only the newest token is processed at each step after the initial prompt.
        :param running_idx_sequence: The initial sequence of tokens (prompt).
        :param num_tokens_to_generate: The number of new tokens to generate.
        :return: The extended sequence of tokens.
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            self.reset_kv_cache()
            # First, process the entire prompt to initialize the KV cache.
            logits = self.forward(running_idx_sequence, use_kv_cache=True)

            for _ in range(num_tokens_to_generate):
                # Get logits for the last token.
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                # Append the new token.
                running_idx_sequence = torch.cat((running_idx_sequence, idx_next), dim=1)
                # Feed only the new token to the model; it will use the cache.
                logits = self.forward(idx_next, use_kv_cache=True)
        return running_idx_sequence


if __name__ == "__main__":
    """
    Test Implementation for KV Cache Equivalence

    The purpose of this test is to verify that the standard `generate` function and the
    optimized `generate_with_kv_cache` function produce the exact same output sequence.
    This confirms that the KV cache implementation is mathematically correct.
    """
    print("--- Running KV Cache Generation Equivalence Test ---")

    # 1. Define Hyperparameters for the model and test
    batch_size = 1
    context_length = 128
    d_model = 64
    num_heads = 4
    n_layer = 4
    vocab_size = 50
    dropout = 0.0  # Dropout must be 0 for a deterministic comparison
    device = 'cpu'

    # Generation parameters
    prompt_length = 10
    num_tokens_to_generate = 256        #20

    # Use a fixed seed for reproducibility of the multinomial sampling
    test_seed = 1337

    print(f"Test Parameters:\n  - Prompt Length: {prompt_length}\n  - New Tokens: {num_tokens_to_generate}\n")

    # 2. Instantiate the GPT model
    model = VanillaGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        n_layer=n_layer,
        dropout=dropout,
        device=device
    ).to(device)

    # 3. Create a random starting prompt
    prompt = torch.randint(0, vocab_size, (batch_size, prompt_length), device=device)

    # --- Run Generation without KV Cache ---
    print("1. Generating tokens without KV cache...")
    torch.manual_seed(test_seed)
    output_no_cache = model.generate(
        prompt.clone(),  # Use clone to ensure the original prompt is not modified
        context_length,
        num_tokens_to_generate
    )
    print("   Done.")

    # --- Run Generation with KV Cache ---
    print("2. Generating tokens with KV cache...")
    torch.manual_seed(test_seed)  # Reset seed to the same value for identical sampling
    output_with_cache = model.generate_with_kv_cache(
        prompt.clone(),
        context_length,
        num_tokens_to_generate
    )
    print("   Done.\n")

    # 4. Compare the outputs
    are_outputs_equal = torch.equal(output_no_cache, output_with_cache)

    print(f"Output shape from standard generate: {output_no_cache.shape}")
    print(f"Output shape from cached generate:  {output_with_cache.shape}\n")

    # 5. Report the result
    if are_outputs_equal:
        print("✅ SUCCESS: The outputs of `generate` and `generate_with_kv_cache` are identical.")
        print("This confirms that the KV cache implementation is correct.")
    else:
        print("❌ FAILURE: The generated token sequences are different.")
        print("\nOutput (no cache):")
        print(output_no_cache[0])
        print("\nOutput (with cache):")
        print(output_with_cache[0])
        # Find the first point of difference
        diff_mask = output_no_cache != output_with_cache
        if diff_mask.any():
            first_diff_idx = diff_mask.nonzero(as_tuple=True)[1][0].item()
            print(f"\nSequences first differ at index {first_diff_idx}.")

    print("\n--- Test Complete ---")