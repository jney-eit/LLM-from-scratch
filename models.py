import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)



class Head(nn.Module):
    """
    Single head of self-attention
    """

    def __init__(self, context_length, d_model, head_size, dropout=0.0):
        """
        :param context_length: length of the context window
        :param d_model: embedding dimension (dimension of each token)
        :param head_size: inner dimension of each head
        :param dropout: dropout rate
        """

        super().__init__()
        self.head_size = head_size

        # weight matrix to calculate key from input
        self.key = nn.Linear(d_model, head_size, bias=False)
        # weight matrix to calculate query from input
        self.query = nn.Linear(d_model, head_size, bias=False)
        # weight matrix to calculate value from input
        self.value = nn.Linear(d_model, head_size, bias=False)
        # triangular self-attention mask
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        inference of a single head
        :param x: input of shape (batch_size, sequence_length, d_model)
        """

        batch_size, sequence_length, d_model = x.shape

        k = self.key(x)                                                                             # (batch_size, sequence_length, head_size)
        q = self.query(x)                                                                           # (batch_size, sequence_length, head_size)
        v = self.value(x)                                                                           # (batch_size, sequence_length, head_size)

        # compute dot product attention: Q * transpose(K) / sqrt(d_model)
        # attention matrix
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5                                               # (batch_size, sequence_length, sequence_length)
        # masked attention matrix
        wei = wei.masked_fill(self.tril[:sequence_length, :sequence_length] == 0, float('-inf'))      # (batch_size, sequence_length, sequence_length)
        wei = F.softmax(wei, dim=-1)                                                                # (batch_size, sequence_length, sequence_length)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        out = wei @ v                                                                               # (B, sequence_length, sequence_length) @ (B, sequence_length, d_model) -> (B, sequence_length, d_model)

        return out


class FeedForward(nn.Module):
    """
    Feed Forward module
    """

    def __init__(self, d_model, dropout=0.0):
        super().__init__()

        # ratio between attention hidden dimension and ff hidden dimension
        ffwd_size_fac = 4

        # simple feed forward network
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * ffwd_size_fac),
            nn.ReLU(),
            nn.Linear(ffwd_size_fac * d_model, d_model),            # Projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """
    Multi Head self-attention
    """

    def __init__(self, context_length, d_model, head_size, num_heads, dropout=0.0):
        super().__init__()

        # multiple heads of self attention
        self.heads = nn.ModuleList([Head(context_length, d_model, head_size, dropout) for _ in range(num_heads)])
        # projection layer to merge information of all heads
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)     # (batch_size, sequence_length, num_heads * head_size)
        out = self.proj(out)
        out = self.dropout(out)
        return out



class CombinedMultiHeadAttention(nn.Module):
    """ We don't need to calculate the heads separately, we can express them as a single matrix multiplication"""

    def __init__(self, context_length, d_model, head_size, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model must be divisible by num_heads, d_model: {d_model}, num_heads: {num_heads}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = head_size

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # projection layer to merge information of all heads
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )

    def forward(self, x):
        batch_size, sequence_length, d_model = x.shape

        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)

        # Implicitly split the matrix by adding num_heads dimension
        keys = keys.view(batch_size, sequence_length, self.num_heads, self.head_size)
        queries = queries.view(batch_size, sequence_length, self.num_heads, self.head_size)
        values = values.view(batch_size, sequence_length, self.num_heads, self.head_size)

        # Transpose: (batch_size, sequence_length, num_heads, head_size) -> (batch_size, num_heads, sequence_length, head_size)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-prodcut attention
        attn_scores = queries @ keys.transpose(2, 3)

        # Truncate mask to actual sequence length
        mask = self.mask.bool()[:sequence_length, :sequence_length]

        # Mask attention scores
        attn_scores.masked_fill_(mask, -torch.inf)

        # Apply softmax and scaling
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) # (batch_size, sequence_length, num_heads, head_size)

        # Combine heads
        context_vec = context_vec.contiguous().view(batch_size, sequence_length, self.d_model)
        context_vec = self.proj(context_vec)

        return context_vec




class TransformerBlock(nn.Module):
    """
    Transformer block: communication followed by computation
    """

    def __init__(self, context_length, d_model, num_heads, dropout =0.0):
        super().__init__()
        # keep total hidden attention size independent of number of heads
        head_size = d_model // num_heads
        self.sa = MultiHeadAttention(context_length, d_model, head_size, num_heads, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x):
        # perform self attention with residual connection
        x = x + self.sa(self.ln1(x))
        # perform feed forward with residual connection
        x = x + self.ffwd(self.ln2(x))
        return x


class LayerNorm1d:
    """
    Layer-wise normalization module to prevent vanishing gradients
    Normalizes all features within a sample, one sample does not influence another one in contrast to batch norm
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True) # batch mean
        xvar = x.var(1, keepdim=True) # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]



class VanillaGPT(nn.Module):
    """
    Simple decoder-only LLM with multi-head dot product attention
    """

    def __init__(self, vocab_size, context_length, d_model, num_heads, n_layer, dropout=0.0, device='cpu'):
        super().__init__()
        self.device = device

        # embedding layer maps each token of vocabulary to embedding vector
        self.token_embedding_layer = nn.Embedding(vocab_size, d_model).to(self.device)
        # embedding maps each position of input to embedding vector
        self.position_embedding_layer = nn.Embedding(context_length, d_model).to(self.device)

        # generate all multi-head attention blocks
        self.blocks = nn.Sequential(*[TransformerBlock(context_length, d_model, num_heads, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

        # final linear layer
        self.lm_head = nn.Linear(d_model, vocab_size).to(self.device)

    def forward(self, idx):
        batch_size, sequence_length = idx.shape

        # idx and targets are both (batch_size, sequence_length) tensors of integers
        tok_emb = self.token_embedding_layer(idx)                                                       # (batch_size, sequence_length, d_model)
        pos_emb = self.position_embedding_layer(torch.arange(sequence_length, device=self.device))       # (sequence_length, d_model)

        x = tok_emb + pos_emb           # (batch_size, sequence_length, d_model)
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)        # (batch_size, context_length, vocab_size)
        return logits


    def generate(self, running_idx_sequence, context_length, num_tokens_to_generate):
        """
        Generate new tokens till max_new_tokens is reached based on input sequences
        :param running_idx_sequence: sequence of start tokens that gets extended
        :param context_length: length of the context window
        :param num_tokens_to_generate: number of new tokens to generate
        :return:
        """

        # running_idx_sequence contains all generated tokens including start tokens
        for _ in range(num_tokens_to_generate):
            # crop running_idx_sequence to the last context_length tokens
            idx_cond = running_idx_sequence[:, -context_length:]
            # get the predictions
            logits = self.forward(idx_cond)
            # focus only on the last timestep
            logits =  logits[:, -1, :] # becomes (batch_size, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (batch_size, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
            # append sampled index to the running sequence
            running_idx_sequence = torch.cat((running_idx_sequence, idx_next), dim=1) # (batch_size, ...)

        return running_idx_sequence


if __name__ == "__main__":

    """
    The purpose of this test is to verify that the `MultiHeadAttention` and 
    `CombinedMultiHeadAttention`  produce identical outputs given the 
    same inputs and weights. 
    """
    print("--- Running Attention Equivalence Test ---")

    # Define Hyperparameters for the test
    batch_size = 4
    context_length = 64
    d_model = 128
    num_heads = 4
    dropout = 0.0  # Dropout must be 0 for a deterministic comparison

    # Ensure d_model is divisible by num_heads
    assert d_model % num_heads == 0
    head_size = d_model // num_heads

    print(f"Test Parameters:\n  - Batch Size: {batch_size}\n  - Context Length: {context_length}\n  - D_Model: {d_model}\n  - Num Heads: {num_heads}\n")

    # Instantiate both attention modules with the same parameters
    mha = MultiHeadAttention(context_length, d_model, d_model // num_heads, num_heads, dropout)
    combined_mha = CombinedMultiHeadAttention(context_length, d_model, head_size, num_heads, dropout)

    # Set models to evaluation mode
    mha.eval()
    combined_mha.eval()

    # Copy weights from the iterative model to the combined model
    with torch.no_grad():
        # The W_q, W_k, and W_v matrices in the combined model are equivalent to the
        # concatenation of the corresponding matrices from each head in the iterative model.
        combined_mha.W_q.weight.data.copy_(torch.cat([head.query.weight for head in mha.heads], dim=0))
        combined_mha.W_k.weight.data.copy_(torch.cat([head.key.weight for head in mha.heads], dim=0))
        combined_mha.W_v.weight.data.copy_(torch.cat([head.value.weight for head in mha.heads], dim=0))

        # The final projection layers are equivalent and can be copied directly.
        combined_mha.proj.weight.data.copy_(mha.proj.weight.data)
        combined_mha.proj.bias.data.copy_(mha.proj.bias.data)

    # Create a random input tensor
    random_input = torch.randn(batch_size, context_length, d_model)

    # Perform the forward pass for both models
    output_mha = mha(random_input)
    output_combined_mha = combined_mha(random_input)

    # Compare the outputs
    are_outputs_equal = torch.allclose(output_mha, output_combined_mha, atol=1e-6)

    print(f"Output shape from MultiHeadAttention: {output_mha.shape}")
    print(f"Output shape from CombinedMultiHeadAttention: {output_combined_mha.shape}\n")

    # Report the result
    if are_outputs_equal:
        print("SUCCESS: The outputs of MultiHeadAttention and CombinedMultiHeadAttention are the same.")
    else:
        print("FAILURE: The outputs are different.")
        difference = torch.abs(output_mha - output_combined_mha).max().item()
        print(f"Maximum absolute difference between outputs: {difference}")

    print("--- Test Complete ---")


