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
        wei = q @ k.transpose(-2, -1) * d_model**-0.5                                               # (batch_size, sequence_length, sequence_length)
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


    B, T, C = 4, 8, 2  # batch, time, channels
    x = torch.randn(B, T, C)
    print(x.shape)

    # We want x[b, t] = mean_{i<=t} x[b, i]

    # Version 1
    xbow = torch.zeros((B, T, C))           # bow = back of words
    for b in range(B):
        for t in range(T):
            xprev = x[b, :t+1] # (t, C)
            xbow[b, t] = torch.mean(xprev, 0)


    # Version 2
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(1, keepdim=True)

    xbow2 = wei @ x # (T, T) @ (B, T, C) ---> (B, T, C)

    # Version 3
    tril = torch.tril(torch.ones(T, T))
    wei = torch.zeros((T, T))
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    xbow3 = wei @ x

    print(f"x[0]: {x[0]}")
    print(f"xbow[1]: {xbow[1]}")
    print(f"xbow2[1]: {xbow2[1]}")

    print(f"xbow same as xbow2? {torch.allclose(xbow, xbow2, rtol=1e-3)}")
    print(f"xbow same as xbow3? {torch.allclose(xbow, xbow3, rtol=1e-3)}")


    # Version 4: self-attention
    B, T, C = 4, 8, 32
    x = torch.randn(B, T, C)

    # let's see a single Head perform self-attention
    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)

    k = key(x)      # (B, T, head_size)
    q = query(x)    # (B, T, head_size)
    wei = q @ k.transpose(-2, -1) * head_size**-0.5  # (B, T, head_size) @ (B, 16, T) ---> (B, T, T)
    tril = torch.tril(torch.ones(T, T))
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)

    v = value(x)
    out = wei @ v




