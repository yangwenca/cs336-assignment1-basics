from einops import einsum, rearrange
import math
import torch


"""
Problem (transformer_accounting)
(a)
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400

total = 2 * vocab_size * d_model + num_layers * (2 * d_model + 4 * d_model ** 2 + 3 * d_model * d_ff) + d_model
= 2127057600 ~ 2.13B

each parameter is using single precision floating point
memory = 2127057600 * 2 ~ 4.25GB

(b) flops
num_layers * (8 * batch * seq_len * d_model**2 + 4 * batch * d_model * seq_len**2 + 6 * batch * seq_len * d_model * d_ff)
+ 2 * batch * seq_len * d_model * vocab_size
= 4513336524800 ~ 4.5e12 = 4.5 Tera FLOPs

(c) ffn requires the most FLOPs
attn: 1.3 Tera FLOPs
ffn: 3.0 Tera FLOPs
lm head: 0.1 Tera FLOPs

(d)
GPT-2 small:
num_layers = 12
d_model = 768
heads = 12
d_ff = 3072

total: 0.35 Tera FLOPs
attn: 0.1 Tera FLOPs 29%
ffn: 0.17 Tera FLOPs 49%
lm head: 0.08 Tera FLOPs 23%

GPT-2 medium:
num_layers = 24
d_model = 1024
heads = 16
d_ff = 4096

total: 1.03 Tera FLOPs
attn: 0.31 Tera FLOPs 30%
ffn: 0.62 Tera FLOPs 60%
lm head: 0.1 Tera FLOPs 10%

GPT-2 large:
num_layers = 36
d_model = 1280
heads = 20
d_ff = 5120

total: 2.26 Tera FLOPs
attn: 0.68 Tera FLOPs 30%
ffn: 1.45 Tera FLOPs 64%
lm head: 0.13 Tera FLOPs 6%

proportion of att stays relative stable, proportion of lm head decreases, proportion of ffn increases

(e)
GPT-2 XL
num_layers = 48
d_model = 1600
heads = 25
d_ff = 6400

total: 4.51 Tera FLOPs
attn: 1.33 Tera FLOPs 29%
ffn: 3.02 Tera FLOPs 67%
lm head: 0.16 Tera FLOPs 4%

proportion of att stays relative stable, proportion of lm head decreases, proportion of ffn increases
"""


'''
uv run pytest -k test_linear
uv run pytest -k test_embedding
uv run pytest -k test_rmsnorm
uv run pytest -k test_swiglu
uv run pytest -k test_rope
uv run pytest -k test_softmax_matches_pytorch
uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_4d_scaled_dot_product_attention
uv run pytest -k test_multihead_self_attention
uv run pytest -k test_transformer_block
uv run pytest -k test_transformer_lm
'''


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MPS has bugs on this op
        # return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return x @ self.weight.T


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(
            torch.ones((d_model), **factory_kwargs)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        square_x = torch.square(x)
        summation = einsum(square_x, "... d_model -> ...")
        rms = torch.sqrt(summation / self.d_model + self.eps)
        division = x / rms[..., None]
        result = einsum(division, self.weight, "... d_model, d_model -> ... d_model")
        return result.to(in_dtype)


"""
training parameters
3 * d_model * d_ff

FLOPs:
6 * batch * seq_len * d_model * d_ff
"""


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        sigmoid = torch.sigmoid(w1_x)
        w3_x = self.w3(x)
        value = w1_x * sigmoid * w3_x
        return self.w2(value)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    tmp_x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = tmp_x.unbind(dim=-1)
    stack_x = torch.stack((-x2, x1), dim=-1)
    return rearrange(stack_x, '... d r -> ... (d r)')


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(max_seq_len, device=device)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = rearrange(torch.stack((freqs, freqs), dim=-1), '... d r -> ... (d r)')
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.register_buffer("pre_cos", emb.cos(), persistent=False)
        self.register_buffer("pre_sin", emb.sin(), persistent=False)


    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        return x * self.pre_cos[token_positions] + rotate_half(x) * self.pre_sin[token_positions]


def Softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    max_value = torch.max(in_features, dim=dim, keepdim=True)
    diff = in_features - max_value.values
    exp = torch.exp(diff)
    total = torch.sum(exp, dim=dim, keepdim=True)
    return exp / total


def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    bias = torch.where(mask, 0.0, float('-inf'))
    scale_factor = 1 / math.sqrt(query.size(-1))
    qk = query @ key.transpose(-2, -1) * scale_factor
    qk += bias
    weight = Softmax(qk, -1)
    return weight @ value


"""
training parameters
4 * d_model ** 2

FLOPs:
8 * batch * seq_len * d_model**2
sdpa
4 * batch * seq_len ** 2 * d_model
total
8 * batch * seq_len * d_model**2 + 4 * batch * d_model * seq_len**2
"""


class Multihead_Self_Attention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device) if max_seq_len is not None else None
        self.q_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.output_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)


    def forward(
        self,
        in_features: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = in_features.device
        seq_len = in_features.size(-2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        key = self.k_proj(in_features)
        query = self.q_proj(in_features)
        value = self.v_proj(in_features)

        key = rearrange(key, '... sequence_length (h d_k) -> ... h sequence_length d_k',
                        h = self.num_heads, d_k = self.d_head)
        query = rearrange(query, '... sequence_length (h d_k) -> ... h sequence_length d_k',
                          h = self.num_heads, d_k = self.d_head)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=device)
            key = self.rope(key, token_positions)
            query = self.rope(query, token_positions)
        value = rearrange(value, '... sequence_length (h d_v) -> ... h sequence_length d_v',
                          h = self.num_heads, d_v = self.d_head)
        result = sdpa(key=key, query=query, value=value, mask=causal_mask)
        result = rearrange(result, '... h sequence_length d_v -> ... sequence_length (h d_v)',
                          h = self.num_heads, d_v = self.d_head)
        return self.output_proj(result)


"""
training paramater
RMSNorm: 2 * d_model
attn: 4 * d_model ** 2
ffn: 3 * d_model * d_ff

flops:
attn
8 * batch * seq_len * d_model**2 + 4 * batch * d_model * seq_len**2
ffn
6 * batch * seq_len * d_model * d_ff
"""


class Transformer_Block(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = Multihead_Self_Attention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )


    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        intermediate = in_features + self.attn(self.ln1(in_features))
        return intermediate + self.ffn(self.ln2(intermediate))


"""
training parameters
embedding: vocab_size * d_model
Per Transformer block:
RMSNorm: 2 * d_model
attn: 4 * d_model ** 2
ffn: 3 * d_model * d_ff
lm norm: d_model
lm head: d_model * vocab_size
total = vocab_size * d_model + num_layers * (2 * d_model + 4 * d_model ** 2 + 3 * d_model * d_ff) + d_model + d_model * vocab_size
= 2 * vocab_size * d_model + num_layers * (2 * d_model + 4 * d_model ** 2 + 3 * d_model * d_ff) + d_model

FLOPs:
attn
8 * batch * seq_len * d_model**2 + 4 * batch * d_model * seq_len**2
ffn
6 * batch * seq_len * d_model * d_ff

lm head: 2 * batch * seq_len * d_model * vocab_size

total = num_layers * (8 * batch * seq_len * d_model**2 + 4 * batch * d_model * seq_len**2 + 6 * batch * seq_len * d_model * d_ff)
+ 2 * batch * seq_len * d_model * vocab_size
"""


class Transformer_LM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        is_normalized: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        '''
        is_normalized: True/False including/excluding softmax
        '''
        super().__init__()
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = torch.nn.ModuleList(
            [
                Transformer_Block(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                ) for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )
        self.is_normalized = is_normalized


    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embeddings(in_features)
        for layer in self.layers:
            tokens = layer(tokens)
        result = self.lm_head(self.ln_final(tokens))
        if self.is_normalized:
            result = Softmax(result, dim=-1)
        return result
