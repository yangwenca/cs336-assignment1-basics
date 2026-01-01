from einops import einsum, rearrange
import math
import torch


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
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


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


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)


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


class Multihead_Self_Attention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len) if max_seq_len is not None else None
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)


    def forward(
        self,
        in_features: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = in_features.size(-2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        key = self.k_proj(in_features)
        query = self.q_proj(in_features)
        value = self.v_proj(in_features)

        key = rearrange(key, '... sequence_length (h d_k) -> ... h sequence_length d_k',
                        h = self.num_heads, d_k = self.d_head)
        query = rearrange(query, '... sequence_length (h d_k) -> ... h sequence_length d_k',
                          h = self.num_heads, d_k = self.d_head)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len)
            key = self.rope(key, token_positions)
            query = self.rope(query, token_positions)
        value = rearrange(value, '... sequence_length (h d_v) -> ... h sequence_length d_v',
                          h = self.num_heads, d_v = self.d_head)
        result = sdpa(key=key, query=query, value=value, mask=causal_mask)
        result = rearrange(result, '... h sequence_length d_v -> ... sequence_length (h d_v)',
                          h = self.num_heads, d_v = self.d_head)
        return self.output_proj(result)


class Transformer_Block(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = Multihead_Self_Attention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
        )

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        intermediate = in_features + self.attn(self.ln1(in_features))
        return intermediate + self.ffn(self.ln2(intermediate))
