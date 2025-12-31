from einops import einsum, rearrange
import math
import torch


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
        self.w1 = torch.nn.Parameter(torch.empty((d_ff, d_model)))
        self.w2 = torch.nn.Parameter(torch.empty((d_model, d_ff)))
        self.w3 = torch.nn.Parameter(torch.empty((d_ff, d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        sigmoid = torch.sigmoid(w1_x)
        w3_x = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        value = w1_x * sigmoid * w3_x
        return einsum(value, self.w2, "... d_ff, d_model d_ff -> ... d_model")


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
