from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


"""
uv run pytest -k test_cross_entropy
uv run pytest -k test_adamw
"""

def CrossEntropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    dim = -1
    max_value, _ = torch.max(inputs, dim=dim, keepdim=True)
    sub = inputs - max_value
    total = torch.log(torch.sum(torch.exp(sub), dim=dim, keepdim=True))
    diff = torch.gather(sub, dim=dim, index=targets.unsqueeze(dim))
    loss = (total - diff).squeeze(dim)
    return torch.mean(loss)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


def toy_example(learning_rate: float, iterations: int):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=learning_rate)

    for t in range(iterations):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(t, loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.


"""
learning rate tuning
1e1: converges slower
1e2: converges faster
1e3: does not converge, diverge
need to choose the right learning rate,
if it is too small, converges slow
if it is too large, it diverges
"""
def play_around():
    toy_example(1e1, 10)
    toy_example(1e2, 10)
    toy_example(1e3, 10)

# play_around()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float=1e-3,
        weight_decay: float=0.01,
        betas: tuple[float, float]=(0.9, 0.999),
        eps: float=1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay,
                    "betas": betas, "eps": eps}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state['m'] = torch.zeros_like(p)
                state['v'] = torch.zeros_like(p)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t * state['m'] / (torch.sqrt(state['v']) + eps)
                p.data *= (1 - lr * weight_decay)
                state["t"] = t + 1 # Increment iteration number.
        return loss


def Adam_Accounting(
    batch_size: int,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
    num_heads: int,
    precision: int,
):
    size = precision // 8
    # parameters
    parameters = 2 * vocab_size * d_model + num_layers * (2 * d_model + 4 * d_model ** 2 + 3 * d_model * d_ff) + d_model
    # gradients (each parameter has a gradient of the same size)
    gradients = parameters
    # AdamW optimizer (first moment m and second memont v), both same size as parameters
    optimizer = 2 * parameters
    # activations
    activation = 0
    # Transformer block
    transformer_block = 0
    # - RMSNorms
    transformer_block += batch_size * context_length * d_model
    # - MHA QKV projections, QK matrix multiply, sofmtax, weighted sum of values, output projection
    transformer_block += 3 * batch_size * context_length * d_model + batch_size * num_heads * context_length**2 + \
                        batch_size * num_heads * context_length**2 + batch_size * context_length * d_model + \
                        batch_size * context_length * d_model
    # - Position wise feedforward: W1 matrix multiply, SiLU, W2 matrix multiply
    transformer_block += 3 * batch_size * context_length * d_ff + batch_size * context_length * d_model

    activation += num_layers * transformer_block
    # final RMSNorm
    activation += batch_size * context_length * d_model
    # output embedding
    activation += batch_size * context_length * vocab_size
    # cross-entropy on logits
    activation += batch_size * context_length * vocab_size
    parameters *= size
    gradients *= size
    optimizer *= size
    activation *= size
    gb = 1e9
    print("values in GB are ", parameters / gb, gradients / gb, optimizer / gb, activation / gb)
    return parameters, gradients, optimizer, activation


"""
Adam Accounting
(a)
Parameters:
total = 2 * vocab_size * d_model + num_layers * (2 * d_model + 4 * d_model ** 2 + 3 * d_model * d_ff) + d_model
= 2 * vocab_size * d_model + num_layers * (2 * d_model + 4 * d_model ** 2 + 3 * d_model * 4 * d_model) + d_model
= 2 * vocab_size * d_model + num_layers * (2 * d_model + 16 * d_model ** 2) + d_model

assume float32
= (2 * vocab_size * d_model + num_layers * (2 * d_model + 16 * d_model ** 2) + d_model) * 4
= 8 * vocab_size * d_model + 4 * num_layers * (2 * d_model + 16 * d_model ** 2) + 2 * d_model

Activations:
Transformer block:
- RMSNorm(s)
batch * context_length * d_model
- Multi-head self-attention sublayer: QKV projections, QK matrix multiply, softmax, weighted sum of values, output projection.
3 * batch * context_length * d_model + 2 * batch * num_heads * context_length ** 2 + batch * context_length * d_model 
+ batch * context_length * d_model
= 5 * batch * context_length * d_model + 2 * batch * num_heads * context_length ** 2
- Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
= 3 * batch * context_length * d_ff + batch * context_length * d_model

final RMSNorm
batch * context_length * d_model

output embedding
batch * context_length * vocab_size

cross-entropy on logits
batch * context_length * vocab_size

total = 2 * batch * context_length * vocab_size + batch * context_length * d_model +
num_layers *
(batch * context_length * d_model + 5 * batch * context_length * d_model + 2 * batch * num_heads * context_length ** 2
+ 3 * batch * context_length * d_ff + batch * context_length * d_model)
= 2 * batch * context_length * vocab_size + batch * context_length * d_model +
num_layers * (7 * batch * context_length * d_model + 2 * batch * num_heads * context_length ** 2 + 3 * batch * context_length * d_ff)

Gradients:
same as paramters

Optimizer state: (first moment and second moment)
2 * parameters


(b)
16.461471744 * batch_size + 34.0329216 <= 80
batch_size = 2

(c)
FLOPs
first moment: 3
second moment: 3
bias correction: 2
paramater update: 4
weight decay: 3
apply update: 1
total 18 per parameter

paramater from part a
= 2 * vocab_size * d_model + num_layers * (2 * d_model + 16 * d_model ** 2) + d_model

(d)
GPT-2 XL
forward: 4.51 Tera FLOPs (from last part)
backward: 2 * forward
optimizer: 0.038 Tera FLOPs
one step: 13.57 Tera FLOPs
4.51 * (2 + 1) * 1024 * 400k teraFLOPs + 0.038 * 400k teraFLOPs
= 5.54e9 teraFLOPs
5.54e9 / (19.5 * 0.5) / 60 / 60 / 24 = 6576 days
"""


def GPT2_XL():
    num_layers = 48
    d_model = 1600
    heads = 25
    d_ff = 6400
    vocab_size = 50257
    context_length = 1024
    Adam_Accounting(
        batch_size=1,
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        d_ff=d_ff,
        num_heads=heads,
        precision=32,
    )


# GPT2_XL()


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate
    return min_learning_rate + \
            0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * \
            (max_learning_rate - min_learning_rate)
