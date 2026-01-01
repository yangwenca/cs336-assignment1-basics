from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


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
