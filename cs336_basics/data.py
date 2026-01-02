import os
import typing

import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.optimizer import CrossEntropy

"""
uv run pytest -k test_get_batch
uv run pytest -k test_checkpointing
"""


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dataset.ndim == 1
    assert len(dataset) > context_length

    # sample random starting indices
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)

    # collect sequences
    inputs = np.stack([dataset[s : s + context_length] for s in starts])
    targets = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts])

    # move to torch tensors on the requested device
    inputs = torch.from_numpy(inputs.astype(np.int32)).to(device=device)
    targets = torch.from_numpy(targets.astype(np.int32)).to(device=device)

    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint["iteration"]


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
    losses = 0

    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = CrossEntropy(logits, y)
        losses += loss.item()

    return losses / eval_iters
