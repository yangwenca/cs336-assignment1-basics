
import numpy as np
import numpy.typing as npt
import torch


"""
uv run pytest -k test_get_batch
"""


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dataset.ndim == 1
    assert len(dataset) > context_length

    # sample random starting indices
    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    # collect sequences
    inputs = np.stack([dataset[s : s + context_length] for s in starts])
    targets = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts])

    # move to torch tensors on the requested device
    inputs = torch.from_numpy(inputs).to(device=device)
    targets = torch.from_numpy(targets).to(device=device)

    return inputs, targets
