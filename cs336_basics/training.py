import torch

def CrossEntropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    dim = -1
    max_value, _ = torch.max(inputs, dim=dim, keepdim=True)
    sub = inputs - max_value
    total = torch.log(torch.sum(torch.exp(sub), dim=dim, keepdim=True))
    diff = torch.gather(sub, dim=dim, index=targets.unsqueeze(dim))
    loss = (total - diff).squeeze(dim)
    return torch.mean(loss)