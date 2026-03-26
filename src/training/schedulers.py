import torch


def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str, epochs: int):
    name = str(scheduler_name).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    return None
