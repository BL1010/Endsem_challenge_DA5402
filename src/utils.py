import torch

def rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

def mae(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()

def accuracy(preds, targets, threshold=3.5):
    preds_bin = (preds >= threshold).float()
    targets_bin = (targets >= threshold).float()
    return (preds_bin == targets_bin).float().mean().item()