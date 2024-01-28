import torch.nn
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.sel = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.sel(output, target) * self.weight
