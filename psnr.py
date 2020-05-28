import torch
from math import log10


class PSNR(torch.nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.criterion = torch.nn.MSELoss()
    def forward(self, prediction, target):
        mse = self.criterion(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        return psnr