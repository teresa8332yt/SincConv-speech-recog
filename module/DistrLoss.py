import torch
import torch.nn as nn


class DistrLoss(nn.Module):

    def __init__(self, channels):
        super(DistrLoss, self).__init__()
        self._channels = channels

    def forward(self, input):
        if input.dim() != 4 and input.dim() != 3 and input.dim() != 2:
            raise ValueError('expected 4D, 3D or 2D input (got {}D input)'
                             .format(input.dim()))
        if input.size()[1] != self._channels:
            raise ValueError('expected {} channels (got {}D input)'
                             .format(self._channels, input.size()[1]))

        if input.dim() == 4:
            mean = input.mean(dim=-1).mean(dim=-1).mean(dim=0)
            var = ((input - mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) ** 2
                   ).mean(dim=-1).mean(dim=-1).mean(dim=0)
        elif input.dim() == 3:
            mean = input.mean(dim=-1).mean(dim=0)
            var = ((input - mean.unsqueeze(0).unsqueeze(2)) ** 2
                   ).mean(dim=-1).mean(dim=0)
        elif input.dim() == 2:
            mean = input.mean(dim=0)
            var = ((input - mean.unsqueeze(0)) ** 2).mean(dim=0)

        var = var + 1e-10  # to avoid 0 variance
        std = var.abs().sqrt()

        distrloss1 = (torch.min(1 - mean - 0.25*std, 0 + mean - 0.25 *
                                std).clamp(min=0) ** 2).mean() + ((std - 4).clamp(min=0) ** 2).mean()  # Gradient mismatch + Saturation
        distrloss2 = ((mean-0.5) ** 2 - std **
                      2).clamp(min=0).mean()  # Degeneration

        return [distrloss1, distrloss2]
