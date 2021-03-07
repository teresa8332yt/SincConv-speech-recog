import torch
import torch.nn as nn
import torch.nn.functional as F


def Ternarize(tensor):
    # return (tensor-1e-10).sign().to(device=tensor.device, dtype=tensor.dtype)
    tensor[tensor > 0.2] = 1.0
    tensor[tensor < -0.2] = -1.0
    tensor[abs(tensor) <= 0.2] = 0.0
    return tensor


def InitWeight(tensor):  # set the initial distribution for the ternary weight −1, 0, +1 to 2.5% : 95% : 2.5%

    output = torch.randn(tensor.size()).to(
        device=tensor.device, dtype=tensor.dtype)
    delta = 2
    output[output >= delta] = 1
    output[output <= -delta] = -1
    output[abs(output) < delta] = 0

    return output


class TernaryConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, split=1):
        super(TernaryConv1d, self).__init__(in_channels, out_channels,
                                            kernel_size, stride, padding, dilation, groups, bias=bias)

        nn.init.normal_(self.weight.data, mean=0.0, std=1.0)

    def forward(self, input, ternarized_weight=True):

        if ternarized_weight is True:
            if not hasattr(self.weight, 'org'):
                self.weight.org = self.weight.data.clone()
            self.weight.data = Ternarize(self.weight.data)

        out = F.conv1d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

        return out
