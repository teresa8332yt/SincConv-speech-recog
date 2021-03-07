import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Quatenary


def Binarize(tensor):
    return (tensor-1e-10).sign().to(device=tensor.device, dtype=tensor.dtype)
#     out = tensor.clone()
#     out[out>0] = 1.0
#     out[out<=0] = -1.0
#     return out


def InitWeight(tensor):  # set the initial distribution for the ternary weight −1, 0, +1 to 2.5% : 95% : 2.5%

    output = torch.randn(tensor.size()).to(
        device=tensor.device, dtype=tensor.dtype)
    delta = 2
    output[output >= delta] = 1
    output[output <= -delta] = -1
    output[abs(output) < delta] = 0

    return output


class BinaryConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, split=1):
        super(BinaryConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias=bias)

        self.partial_size = None
        self.split = split

        if self.split != 1:
            self.htanh = nn.Hardtanh()
            self.quan = Quatenary.Quantize(num_of_bits=3)
            self.weight.data = self.weight.data.view(
                out_channels*split, -1, kernel_size//split)

    def forward(self, input, binarized_weight=True, partial_test=False):

        if binarized_weight is True:
            if not hasattr(self.weight, 'org'):
                self.weight.org = self.weight.data.clone()
            self.weight.data = Binarize(self.weight.data)

        # if partial_test is True:
        #     self._weight.data = Binarize(self._weight.data)

        #     out = F.conv1d(input, self._weight, None, self.stride, self.padding,
        #                    self.dilation, self.in_channels//self.partial_size)

        #     # self.sta_s += out.flatten().tolist()

        #     out = self.quan(out)

        #     acc_out = None

        #     out = out.view(out.size(0), self.in_channels //
        #                    self.partial_size, self.kernel_size[0], -1, out.size(2))
        #     out = out.transpose(1, 2).contiguous()
        #     out = out.view(out.size(0), self.kernel_size[0], -1, out.size(4))

        #     output_width = (input.size(
        #         2)+2*self.padding[0]-self.kernel_size[0])//self.stride[0]+1
        #     for windex in range(self.kernel_size[0]):
        #         sindex = windex
        #         slice_out = out[:, sindex, :, windex:windex+output_width]

        #         if acc_out is not None:
        #             acc_out = acc_out + slice_out
        #         else:
        #             acc_out = slice_out
        #     out = acc_out

        #     out = out.view(out.size(0), -1, self.out_channels //
        #                    self.groups, out.size(2))
        #     out = out.transpose(1, 2).contiguous()
        #     out = out.view(out.size(0), -1, self.groups, self.in_channels //
        #                    (self.groups*self.partial_size), out.size(3))
        #     out = out.transpose(1, 2).contiguous()
        #     out = out.sum(dim=3)
        #     out = out.view(out.size(0), -1, out.size(3))

        if self.split != 1:
            partial_out = F.conv1d(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)

            partial_out = partial_out.view(partial_out.size(
                0), self.groups, self.out_channels*self.split//self.groups, partial_out.size(2))

            partial_out.data = partial_out.data / 64.
            partial_out = self.htanh(partial_out)
            partial_out = self.quan(partial_out)
            partial_out.data = partial_out.data * 64.

            out_width = (input.size(
                2)+2*self.padding[0]-self.kernel_size[0])//self.stride[0]+1
            out_group_size = self.out_channels // self.groups
            split_size = self.kernel_size[0] // self.split
            for s in range(self.split):
                slice_out = partial_out[:, :, s*out_group_size:(
                    s+1)*out_group_size, s*split_size:s*split_size+out_width]
                if s == 0:
                    out = slice_out
                else:
                    out = out + slice_out
            out = out.view(out.size(0), self.out_channels, out_width)
        else:
            out = F.conv1d(input, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)

        # self.sta_o += out.flatten().tolist()
        # self.sta += self.bn(out).flatten().tolist()
        # self.sta += out.flatten().tolist()

        return out

    def test_reshape(self, partial_size=1):
        self.partial_size = partial_size

        # self._weight = self.weight.view(self.groups, -1, self.in_channels//(self.groups*partial_size), partial_size,*self.kernel_size)
        # self._weight = self._weight.transpose(1, 2).contiguous()
        # self._weight = self._weight.view(-1, partial_size, *self.kernel_size)

        self._weight = self.weight.view(self.groups, self.out_channels//self.groups,
                                        self.in_channels//(self.groups*partial_size), partial_size, self.kernel_size[0])
        self._weight = self._weight.transpose(1, 2).contiguous()
        self._weight = self._weight.view(-1, partial_size, self.kernel_size[0])

        self._weight = self._weight.view(
            self.in_channels//partial_size, -1, partial_size, 1, 1, self.kernel_size[0])
        self._weight = self._weight.permute(0, 5, 1, 2, 3, 4).contiguous()
        self._weight = self._weight.view(-1, partial_size, 1)

        self._weight = nn.Parameter(self._weight)

    class BinaryLSTM(nn.Module):
        def __init__(self, inupt_size, hidden_size, bias=False):
            super(BinaryLSTM, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            # self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
            # self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
            self.x2h = nn.Parameter(torch.Tensor(out_features*4, in_features))
            self.h2h = nn.Parameter(torch.Tensor(out_features*4, in_features))
            if bias is True:
                self.bias = nn.Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)

            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.x2h, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.h2h, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.x2h)
                bound = 1 / math.sqrt(fan_in)

        def forward(self, input, hidden=None):

            if hidden is not None:
                hx, cx = hidden
            else:
                hx = torch.zeros(())
