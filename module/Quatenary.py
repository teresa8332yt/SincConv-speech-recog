import torch
import torch.nn as nn
import torch.nn.functional as F

from .Binary import Binarize


class quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, num_of_bits):
        n = 2 ** (num_of_bits-1)
        input_ = input_ * n

        return input_.round() / n

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class Quantize(nn.Module):
    def __init__(self, num_of_bits=4):
        super(Quantize, self).__init__()
        self.num_of_bits = num_of_bits
        self.quan = quantize.apply

    def forward(self, input):
        return self.quan(input, self.num_of_bits)

    def extra_repr(self):
        s = ('num_of_bits={num_of_bits}')

        return s.format(**self.__dict__)


class SplitQuatConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, num_of_bits=4, split=1):
        super(SplitQuatConv2d, self).__init__()
        if in_channels % split != 0:
            raise ValueError('in_channels must be divisible by split')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.split = split

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)

        self.weight = nn.Parameter(torch.randn(
            (self.out_channels * self.split, self.in_channels // split, self.kernel_size[0], self.kernel_size[1])))

        self.activation_p = nn.Hardtanh()
        self.num_of_bits = num_of_bits
        self.quan = quantize.apply

        self.mirror_scale = [2.38154823, 1.19320314, 1.08395584, 1.0409469,
                             1.01476617, 0.9955532, 0.98009437, 0.96562433, 0.95384186, 0.94339044]

    def forward(self, input, quantized_weight=True, partial_test=False):
        nB = input.size(0)

        if quantized_weight is True:
            # if not hasattr(self.weight,'org'):
            #     self.weight.org = self.weight.data.clone()
            # self.weight.data = Binarize(self.weight.data)
            # if not hasattr(self.weight,'org'):
            #     self.weight.org = self.weight.data.clone()
            self.weight.data = F.hardtanh(self.weight.data)
            self.weight.data = self.quan(self.weight.data, self.num_of_bits-1)

        if partial_test is True and self.in_channels != 3:
            self._weight.data = Binarize(self._weight.data)
            self._weight.data[self._weight.data == -1.0] = 0.0

            norm_scale = 2**self.num_of_bits - 1
            int_input = input * norm_scale

            out = None
            ref = None
            for b in range(self.num_of_bits):
                b_input = int_input % (2**(b+1)) // (2**b)

                partial_out = F.conv2d(b_input, self._weight, bias=None, stride=self.stride,
                                       padding=self.padding, dilation=1, groups=self.in_channels//self.partial_size)

                for i in range(1, 10):
                    # for i in range(6,19):
                    mask = partial_out.eq(i).float()
                    partial_out.data = partial_out.data + \
                        mask*(self.mirror_scale[i]-1)*i

                partial_out.data = partial_out.data * (2**b)

                partial_ref = F.conv2d(b_input, torch.ones_like(self._weight), bias=None, stride=self.stride,
                                       padding=self.padding, dilation=1, groups=self.in_channels//self.partial_size)

                for i in range(1, 10):
                    # for i in range(6,19):
                    mask = partial_ref.eq(i).float()
                    partial_ref.data = partial_ref.data + \
                        mask*(self.mirror_scale[i]-1)*i

                partial_ref.data = partial_ref.data * (2**b)

                if out is None:
                    out = partial_out
                    ref = partial_ref
                else:
                    out = out + partial_out
                    ref = ref + partial_ref

            out.data = out.data*2 - ref
            out = out / norm_scale

            out = out.view(out.size(0), -1, self.out_channels,
                           out.size(2), out.size(3))
            out = out.transpose(1, 2).contiguous()
            out = out.view(out.size(0), -1, self.split, self.in_channels //
                           (self.split*self.partial_size), out.size(3), out.size(4))
            out = out.transpose(1, 2).contiguous()
            out = out.sum(dim=3)

            out = out.view(out.size(0), -1, out.size(3), out.size(4))
        else:
            out = F.conv2d(input, self.weight, bias=None, stride=self.stride,
                           padding=self.padding, dilation=1, groups=self.split)

        out = out / 16
        out = self.activation_p(out)
        out.data = self.quan(out.data, self.num_of_bits-1)

        out = out.view(nB, -1, self.out_channels, out.size(2), out.size(3))
        out = out.sum(1)

        return out

    def test_reshape(self, partial_size=1):
        self.partial_size = partial_size

        self._weight = self.weight.view(
            self.split, -1, self.in_channels//(self.split*partial_size), partial_size, *self.kernel_size)
        self._weight = self._weight.transpose(1, 2).contiguous()
        self._weight = self._weight.view(-1, partial_size, *self.kernel_size)
        # print(self.weight)

        self._weight = nn.Parameter(self._weight)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.split != 0:
            s += ', split={split}'

        return s.format(**self.__dict__)


class QuaternaryConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, num_of_bits=4):
        super(QuaternaryConv2d, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding, dilation, groups, bias=bias)
        self.num_of_bits = num_of_bits
        self.quan = quantize.apply

    def forward(self, input, quantized_weight=True):

        if quantized_weight is True:
            # if not hasattr(self.weight, 'org'):
            #     self.weight.org = self.weight.data.clone()
            self.weight.data = F.hardtanh(self.weight.data)
            self.weight.data = self.quan(self.weight.data, self.num_of_bits-1)

        out = F.conv2d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

        return out


class QuaternaryConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, num_of_bits=4):
        super(QuaternaryConv1d, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding, dilation, groups, bias=bias)
        self.num_of_bits = num_of_bits
        self.quan = quantize.apply

        nn.init.normal_(self.weight.data, mean=0.0, std=0.3)

    def forward(self, input, quantized_weight=True):

        if quantized_weight is True:
            # if not hasattr(self.weight, 'org'):
            #     self.weight.org = self.weight.data.clone()
            self.weight.data = F.hardtanh(self.weight.data)
            self.weight.data = self.quan(self.weight.data, self.num_of_bits-1)

        out = F.conv1d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)

        return out


class QuaternaryLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, num_of_bits=8):
        super(QuaternaryLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.num_of_bits = num_of_bits
        self.quan = quantize.apply

    def forward(self, input, quantized_weight=True):

        input = self.quan(input, 4)

        if quantized_weight is True:
            if not hasattr(self.weight, 'org'):
                self.weight.org = self.weight.data.clone()
                if self.bias is not None:
                    self.bias.org = self.bias.data.clone()
            # self.weight.data = self.weight.data * self.weight_scale
            self.weight.data = F.hardtanh(self.weight.data)
            self.weight.data = self.quan(self.weight.data, self.num_of_bits-1)
            if self.bias is not None:
                self.bias.data = F.hardtanh(self.bias.data)
                self.bias.data = self.quan(self.bias.data, self.num_of_bits-1)

        out = F.linear(input, self.weight, self.bias)

        # self.weight.data = self.weight.data * (1/self.weight_scale)

        return out
