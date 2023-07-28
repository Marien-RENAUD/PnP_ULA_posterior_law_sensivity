from torch import nn
import os
import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
import torch.nn.functional as Func
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class DnCNN(nn.Module):
    def __init__(self, network= 'Dncnn', net_mode=1, depth=12, n_channels=64, image_channels=1, kernel_size=3, is_traing=True):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        self.is_traing = is_traing
        layers.append(SpectralNorm(nn.Conv2d(
            in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False)))
        layers.append(nn.ReLU())

        for _ in range(depth-1):
            layers.append(SpectralNorm(nn.Conv2d(
                in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)))
            layers.append(nn.ReLU())

        layers.append(SpectralNorm(nn.Conv2d(
            in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False)))

        self.dncnn_3 = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn_3(x)