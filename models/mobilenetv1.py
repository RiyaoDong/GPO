import torch.nn as nn

from .layers import *

import functools
from torch.nn import init
import copy


__all__ = ['mobilenet']

class MobileNetV1(nn.Module):

    def __init__(self, mask_initial_value=0., num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.num_classes = num_classes
        self.Conv = functools.partial(SoftMaskedConv2d, mask_initial_value=mask_initial_value)

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                self.Conv(inp, oup, 3, 1, stride, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                self.Conv(inp, inp, 3, 1, stride, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                self.Conv(inp, oup, 1, 0, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        self.mask_modules = [m for m in self.modules() if type(m) == SoftMaskedConv2d]

    def checkpoint(self):
        for m in self.mask_modules: m.checkpoint()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules: m.rewind_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.load_state_dict(m.checkpoint)

    def prune(self):
        for m in self.mask_modules: m.prune()

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        x = x.view(-1, self.num_classes)
        return x

def mobilenet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return MobileNetV1(**kwargs)
