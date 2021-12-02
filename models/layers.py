import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

def sigmoid(x):
    return float(1./(1.+np.exp(-x)))

def sparseFunction(x, s, b=None, activation=torch.relu, f=torch.sigmoid, ticket=False):
    #if ticket: return torch.sign(x)*(torch._C._nn.elu_(torch.abs(x)-f(s), 0.0) + 0.0)
    #if ticketxx: return torch.sign(x)*(
#torch.max(torch.tensor(0.0).cuda(),torch.abs(x)-torch.abs(s)) + torch.min(torch.tensor(0.0).cuda(),0.0*torch.abs(b)*(torch.abs(x)-torch.abs(s))) + 0.0*torch.abs(b)*torch.abs(s))
#
    #else: return torch.sign(x)*(
#torch.max(torch.tensor(0.0).cuda(),torch.abs(x)-torch.abs(s)) + torch.min(torch.tensor(0.0).cuda(),torch.abs(b)*(torch.abs(x)-torch.abs(s))) + torch.abs(b)*torch.abs(s))
#
    #r = torch.abs(x)-f(s)
    #tp = torch.tensor(0.5).cuda() + torch.sum(torch.sign(r))/(torch.sum(torch.sign(r)*torch.sign(r))*torch.tensor(2.0))
    if ticket: return torch.sign(x)*(
torch.max(torch.tensor(0.0).cuda(),torch.abs(x)-f(s)) + torch.min(torch.tensor(0.0).cuda(),0.0*(torch.abs(x)-f(s))) + 0.0*f(s.detach())
)
    else: return torch.sign(x)*(
torch.max(torch.tensor(0.0).cuda(),torch.abs(x)-f(s)) + torch.min(torch.tensor(0.0).cuda(),1e-6*b*(torch.abs(x)-f(s))) + 1e-6*b*f(s.detach())
)

class SoftMaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, mask_initial_value=0., groups=1, bias=False, dilation=1):
        super(SoftMaskedConv2d, self).__init__()
        self.mask_initial_value = mask_initial_value
        
        self.in_channels = in_channels
        self.out_channels = out_channels    
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels//groups, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        self.init_weight = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.init_mask()

        self.ticket = False
        self.activation = torch.relu
        self.f = torch.sigmoid
        self.sparseThreshold = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.betaThreshold = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels//self.groups, self.kernel_size, self.kernel_size))
        nn.init.constant_(self.mask_weight, self.mask_initial_value)

    def forward(self, x):
        sparseWeight = sparseFunction(self.weight, self.sparseThreshold, self.betaThreshold, self.activation, self.f, self.ticket)
        out = F.conv2d(x, sparseWeight, stride=self.stride, padding=self.padding, groups=self.groups, dilation = self.dilation)     
        return out

    def getSparsity(self, f=torch.sigmoid, ticket=False):
        sparseWeight = sparseFunction(self.weight, self.sparseThreshold, self.betaThreshold, self.activation, self.f, ticket)
        temp = sparseWeight.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.numel(), f(self.sparseThreshold).item(), (torch.abs(self.betaThreshold)*self.f(self.sparseThreshold)).item(), (torch.abs(self.betaThreshold)).item()
#        return (100 - temp.mean().item()*100), temp.numel(), f(self.sparseThreshold).item(), (torch.abs(self.betaThreshold)*self.f(self.sparseThreshold)).item(), (torch.abs(self.betaThreshold)).item()
     

    def extra_repr(self):
        return '{}, {}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
