import torch
import torch.nn as nn
import torch.nn.functional as F

class CellPop(nn.Module):
    def __init__(self, in_ch, out_ch=None, ks=4, stride=1, padding=1,
                 lateral_size=5, adapt_gain=2.0, lateral_strength=1.0, use_pointwise=True):
        super().__init__()
        out_ch = out_ch or in_ch

        # depthwise conv for local recurrent field
        self.dw = nn.Conv2d(in_ch, in_ch, ks, stride, padding, groups=in_ch, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)

        # optional pointwise for channel mixing
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False) if use_pointwise else nn.Identity()

        # nonlinearity
        self.relu = nn.ReLU(inplace=True)

        # lateral surround (DoG)
        center = torch.zeros((lateral_size, lateral_size))
        mid = lateral_size // 2
        center[mid, mid] = 1.0
        gauss = torch.exp(-((torch.arange(lateral_size)-mid)[:,None]**2 +
                             (torch.arange(lateral_size)[None,:]-mid)**2)/4.0)
        gauss /= gauss.sum()
        dog = center - gauss
        self.register_buffer("lateral_kernel", dog[None,None,:,:].repeat(in_ch,1,1,1))

        self.lateral_strength = lateral_strength
        self.adapt_gain = adapt_gain
        self.register_buffer("adapt_state", torch.zeros(1, out_ch, 1, 1))

    def forward(self, x):
        y = self.relu(self.bn(self.dw(x)))
        y = self.pw(y)

        # lateral inhibition
        if self.lateral_strength > 0:
            pad = self.lateral_kernel.size(-1) // 2
            inhib = F.conv2d(y, self.lateral_kernel, padding=pad, groups=y.size(1))
            y = y - self.lateral_strength * inhib

        # adaptation
        if self.adapt_gain > 0:
            with torch.no_grad():
                mean_act = y.mean(dim=(0,2,3), keepdim=True)
                self.adapt_state.mul_(0.9).add_(0.1 * mean_act)
            y = y - self.adapt_gain * self.adapt_state

        return y
