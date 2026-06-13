import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class GRL(autograd.Function):
    """Gradient Reversal Layer (Ganin et al., ICML 2016)."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim=512, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat, alpha=1.0):
        return self.net(GRL.apply(feat, alpha)).squeeze(1)


class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        mid = max(channels // r, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=32):
        super().__init__()
        g = lambda c: min(groups, c)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1    = nn.GroupNorm(g(out_ch), out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2    = nn.GroupNorm(g(out_ch), out_ch)
        self.se       = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(g(out_ch), out_ch),
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.se(self.norm2(self.conv2(out)))
        return F.relu(out + self.shortcut(x))


class ResNetUDA(nn.Module):
    """ResNet-18-scale with GroupNorm, SE blocks, projection + classifier heads."""
    def __init__(self, num_classes=200, proj_dim=128, groups=32, two_classifier=False):
        super().__init__()
        g = lambda c: min(groups, c)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(g(64), 64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64,  64,  2, stride=1, groups=groups)
        self.layer2 = self._make_layer(64,  128, 2, stride=2, groups=groups)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, groups=groups)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, groups=groups)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.projector  = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, proj_dim))
        self.classifier  = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes) if two_classifier else None

    @staticmethod
    def _make_layer(in_ch, out_ch, n, stride, groups):
        return nn.Sequential(
            ResBlock(in_ch, out_ch, stride=stride, groups=groups),
            *[ResBlock(out_ch, out_ch, groups=groups) for _ in range(1, n)],
        )

    def encode(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.pool(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.encode(x))

    def project(self, x):
        return self.projector(self.encode(x))

    def feat_parameters(self):
        return (list(self.stem.parameters()) +
                list(self.layer1.parameters()) + list(self.layer2.parameters()) +
                list(self.layer3.parameters()) + list(self.layer4.parameters()))

    def cls_parameters(self):
        params = list(self.classifier.parameters())
        if self.classifier2 is not None:
            params += list(self.classifier2.parameters())
        return params
