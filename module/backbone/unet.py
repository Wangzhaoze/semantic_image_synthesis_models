# backbone/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def conv_block(in_ch, out_ch, norm='in', kernel_size=3, padding=1, activation=nn.LeakyReLU(0.2, inplace=True)):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=(norm is None))]
    if norm == 'bn':
        layers.append(nn.BatchNorm2d(out_ch))
    elif norm == 'in':
        layers.append(nn.InstanceNorm2d(out_ch, affine=False))
    elif norm == 'gn':
        layers.append(nn.GroupNorm(8, out_ch))
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    """
    Downsampling block: conv -> conv -> (optional) pooling
    """
    def __init__(self, in_ch, out_ch, norm='in', pool=True):
        super().__init__()
        self.conv1 = conv_block(in_ch, out_ch, norm=norm)
        self.conv2 = conv_block(out_ch, out_ch, norm=norm)
        self.pool = nn.MaxPool2d(2) if pool else None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        before_pool = x
        if self.pool is not None:
            x = self.pool(x)
            return x, before_pool
        return x, before_pool


class UpBlock(nn.Module):
    """
    Upsampling block with skip connection: upsample -> conv -> conv
    """
    def __init__(self, in_ch, out_ch, norm='in', up_mode='bilinear'):
        super().__init__()
        self.up_mode = up_mode
        if up_mode == 'transpose':
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            conv_in = out_ch * 2
        else:
            # bilinear upsample then conv to reduce channels
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )
            conv_in = out_ch * 2

        self.conv1 = conv_block(conv_in, out_ch, norm=norm)
        self.conv2 = conv_block(out_ch, out_ch, norm=norm)

    def forward(self, x, skip):
        x = self.up(x)
        # sometimes due to rounding shapes might differ by 1
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    """
    Standard U-Net with configurable depth and base filter number.
    - Input: (B, in_channels, H, W)
    - Output: (B, out_channels, H, W)
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_filters: int = 64,
        depth: int = 5,
        norm: str = 'in',
        up_mode: str = 'bilinear',
        final_activation: Optional[nn.Module] = nn.Tanh()
    ):
        """
        depth: number of downsampling levels including the bottleneck. e.g., depth=5 produces 4 downsamplings (H/16).
        """
        super().__init__()
        assert depth >= 3, "depth should be >=3"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.depth = depth
        self.norm = norm

        # Encoder
        self.downs: nn.ModuleList = nn.ModuleList()
        chs = []
        prev_ch = in_channels
        for d in range(depth):
            out_ch = num_filters * (2 ** d)
            pool = True if d < (depth - 1) else False  # last block is bottleneck without pooling
            block = DownBlock(prev_ch, out_ch, norm=norm, pool=pool)
            self.downs.append(block)
            chs.append(out_ch)
            prev_ch = out_ch

        # Decoder
        self.ups: nn.ModuleList = nn.ModuleList()
        for d in reversed(range(depth - 1)):
            in_ch = chs[d + 1]  # channels coming from previous level
            out_ch = chs[d]
            up_block = UpBlock(in_ch, out_ch, norm=norm, up_mode=up_mode)
            self.ups.append(up_block)

        # Final conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(chs[0], out_channels, kernel_size=1),
            # optional activation for output (e.g., Tanh for [-1,1] or Sigmoid for [0,1])
        )
        self.final_activation = final_activation

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder pass
        skips: List[torch.Tensor] = []
        out = x
        for down in self.downs:
            out, before_pool = down(out)
            skips.append(before_pool)

        # skips currently includes the bottleneck as last element; for decoder we ignore the last skip (it's the bottleneck)
        skips = skips[:-1][::-1]  # reverse order for decoder

        # Start decoding from bottleneck (out)
        for i, up in enumerate(self.ups):
            skip = skips[i]
            out = up(out, skip)

        out = self.final_conv(out)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out


if __name__ == "__main__":
    # quick sanity test
    net = UNet(in_channels=3, out_channels=3, num_filters=32, depth=5)
    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print(y.shape)  # expect (2,3,256,256)
