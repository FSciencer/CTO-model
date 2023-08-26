import torch.nn as nn
import torch
import torch.nn.functional as F
from CTrans import ChannelTransformer


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(Flatten(),
                                   nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(Flatten(),
                                   nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):  # g: upsampled feature-[B, 2048, 32, 32]; x: skip feature-[B, 1024, 32, 32]
        # channel-wise attention
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # global average pooling, [B, 1024, 32, 32] -> [B, 1024, 1, 1]
        channel_att_x = self.mlp_x(avg_pool_x)  # [B, 1024, 1, 1] -> [B, 1024]
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))  # global average pooling, [B, 2048, 32, 32] -> [B, 2048, 1, 1]
        channel_att_g = self.mlp_g(avg_pool_g)  # [B, 2048, 1, 1] -> [B, 1024]
        channel_att_sum = (channel_att_x + channel_att_g)/2.0  # [B, 1024]
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)  # [B, 1024] -> [B, 1024, 32, 32]
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels, F_x=skip_channels)
        self.nConvs = _make_nConv(in_channels + skip_channels, out_channels, nb_Conv, activation)  # input channel: F_g+F_x

    def forward(self, x, skip_x):
        up = self.up(x)  # [2048,  16,  16] -> [2048,  32,  32]
        skip_x_att = self.coatt(g=up, x=skip_x)  # {g=[2048, 32, 32], x=[1024, 32, 32]} -> [1024, 32, 32]
        x = torch.cat([skip_x_att, up], dim=1)  # [2048, 32, 32]
        return self.nConvs(x)  # # [2048, 32, 32] -> # [512, 32, 32]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UCTNet(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, num_blocks=None, feature_size=512, vis=False):
        super().__init__()
        if num_blocks is None:
            num_blocks = [3, 4, 6, 3]
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.in_planes = config.base_channel

        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, in_channels, kernel_size=7, stride=2, padding=3, bias=False),  # [1, 512, 512] -> [64, 256, 256]
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # [64, 256, 256] -> [64, 128, 128]
        self.down1 = self._make_layer(Bottleneck, in_channels, num_blocks[0], stride=1)    # [ 64, 128, 128] -> [ 256, 128, 128]
        self.down2 = self._make_layer(Bottleneck, in_channels*2, num_blocks[1], stride=2)  # [256, 128, 128] -> [ 512,  64,  64]
        self.down3 = self._make_layer(Bottleneck, in_channels*4, num_blocks[2], stride=2)  # [512,  64,  64] -> [1024,  32,  32]
        self.down4 = self._make_layer(Bottleneck, in_channels*8, num_blocks[3], stride=2)  # [1024, 32, 32]  -> [2048,  16,  16]

        fea_channels = [in_channels*16, in_channels*8, in_channels*4, in_channels]

        self.mtc = ChannelTransformer(config, vis, feature_size,
                                      patchSize=config.patch_sizes,  # config.patch_sizes = [16, 8, 4, 2]
                                      channel_num=[in_channels, in_channels*4, in_channels*8, in_channels*16])
        self.up4 = UpBlock_attention(in_channels*32, fea_channels[0], in_channels*8, nb_Conv=2)  # {[2048,  16,  16], [1024,  32,  32]} -> [512,  32,  32]
        self.up3 = UpBlock_attention(in_channels*8,  fea_channels[1], in_channels*4, nb_Conv=2)  # {[ 512,  32,  32], [ 512,  64,  64]} -> [256,  64,  64]
        self.up2 = UpBlock_attention(in_channels*4,  fea_channels[2], in_channels*2, nb_Conv=2)  # {[ 256,  64,  64], [ 256, 128, 128]} -> [128, 128, 128]
        self.up1 = UpBlock_attention(in_channels*2,  fea_channels[3], in_channels*1, nb_Conv=2)  # {[ 128, 128, 128], [  64, 256, 256]} -> [ 64, 256, 256]
        self.up0 = nn.Upsample(scale_factor=2)  # [64, 256, 256] -> [64, 512, 512]
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))  # [64, 512, 512] -> [1, 512, 512]
        self.last_activation = nn.Sigmoid()  # if using BCELoss

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # 64 -> 256 -> 1024 -> 4096
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        x1 = self.stem(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(self.up0(x)))
        else:
            logits = self.outc(self.up0(x))  # if nusing BCEWithLogitsLoss or class>1
        if self.vis:  # visualize the attention maps
            return logits, att_weights
        else:
            return logits
