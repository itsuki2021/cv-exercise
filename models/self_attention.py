import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation):
        super().__init__()
        self.in_dim = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        N, C, W, H = x.size()
        D = W * H
        proj_query = self.query_conv(x).view(N, -1, D).permute(0, 2, 1)     # BxNxC1
        proj_key = self.key_conv(x).view(N, -1, D)                          # BxC1xN
        energy = torch.bmm(proj_query, proj_key)                            # BxNxN
        attention = self.softmax(energy)                                    # BxNxN
        proj_value = self.value_conv(x).view(N, -1, D)                      # BxCxN
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(N, C, W, H)
        out = self.gamma * out + x

        return out, attention


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        mid_channel = channel // reduction
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        B, C, W, H = x.size()
        pool_m = self.max_pool(x)
        pool_a = self.avg_pool(x)
        pool_m1 = self.shared_MLP(pool_m.view(B, -1)).unsqueeze(2).unsqueeze(3)
        pool_a1 = self.shared_MLP(pool_a.view(B, -1)).unsqueeze(2).unsqueeze(3)

        return self.sigmoid(pool_m1 + pool_a1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7), stride=(1, 1), padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        pool1 = torch.mean(x, dim=1, keepdim=True)      # Nx1xWxH
        pool2, _ = torch.max(x, dim=1, keepdim=True)    # Nx1xWxH
        pool3 = torch.concat([pool1, pool2], dim=1)     # Nx2xWxH
        out = self.sigmoid(self.conv2d(pool3))

        return out
