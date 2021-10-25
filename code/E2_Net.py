# -*- coding: utf-8 -*-

from torch import nn, cat
from torch.nn.functional import dropout

class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x

class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x
    
class E2_Net(nn.Module):
    def __init__(self, n_channels, n_classes, n_h=10, branch=4):
        super(E2_Net, self).__init__()

        self.maxpooling = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        self.conv14 = EncoderConv(n_channels, 16)
        self.conv15 = EncoderConv(16, 16)
        self.conv16 = EncoderConv(16, 32)
        self.conv17 = EncoderConv(32, 32)
        self.conv18 = EncoderConv(32, 64)
        self.conv19 = EncoderConv(64, 64)
        self.conv20 = DecoderConv(96, 32)
        self.conv21 = DecoderConv(32, 32)
        self.conv22 = DecoderConv(48, 16)
        self.conv23 = DecoderConv(16, 16)
        self.h_conv1 = nn.Conv3d(16, n_h, 1)
        self.out_conv1 = nn.Conv3d(16, n_classes, 1)

    def forward(self, x):
        # net2
        x1 = self.maxpooling(x)

        # block0
        x1_0_0 = self.conv14(x1)
        x1_0_1 = self.conv15(x1_0_0)

        # block1
        x1 = self.maxpooling(x1_0_1)
        x1_1_0 = self.conv16(x1)
        x1_1_1 = self.conv17(x1_1_0)

        # block2
        x1 = self.maxpooling(x1_1_1)
        x1_2_0 = self.conv18(x1)
        x1_2_1 = self.conv19(x1_2_0)

        # block3
        x1 = self.up(x1_2_1)
        x1_1_2 = self.conv20(cat([x1,x1_1_1], dim=1))
        x1_1_3 = self.conv21(x1_1_2)

        # block4
        x1 = self.up(x1_1_3)
        x1_0_2 = self.conv22(cat([x1,x1_0_1], dim=1))
        x1_0_3 = self.conv23(x1_0_2)
        # x = self.dropout(x)
        out1 = self.out_conv1(x1_0_3)
        out1 = self.sigmoid(out1)
                
        out1 = self.up(out1)
        
        return out1

