# -*- coding: utf-8 -*-

from torch import nn, cat
from torch.nn.functional import dropout

# Define Encoder
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

# Define Decoder
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


class E1_Net(nn.Module):
    def __init__(self, n_channels, n_classes, n_h=10, branch=4):
        super(E1_Net, self).__init__()
        self.conv0 = EncoderConv(n_channels, 16)
        self.conv1 = EncoderConv(16, 16)
        self.conv2 = EncoderConv(16, 32)
        self.conv3 = EncoderConv(32, 32)
        self.conv4 = EncoderConv(32, 64)
        self.conv5 = EncoderConv(64, 64)
        self.conv6 = EncoderConv(64, 128)
        self.conv7 = EncoderConv(128, 128)
        self.conv8 = EncoderConv(128, 256)
        self.conv9 = EncoderConv(256, 256)
        self.conv10 = EncoderConv(384, 128)
        self.conv11 = EncoderConv(128, 128)
        self.conv12 = DecoderConv(192, 64)
        self.conv13 = DecoderConv(64, 64)
        self.conv14 = DecoderConv(96, 32)
        self.conv15 = DecoderConv(32, 32)
        self.conv16 = DecoderConv(48, 16)
        self.conv17 = DecoderConv(16, 16)
        self.h_conv = nn.Conv3d(16, n_h, 1)
        self.out_conv = nn.Conv3d(16, n_classes, 1)
        self.maxpooling = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Due to the large size of the image, we pre-sampled
        x = self.maxpooling(x)

        # block0
        x_0_0 = self.conv0(x)
        x_0_1 = self.conv1(x_0_0)

        # block1
        x = self.maxpooling(x_0_1)
        x_1_0 = self.conv2(x)
        x_1_1 = self.conv3(x_1_0)

        # block2
        x = self.maxpooling(x_1_1)
        x_2_0 = self.conv4(x)
        x_2_1 = self.conv5(x_2_0)

        # block3
        x = self.maxpooling(x_2_1)
        x_3_0 = self.conv6(x)
        x_3_1 = self.conv7(x_3_0)
        
        # block3
        x = self.maxpooling(x_3_1)
        x_4_0 = self.conv8(x)
        x_4_1 = self.conv9(x_4_0)
        
        # block4
        x = self.up(x_4_1)
        x_3_2 = self.conv10(cat([x,x_3_1], dim=1))
        x_3_3 = self.conv11(x_3_2)        

        # block4
        x = self.up(x_3_3)
        x_2_2 = self.conv12(cat([x,x_2_1], dim=1))
        x_2_3 = self.conv13(x_2_2)

        # block5
        x = self.up(x_2_3)
        x_1_2 = self.conv14(cat([x,x_1_1], dim=1))
        x_1_3 = self.conv15(x_1_2)

        # block6
        x = self.up(x_1_3)
        x_0_2 = self.conv16(cat([x,x_0_1], dim=1))
        x_0_3 = self.conv17(x_0_2)
        # x = self.dropout(x)
        out = self.out_conv(x_0_3)
        out = self.sigmoid(out)
        out = self.up(out)
        
        return out
    
