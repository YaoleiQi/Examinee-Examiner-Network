# -*- coding: utf-8 -*-


import torch

from torch import nn
from torch.nn.functional import max_pool3d

class crossentry(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred+smooth))
    
class cross_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred+smooth)+ (1-y_true) * torch.log(1 - y_pred+smooth))


class dropoutput_layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1e-6
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        loss_ce = -((torch.sum(w * y_true * torch.log(y_pred + smooth)) / torch.sum(w * y_true+ smooth)) +
                             (torch.sum(w * (1 - y_true) * torch.log(1 - y_pred + smooth)) / torch.sum(w * (1 - y_true) + smooth)))/2
        
        #temp_y = torch.where(y_pred>=0, torch.full_like(y_pred, 1), torch.full_like(y_pred, 0))
        #print(torch.sum(w)/torch.sum(temp_y))
        return loss_ce
    
class dropoutput_layer_softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1e-6
        y_pred = y_pred[:, 1, :, :, :]
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        loss_ce = -((torch.sum(w * y_true * torch.log(y_pred + smooth)) / torch.sum(w * y_true+ smooth)) +
                             (torch.sum(w * (1 - y_true) * torch.log(1 - y_pred + smooth)) / torch.sum(w * (1 - y_true) + smooth)))/2
        return loss_ce
