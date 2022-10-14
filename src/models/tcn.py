## Reference By https://github.com/locuslab/TCN

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import math
from torch.nn import functional as F
from models.attentions import AttentionBlock

class Chomp1d(nn.Module):
    def __init__(self, chomp_size, 
                n_outputs= None,
                batch_norm= False, 
                symm_chomp=False):
        # Causal Convolutions
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        self.batch_norm = batch_norm
        self.n_outputs = n_outputs
        if self.batch_norm: #Custom Part
            self.bn = nn.BatchNorm1d(self.n_outputs)
            
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    
    def forward(self, x):
        if self.batch_norm:
            x = self.bn(x)
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module): #Residual Block
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, 
                batch_norm= False,
                dropout= 0.2,
                non_linear= 'relu'):
        super(TemporalBlock, self).__init__()
        
        # DepthWise first
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding, n_outputs=n_outputs, batch_norm=batch_norm)
        self.non_li1 = nn.ReLU() if non_linear == 'relu' else nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding, n_outputs=n_outputs, batch_norm=batch_norm)
        self.non_li2 = nn.ReLU() if non_linear == 'relu' else nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
            
        
        self.net = nn.Sequential(
            self.conv1, 
            self.chomp1,
            self.non_li1,
            self.dropout1,
            
            self.conv2, 
            self.chomp2, 
            self.non_li2, 
            self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        #nn.init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
        self.conv2.weight.data.normal_(0, 0.01)
        #nn.init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            #nn.init.xavier_uniform(self.downsample.weight, gain=np.sqrt(2))

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) #residual block

from typing import List

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs:int, num_channels:List[int]=[3, 5, 7], 
                kernel_size:int=3, dropout:float= 0.2, 
                batch_norm=False, 
                max_length:int= 200, attention:bool= False, non_linear:str= 'relu'):
        super(TemporalConvNet, self).__init__()
        layers = []
        # num_levels = len(num_channels)
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            
            # out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, 
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, 
                                    batch_norm=batch_norm, non_linear=non_linear)]
            if attention:
                layers += [AttentionBlock(max_length, max_length, max_length)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, in_channel, out_channel, num_channels, 
                kernel_size=3, dropout=0.2, 
                batch_norm=False, 
                max_length:int= 200, attention:bool= False, non_linear:str= 'relu'):
        super(TCN, self).__init__()
        """
            Temporal Convolutional Network (TCN) https://arxiv.org/pdf/1803.01271.pdf
        """
        self.tcn = TemporalConvNet(in_channel, num_channels, kernel_size, dropout=dropout,
                                    max_length= max_length,
                                    batch_norm=batch_norm,
                                    attention= attention,
                                    non_linear= non_linear)
        self.linear = nn.Linear(num_channels[-1], out_channel)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """Input ought to have dimension # (Batch, Channels, Length), 
            where L_in is the seq_len; here the input is (N, L, C)"""
        
        # take the last output
        x = self.tcn(x).transpose(1, 2)
        x = self.linear(x)
        return F.log_softmax(x, dim = 1)
    