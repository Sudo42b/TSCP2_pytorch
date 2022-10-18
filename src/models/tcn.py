## Reference By https://github.com/locuslab/TCN
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch

from torch.nn import functional as F
from models.attentions import AttentionBlock
class SpatialDropout1D(nn.Module):
    """
    Spatial Dropout drops a certain percentage of dimensions from each word vector in the training sample.
    implementation: https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400;
    explanation: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/76883.
    """

    def __init__(self, p: float):
        super(SpatialDropout1D, self).__init__()
        self.spatial_dropout = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 1, 0).unsqueeze(
            0
        )  # convert to [fake_dim, emb_dim, seq_len, batch_size]
        x = self.spatial_dropout(x)
        x = x.squeeze(0).permute(2, 1, 0)  # back to [batch_size, seq_len, emb_dim]
        return x
class Chomp1d(nn.Module):
    def __init__(self, 
                chomp_size, 
                symm_chomp=False):
        # Causal Convolutions
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp #chomp_size mean zero padding size
            
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module): # Residual Block
    def __init__(self, 
                n_inputs, 
                n_outputs, 
                kernel_size, 
                stride, 
                dilation, 
                padding, 
                batch_norm= True,
                dropout= 0,
                non_linear= 'relu'):
        super(TemporalBlock, self).__init__()
        
        # DepthWise first
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.non_li1 = nn.ReLU() if non_linear == 'relu' else nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
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


    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) #residual block
import numpy as np
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, 
                dropout=0.2, attention=True, window_size=100, non_linear='relu',
                batch_norm=True):
        """
        Args:
            num_inputs: in_channels (width) of input
            num_channels: array-like with length of # layers. starts with first layer's # out_channels
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout,
                                    non_linear=non_linear, batch_norm=batch_norm)]
            if attention:
                layers += [AttentionBlock(window_size, window_size, window_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.init_weights()
        
    def init_weights(self):
        # self.conv1.weight.data.normal_(0, 0.01)
        # #nn.init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
        # self.conv2.weight.data.normal_(0, 0.01)
        # #nn.init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
        # if self.downsample is not None:
        #     self.downsample.weight.data.normal_(0, 0.01)
        #     #nn.init.xavier_uniform(self.downsample.weight, gain=np.sqrt(2))
        use_sqrt = True
        import math
        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))
                
    def forward(self, x):
        return self.network(x)
