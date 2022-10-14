import torch
import torch.nn as nn
import math
import numpy as np
from models.resnet1d import ResNet1D, BasicBlock1D
from models.tcn import TCN
class RES_TCN(nn.Module):
    def __init__( self, hidden_dim=100, num_classes=10,
                  relu_type='prelu', tcn_options={},
                  use_boundary=False, extract_feats=False):
        super(RES_TCN, self).__init__()
        self.extract_feats = extract_feats
        self.use_boundary = use_boundary

        self.frontend_nout = 1
        self.backend_out = 512
        self.trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type=relu_type)

        num_channels=[3, 5, 7]
        self.tcn = TCN(in_channel=self.backend_out,
                        num_channels= num_channels,
                        out_channel=num_classes)
    

        # -- initialize
        self._initialize_weights_randomly()


    def forward(self, x, boundaries=None):
        B, C, T = x.size()
        x = self.trunk(x)
        print(x.shape)
        exit()
        x = x.transpose(1, 2)
        lengths = [_//640 for _ in lengths]

        # -- duration
        if self.use_boundary:
            x = torch.cat([x, boundaries], dim=-1)

        return x if self.extract_feats else self.tcn(x, lengths, B)


    def _initialize_weights_randomly(self):
        use_sqrt = True

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