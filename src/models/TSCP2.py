
from torch import nn
from typing import List
from models.tcn import TemporalConvNet
import torch    

class Flatten(nn.Module):
    def __init__(self, feature_size):
        super(Flatten, self).__init__()
        self.fs = feature_size

    def forward(self, x):
        bs, ws, fc = x.shape[:3]
        x = x.reshape(bs, ws*fc)
        
        return x 

class TSCP(nn.Module):
    def __init__(self,
                #Prediction/Projection Head
                input_size:int, 
                num_channels:List[int]= [1, 32, 64],
                #Encoder part
                output_size:int=20, 
                hidden_dim:int=100, 
                kernel_size= 4, 
                dropout= 0.2, 
                batch_norm=False,
                attention=False, non_linear= 'relu'):
        super(TSCP, self).__init__()
        self.window_size = hidden_dim
        self.fs = num_channels[-1]
        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size, 
                            dropout=dropout, window_size=hidden_dim,
                            attention=attention)
        
        #bs, ws, 64
        self.out_place = output_size
        
        self.encoder = nn.Sequential(*[
            Flatten(self.window_size),
            nn.Linear(self.window_size*self.fs,
                    hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,
                    hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2,
                    self.out_place)
            ])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = self.tcn(x).transpose(1, 2)
        x = self.encoder(x)
        return x
