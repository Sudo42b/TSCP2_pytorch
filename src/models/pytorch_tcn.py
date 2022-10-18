from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import BatchNorm1d
from torch.nn import LayerNorm
from torch import nn
import numpy as np
from torch.nn import functional as F
from typing import Tuple
import torch
from torch.nn import init
import math

def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations
    
class Lambda(nn.Module):
    def __init__(self, Lambda):
        self._lambda = Lambda
    def forward(self, x):
        return self._lambda(x)
    
class CausalConv1d(nn.Conv1d):
    def __init__(self,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    groups=1,
                    bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result
        
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

class ResidualBlock(Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                dilation_rate: int,
                kernel_size: int,
                stride:int = 1,
                activation: str = 'relu',
                dropout_rate: float = 0,
                kernel_initializer: str = 'he_normal',
                use_batch_norm: bool = False,
                use_layer_norm: bool = False,
                use_weight_norm: bool = False,
                
                **kwargs):
        """Defines the residual block for the WaveNet TCN
        Args:
            x: The previous layer in the model
            training: boolean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using for this residual block
            out_channels: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            kwargs: Any initializers for Layer class.
        """

        self.dilation_rate = dilation_rate
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = kernel_initializer
        
        self.layers = []
        
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = nn.Conv1d(in_channels= in_channels, 
                                out_channels= out_channels, 
                                kernel_size=self.kernel_size,
                                stride=stride,
                                #padding=self.padding, 
                                dilation=self.dilation_rate)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.chomp1 = CausalConv1d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size= self.kernel_size, dilation=self.dilation_rate)
        self.act1 = nn.ReLU() if use_layer_norm == 'relu' else nn.GELU()
        self.dropout1 = SpatialDropout1D(p=self.dropout_rate)

        self.conv2 = nn.Conv1d(in_channels= out_channels, 
                                out_channels= out_channels, 
                                kernel_size=self.kernel_size,
                                stride=stride,
                                dilation=self.dilation_rate)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.chomp2 = CausalConv1d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size= self.kernel_size, dilation=self.dilation_rate)
        self.act2 = nn.ReLU() if use_layer_norm == 'relu' else nn.GELU()
        self.dropout2 = SpatialDropout1D(p=self.dropout_rate)
        
        self.net = nn.Sequential(self.conv1,
                                self.chomp1,
                                self.act1,
                                self.dropout1,
                                
                                self.conv2, 
                                self.chomp2, 
                                self.act2, 
                                self.dropout2)
        ### 'matching_conv1D' or 'matching_identity'
        # 1x1 conv to match the shapes (channel dimension).
        # make and build this layer separately because it directly uses input_shape.
        # 1x1 conv.
        self.shape_match_conv = nn.Conv1d(self.out_channels, 
                                    self.out_channels,
                                    kernel_size=1) if in_channels != out_channels else None
        
        self.final_activation = nn.ReLU(True) if use_layer_norm == 'relu' else nn.GELU()
        self.init_weights()
        
    def forward(self, x):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                is the skip connection tensor.
        """
        # https://arxiv.org/pdf/1803.01271.pdf  page 4, Figure 1 (b).
        # x1: Dilated Conv -> Norm -> Dropout (x2).
        # x2: Residual (1x1 matching conv - optional).
        # Output: x1 + x2.
        # x1 -> connected to skip connections.
        # x1 + x2 -> connected to the next block.
        #       input
        #     x1      x2
        #   conv1D    1x1 Conv1D (optional)
        #    ...
        #   conv1D
        #    ...
        #       x1 + x2
        x1 = x
        x1 = self.net(x)
        if self.shape_match_conv:
            x2 = self.shape_match_conv(x)
            return self.final_activation(x2)
        else:
            x2_x1 = self.final_activation(x2+x1)
            return x2_x1

    def init_weights(self):
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

class TCN(Module):
    """Creates a TCN layer.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional layers. Can be a list.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            kwargs: Any other arguments for configuring parent class Layer. For example "name=str", Name of the model.
                    Use unique names when using multiple TCN.

        Returns:
            A TCN layer.
        """

    def __init__(self,
                    in_channels:int,
                    out_channels:int= 64,
                    kernel_size=3,
                    nb_stacks=1,
                    dilations=(1, 2, 4, 8, 16, 32),
                    use_skip_connections=True,
                    dropout_rate=0.0,
                    return_sequences=False,
                    activation='relu',
                    kernel_initializer='he_normal',
                    use_batch_norm=False,
                    use_layer_norm=False,
                    use_weight_norm=False,
                    go_backwards=False,
                    return_state=False,
                    attention=False):
        # initialize parent class
        super(TCN, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.activation_name = activation
        # self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.return_sequences = return_sequences
        self.use_skip_connections = use_skip_connections
        self.nb_stacks = nb_stacks
        self.use_batch_norm = use_batch_norm
        self.use_weight_norm = use_weight_norm
        self.return_state = return_state
        self.go_backwards = go_backwards
        
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None  # in case return_sequence=False
        self.output_slice_index = None  # in case return_sequence=False
        self.padding_same_and_time_dim_unknown = False  # edge case if padding='same' and time_dim = None
        
        # list to hold all the member ResidualBlocks
        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below
            
        if self.use_batch_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')

        if isinstance(self.out_channels, list):
            assert len(self.out_channels) == len(self.dilations)
            if len(set(self.out_channels)) > 1 and self.use_skip_connections:
                raise ValueError('Skip connections are not compatible '
                                'with a list of filters, unless they are all equal.')

        # if padding != 'causal' and padding != 'same':
        #     raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")
        
        # member to hold current output shape of the layer for building purposes
        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                res_block_filters = self.out_channels[i] if isinstance(self.out_channels, list) else self.out_channels
                self.residual_blocks.append(ResidualBlock(in_channels= self.in_channels,
                                                            out_channels= res_block_filters,
                                                            dilation_rate= d,
                                                            kernel_size= self.kernel_size,
                                                            activation= self.activation_name,
                                                            dropout_rate= self.dropout_rate,
                                                            kernel_initializer= self.kernel_initializer))
                if self.use_skip_connections:
                    self.downsample = nn.Conv1d(self.in_channels, res_block_filters, 1) if self.in_channels != res_block_filters else None
                    self.residual_blocks.append(self.downsample)
                    self.slicer_layer = Lambda(lambda tt: tt[:, -1, :]) # -1 causal case.
                    self.residual_blocks.append(self.slicer_layer)
        self.net = nn.Sequential(*self.residual_blocks)
            
    @property
    def receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * self.nb_stacks * sum(self.dilations)

    def forward(self, x):
        
        self.skip_connections = []
        if self.go_backwards:
            # reverse x in the time axis
            x = torch.flip(x, dim=1)
        x = self.net(x)

        return x