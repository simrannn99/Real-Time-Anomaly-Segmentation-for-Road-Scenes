"""
Paper:      ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
Url:        https://arxiv.org/abs/1606.02147
Create by:  zh320
Date:       2023/04/15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Regular convolution with kernel size 3x3
def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                    padding=1, bias=bias)


# Regular convolution with kernel size 1x1, a.k.a. point-wise convolution
def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, 
                    padding=0, bias=bias)


def channel_shuffle(x, groups=2):
    # Codes are borrowed from 
    # https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


# Depth-wise seperable convolution with batchnorm and activation
class DSConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                    dilation=1, act_type='relu', **kwargs):
        super(DSConvBNAct, self).__init__(
            DWConvBNAct(in_channels, in_channels, kernel_size, stride, dilation, act_type, **kwargs),
            PWConvBNAct(in_channels, out_channels, act_type, **kwargs)
        )


# Depth-wise convolution -> batchnorm -> activation
class DWConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                    dilation=1, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation

        super(DWConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                        dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


# Point-wise convolution -> batchnorm -> activation
class PWConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type='relu', bias=True, **kwargs):
        super(PWConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


# Regular convolution -> batchnorm -> activation
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                    bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation

        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


# Transposed /de- convolution -> batchnorm -> activation
class DeConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=None, 
                    padding=None, act_type='relu', **kwargs):
        super(DeConvBNAct, self).__init__()
        if kernel_size is None:
            kernel_size = 2*scale_factor - 1
        if padding is None:    
            padding = (kernel_size - 1) // 2
        output_padding = scale_factor - 1
        self.up_conv = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels, out_channels, 
                                                        kernel_size=kernel_size, 
                                                        stride=scale_factor, padding=padding, 
                                                        output_padding=output_padding),
                                    nn.BatchNorm2d(out_channels),
                                    Activation(act_type, **kwargs)
                                    )

    def forward(self, x):
        return self.up_conv(x)


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
        activation_hub = {'relu': nn.ReLU,             'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU,    'prelu': nn.PReLU,
                          'celu': nn.CELU,              'elu': nn.ELU, 
                          'hardswish': nn.Hardswish,    'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU,              'glu': nn.GLU, 
                          'selu': nn.SELU,              'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid,        'softmax': nn.Softmax, 
                          'tanh': nn.Tanh,              'none': nn.Identity,
                        }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, pool_sizes=[1,2,4,6], bias=False):
        super(PyramidPoolingModule, self).__init__()
        assert len(pool_sizes) == 4, 'Length of pool size should be 4.\n'
        hid_channels = int(in_channels // 4)
        self.stage1 = self._make_stage(in_channels, hid_channels, pool_sizes[0])
        self.stage2 = self._make_stage(in_channels, hid_channels, pool_sizes[1])
        self.stage3 = self._make_stage(in_channels, hid_channels, pool_sizes[2])
        self.stage4 = self._make_stage(in_channels, hid_channels, pool_sizes[3])
        self.conv = PWConvBNAct(2*in_channels, out_channels, act_type=act_type, bias=bias)

    def _make_stage(self, in_channels, out_channels, pool_size):
        return nn.Sequential(
                        nn.AdaptiveAvgPool2d(pool_size),
                        conv1x1(in_channels, out_channels)
                )

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.stage1(x), size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.stage2(x), size, mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.stage3(x), size, mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.stage4(x), size, mode='bilinear', align_corners=True)
        x = self.conv(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x


class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type, hid_channels=128):
        super(SegHead, self).__init__(
            ConvBNAct(in_channels, hid_channels, 3, act_type=act_type),
            conv1x1(hid_channels, num_class)
        )

class ENet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='prelu', 
                    upsample_type='deconvolution'):
        super(ENet, self).__init__()
        self.initial = InitialBlock(n_channel, 16, act_type)
        self.bottleneck1 = BottleNeck1(16, 64, act_type)
        self.bottleneck2 = BottleNeck23(64, 128, act_type, True)
        self.bottleneck3 = BottleNeck23(128, 128, act_type, False)
        self.bottleneck4 = BottleNeck45(128, 64, act_type, upsample_type, True)
        self.bottleneck5 = BottleNeck45(64, 16, act_type, upsample_type, False)
        self.fullconv = Upsample(16, num_class, scale_factor=2, act_type=act_type)
        
    def forward(self, x):
        x = self.initial(x)
        x, indices1 = self.bottleneck1(x)     # 2x downsample
        x, indices2 = self.bottleneck2(x)     # 2x downsample
        x = self.bottleneck3(x)
        x = self.bottleneck4(x, indices2)     # 2x upsample
        x = self.bottleneck5(x, indices1)     # 2x upsample
        x = self.fullconv(x)
        
        return x
 

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, kernel_size=3, **kwargs):
        super(InitialBlock, self).__init__()
        assert out_channels > in_channels, 'out_channels should be larger than in_channels.\n'
        self.conv = ConvBNAct(in_channels, out_channels - in_channels, kernel_size, 2, act_type=act_type, **kwargs)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], dim=1)

        return x        


class BottleNeck1(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='prelu', drop_p=0.01):
        super(BottleNeck1, self).__init__()
        self.conv_pool = Bottleneck(in_channels, out_channels, 'downsampling', act_type, drop_p=drop_p)
        self.conv_regular = nn.Sequential(
            Bottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
            Bottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
            Bottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
            Bottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
        )
        
    def forward(self, x):
        x, indices = self.conv_pool(x)
        x = self.conv_regular(x)
        
        return x, indices
        
        
class BottleNeck23(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='prelu', downsample=True):
        super(BottleNeck23, self).__init__()
        self.downsample = downsample
        if downsample:
            self.conv_pool = Bottleneck(in_channels, out_channels, 'downsampling', act_type=act_type)

        self.conv_regular = nn.Sequential(
            Bottleneck(out_channels, out_channels, 'regular', act_type),
            Bottleneck(out_channels, out_channels, 'dilate', act_type, dilation=2),
            Bottleneck(out_channels, out_channels, 'asymmetric', act_type),
            Bottleneck(out_channels, out_channels, 'dilate', act_type, dilation=4),
            Bottleneck(out_channels, out_channels, 'regular', act_type),
            Bottleneck(out_channels, out_channels, 'dilate', act_type, dilation=8),
            Bottleneck(out_channels, out_channels, 'asymmetric', act_type),
            Bottleneck(out_channels, out_channels, 'dilate', act_type, dilation=16),
        )
    
    def forward(self, x):
        if self.downsample:
            x, indices = self.conv_pool(x)
        x = self.conv_regular(x)
        
        if self.downsample:
            return x, indices
            
        return x
        
        
class BottleNeck45(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='prelu', upsample_type=None, 
                    extra_conv=False):
        super(BottleNeck45, self).__init__()
        self.extra_conv = extra_conv
        self.conv_unpool = Bottleneck(in_channels, out_channels, 'upsampling', act_type, upsample_type)
        self.conv_regular = Bottleneck(out_channels, out_channels, 'regular', act_type)

        if extra_conv:
            self.conv_extra = Bottleneck(out_channels, out_channels, 'regular', act_type)

    def forward(self, x, indices):
        x = self.conv_unpool(x, indices)
        x = self.conv_regular(x)
        
        if self.extra_conv:
            x = self.conv_extra(x)
            
        return x

  
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type, act_type='prelu', 
                    upsample_type='regular', dilation=1, drop_p=0.1, shrink_ratio=0.25):
        super(Bottleneck, self).__init__()
        self.conv_type = conv_type
        hid_channels = int(in_channels * shrink_ratio)
        if conv_type == 'regular':
            self.right_init_conv = nn.Sequential(
                                    ConvBNAct(in_channels, hid_channels, 1),
                                    ConvBNAct(hid_channels, hid_channels),
                                )
        elif conv_type == 'downsampling':
            self.left_pool = nn.MaxPool2d(2, 2, return_indices=True)
            self.left_conv = ConvBNAct(in_channels, out_channels, 1)         
            self.right_init_conv = nn.Sequential(
                                    ConvBNAct(in_channels, hid_channels, 3, 2),
                                    ConvBNAct(hid_channels, hid_channels),
                                )
        elif conv_type == 'upsampling':
            self.left_conv = ConvBNAct(in_channels, out_channels, 1)
            self.left_pool = nn.MaxUnpool2d(2, 2)
            self.right_init_conv = nn.Sequential(
                                    ConvBNAct(in_channels, hid_channels, 1),
                                    Upsample(hid_channels, hid_channels, scale_factor=2,  
                                                kernel_size=3, upsample_type=upsample_type),
                                )
        elif conv_type == 'dilate':
            self.right_init_conv = nn.Sequential(
                                    ConvBNAct(in_channels, hid_channels, 1),
                                    ConvBNAct(hid_channels, hid_channels, dilation=dilation),
                                )
        elif conv_type == 'asymmetric':
            self.right_init_conv = nn.Sequential(
                                    ConvBNAct(in_channels, hid_channels, 1),
                                    ConvBNAct(hid_channels, hid_channels, (5,1)),
                                    ConvBNAct(hid_channels, hid_channels, (1,5)),
                                )
        else:
            raise ValueError(f'[!] Unsupport convolution type: {conv_type}')
                        
        self.right_last_conv = nn.Sequential(
                                    conv1x1(hid_channels, out_channels),
                                    nn.Dropout(drop_p)
                            )
        self.act = Activation(act_type)

    def forward(self, x, indices=None):
        x_right = self.right_last_conv(self.right_init_conv(x))
        if self.conv_type == 'downsampling':
            x_left, indices = self.left_pool(x)
            x_left = self.left_conv(x_left)
            x = self.act(x_left + x_right)
            return x, indices
            
        elif self.conv_type == 'upsampling':
            if indices is None:
                raise ValueError('Upsampling-type conv needs pooling indices.')
            
            x_left = self.left_conv(x)
            x_left = self.left_pool(x_left, indices)
            x = self.act(x_left + x_right)
            
        else:
            x = self.act(x + x_right)    # shortcut
            
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=None, 
                    upsample_type=None, act_type='relu'):
        super(Upsample, self).__init__()
        if upsample_type == 'deconvolution':
            if kernel_size is None:
                kernel_size = 2*scale_factor - 1
            padding = (kernel_size - 1) // 2
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                stride=scale_factor, padding=padding, 
                                                output_padding=1, bias=False)
        else:
            self.up_conv = nn.Sequential(
                                    ConvBNAct(in_channels, out_channels, 1, act_type=act_type),
                                    nn.Upsample(scale_factor=scale_factor, mode='bilinear')
                            )

    def forward(self, x):
        return self.up_conv(x)