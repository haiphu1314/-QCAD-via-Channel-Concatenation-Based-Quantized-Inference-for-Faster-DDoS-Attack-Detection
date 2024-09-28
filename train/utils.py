import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import pandas as pd
import numpy as np
# from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys
def _ternary(x: torch.Tensor, delta: float):
    return (x >= delta).float() - (x <= -delta).float()

class _ternary_py(torch.autograd.Function):
    @staticmethod
    def ternary_backward(grad_output: torch.Tensor, x: torch.Tensor, delta: float, order: int, threshold: float):
        scale = 2 * delta
        assert threshold <= scale
        tmp = torch.zeros_like(grad_output)
        tmp += ((x >= -threshold) & (x <= threshold)).float() * order * \
               (torch.fmod(x / delta + 3, 2) - 1).abs().pow(order - 1)
        return grad_output * tmp

    @staticmethod
    def forward(ctx, *inputs) -> torch.Tensor:
        input_f, running_delta, delta, momentum, training, ctx.order = inputs
        if momentum > 0:
            if training:
                ctx.delta = input_f.norm(1).item() * (delta / input_f.numel())  # = delta * |input_f|_1 / n
                running_delta.data = momentum * ctx.delta + (1.0 - momentum) * running_delta.data
            else:
                ctx.delta = running_delta.data.item()
        else:
            ctx.delta = delta
        # input_t = _ternary(input_f, ctx.delta) * (2 * ctx.delta)
        input_t = _ternary(input_f, ctx.delta)
        ctx.save_for_backward(input_f)
        return input_t

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        input_f, = ctx.saved_tensors
        grad_input = _ternary_py.ternary_backward(grad_output, input_f, ctx.delta, ctx.order, 2. * ctx.delta)
        return grad_input, None, None, None, None, None, None, None, None, None


def ternary(input_f: torch.Tensor, running_delta, delta, momentum, training, order):
    return _ternary_py.apply(input_f, running_delta, delta, momentum, training, order)


class Ternary(torch.nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        config = {}
        self.config = config
        self.delta = config.setdefault("delta", 0.5)
        self.momentum = config.setdefault("momentum", 0.01)
        self.track_running_stats = config.setdefault("track_running_stats", True)
        self.order = config.setdefault('order', 2)
        # self.use_scale = config.setdefault('use_scale', True)
        assert self.momentum <= 1 and self.order > 0 and self.delta > 0
        self.register_buffer("running_delta", torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.momentum > 0:
            self.running_delta.fill_(self.delta * 0.7979)
        else:
            self.running_delta.fill_(self.delta)

    def forward(self, input_f):
        return ternary(input_f, self.running_delta, self.delta, self.momentum,
                       self.training and self.track_running_stats, self.order)

    def extra_repr(self):
        return ", ".join(["{}={}".format(k, v) for k, v in self.config.items()])


def _binary(x: torch.Tensor, delta: float):
    return (x >= delta).float() - (x < delta).float()

class _binary_py(torch.autograd.Function):
    @staticmethod
    def binary_backward(grad_output: torch.Tensor, x: torch.Tensor, delta: float):
        tmp = torch.zeros_like(grad_output)
        tmp += ((x >= -delta) & (x <= delta)).float() * (1 - (torch.abs(x) / delta))
        return grad_output * tmp

    @staticmethod
    def forward(ctx, *inputs) -> torch.Tensor:
        input_f, running_delta, delta, momentum, training = inputs
        if momentum > 0:
            if training:
                ctx.delta = input_f.norm(1).item() * (delta / input_f.numel())  # = delta * |input_f|_1 / n
                running_delta.data = momentum * ctx.delta + (1.0 - momentum) * running_delta.data
            else:
                ctx.delta = running_delta.data.item()
        else:
            ctx.delta = delta
        input_b = _binary(input_f, ctx.delta)
        ctx.save_for_backward(input_f)
        return input_b

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        input_f, = ctx.saved_tensors
        grad_input = _binary_py.binary_backward(grad_output, input_f, ctx.delta)
        return grad_input, None, None, None, None, None


def binary(input_f: torch.Tensor, running_delta, delta, momentum, training):
    return _binary_py.apply(input_f, running_delta, delta, momentum, training)


class Binary(torch.nn.Module):
    def __init__(self):
        super(Binary, self).__init__()
        config = {}
        self.config = config
        self.delta = config.setdefault("delta", 0.5)
        self.momentum = config.setdefault("momentum", 0.01)
        self.track_running_stats = config.setdefault("track_running_stats", True)
        assert self.momentum <= 1 and self.delta > 0
        self.register_buffer("running_delta", torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.momentum > 0:
            self.running_delta.fill_(self.delta * 0.7979)
        else:
            self.running_delta.fill_(self.delta)

    def forward(self, input_f):
        return binary(input_f, self.running_delta, self.delta, self.momentum,
                      self.training and self.track_running_stats)

    def extra_repr(self):
        return ", ".join(["{}={}".format(k, v) for k, v in self.config.items()])
    
    
class QConv2d(torch.nn.Conv2d):
    qa_config = {}
    qw_config = {}

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', quant = 'BNN'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        if quant == 'BNN':
            self.input_quantizer = Binary()
            self.weight_quantizer = Binary()
        elif quant == 'TNN':
            self.input_quantizer = Ternary()
            self.weight_quantizer = Ternary()
        elif quant == 'TBN':
            self.input_quantizer = Ternary()
            self.weight_quantizer = Binary()    

    def forward(self, input_f):
        input_t = self.input_quantizer(input_f)
        # print(input_t)
        weight_b = self.weight_quantizer(self.weight)
        out = F.conv2d(input_t, weight_b, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class QLinear(torch.nn.Linear):
    qa_config = {}
    qw_config = {}
    def __init__(self, in_features, out_features, bias=False, quant = 'BNN'):
        super().__init__(in_features, out_features, bias)
        if quant == 'BNN':
            self.input_quantizer = Binary()
            self.weight_quantizer = Binary()
        elif quant == 'TNN':
            self.input_quantizer = Ternary()
            self.weight_quantizer = Ternary()
        elif quant == 'TBN':
            self.input_quantizer = Ternary()
            self.weight_quantizer = Binary()    
    def forward(self, input_f):
        input_b = self.input_quantizer(input_f)
        weight_b = self.weight_quantizer(self.weight)
        out = F.linear(input_b, weight_b, self.bias)
        return out

class QConvTranspose2d(nn.ConvTranspose2d):
    qa_config = {}
    qw_config = {}

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', quant='BNN'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        
        if quant == 'BNN':
            self.input_quantizer = Binary()
            self.weight_quantizer = Binary()
        elif quant == 'TNN':
            self.input_quantizer = Ternary()
            self.weight_quantizer = Ternary()
        elif quant == 'TBN':
            self.input_quantizer = Ternary()
            self.weight_quantizer = Binary()    

    def forward(self, input_f):
        input_t = self.input_quantizer(input_f)
        weight_b = self.weight_quantizer(self.weight)
        out = F.conv_transpose2d(input_t, weight_b, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        return out
    

    
def convert_and_concatenate(arr):
    arr = torch.flip(arr, dims=[0])
    converted_arr = [1 if x == -1 else 0 for x in arr]
    concatenated_str = ''.join(map(str, converted_arr))

    current_length = len(concatenated_str)
    total_length = (np.ceil(current_length/32)*32).astype(int)
    if current_length < total_length:
        padding_length = int(total_length - current_length)
        concatenated_str = '0' * padding_length + concatenated_str
    hex_value = hex(int(concatenated_str, 2))[2:] 

    hex_length = total_length // 4  
    if len(hex_value) < hex_length:
        hex_value = '0' * (hex_length - len(hex_value)) + hex_value
    return hex_value

def convert_and_concatenate_tnn(arr):
    arr = torch.flip(arr, dims=[0])
    # Chuyển đổi -1 thành 1 và 1 thành 0
    converted_bit0_arr = [1 if x == -1 else 0 for x in arr]
    converted_bit1_arr = [1 if x == 1 else 0 for x in arr]
    concatenated_bit0_str = ''.join(map(str, converted_bit0_arr))
    concatenated_bit1_str = ''.join(map(str, converted_bit1_arr))


    current_length = np.max([len(concatenated_bit0_str),len(concatenated_bit0_str)])
    total_length = (np.ceil(current_length/32)*32).astype(int)
    if current_length < total_length:
        padding_length0 = int(total_length - len(concatenated_bit0_str))
        padding_length1 = int(total_length - len(concatenated_bit1_str))
        concatenated_bit0_str = '0' * padding_length0 + concatenated_bit0_str
        concatenated_bit1_str = '0' * padding_length1 + concatenated_bit1_str

    # Chuyển chuỗi thành số nhị phân và sau đó thành số hex
    hex_value_bit0 = hex(int(concatenated_bit0_str, 2))[2:]  # Bỏ tiền tố '0x'
    hex_value_bit1 = hex(int(concatenated_bit1_str, 2))[2:]  # Bỏ tiền tố '0x'
    # Đảm bảo chuỗi hex đủ độ dài cần thiết
    if(len(hex_value_bit0)>len(hex_value_bit1)):
        hex_value_bit1 = '0' * (len(hex_value_bit0)-len(hex_value_bit1)) + hex_value_bit1
    elif(len(hex_value_bit0)<len(hex_value_bit1)):
        hex_value_bit0 = '0' * (len(hex_value_bit1)-len(hex_value_bit0)) + hex_value_bit0

    hex_length = total_length // 4  # 1 hex digit = 4 bits
    if len(hex_value_bit0) < hex_length:
        hex_value_bit0 = '0' * (hex_length - len(hex_value_bit0)) + hex_value_bit0
        hex_value_bit1 = '0' * (hex_length - len(hex_value_bit1)) + hex_value_bit1
    return hex_value_bit0, hex_value_bit1

def to_hex(x):
    return f"0x{x:08x}"

def save_model_parameters_to_txt(model, file_path):
    input_thres = []
    weight_thres = []
    quant_type = []
    i_q = '0'
    w_q = '0'
    for name, module in model.named_modules():        
        if 'input_quantizer' in name:
            if 'Binary' in str(module):
                input_thres = np.append(input_thres,0.0)
                i_q = 'B'
            else:
                input_thres = np.append(input_thres, module.running_delta.item())
                i_q = 'T'
        elif 'weight_quantizer' in name:
            if 'Binary' in str(module):
                weight_thres = np.append(weight_thres, 0.0)
                w_q = 'B'
            else:
                weight_thres = np.append(weight_thres, module.running_delta.item())
                w_q = 'T'
        if i_q != '0' and w_q != '0':
            if i_q == 'B' and w_q == 'B':
                quant_type = np.append(quant_type,'BNN')
            elif i_q == 'T' and w_q == 'B':
                quant_type = np.append(quant_type,'TBN')
            elif i_q == 'T' and w_q == 'T':
                quant_type = np.append(quant_type,'TNN')
            i_q = '0'
            w_q = '0'
    cnt=0
    with open(file_path, 'w') as f:
        for name, param in model.named_parameters():
            layer_name = name.split('.')[0]
            if 'conv' in name or 'dconv' in name or 'tconv' in name:
                f.write(f'{name}\n')
                f.write(f'layer_name: {layer_name}\n')
                f.write(f'input_channel: {param.shape[1]}\n')
                f.write(f'output_channel: {param.shape[0]}\n')
                f.write(f'kernel_size: 3\n')
                f.write(f'stride: 1\n')
                f.write(f'padding: 1\n')
                f.write(f'dilation: 1\n')
                if(quant_type[cnt] == 'BNN'):
                    f.write(f'quant_type: 0\n')
                    quant_write = 1
                elif(quant_type[cnt] == 'TBN'):
                    f.write(f'quant_type: 1\n')
                    quant_write = 1
                elif(quant_type[cnt] == 'TNN'):
                    f.write(f'quant_type: 2\n')
                    quant_write = 2
                f.write(f'input_thres: {input_thres[cnt]}\n')
                namen = name.split('.')[0]
                # param = getattr(getattr(model, namen).weight_quantizer, 'forward')(param)
                param = _ternary(param, weight_thres[cnt])
                for param_e in param:
                    # param_e = getattr(getattr(model, namen).weight_quantizer, 'forward')(param_e)
                    # param_e = _ternary(param_e, weight_thres[cnt])
                    param_padding = torch.zeros(32*int(np.ceil(param_e.shape[0]/32)), param_e.shape[1],param_e.shape[2])
                    param_padding[0:param_e.shape[0],:,:] = param_e
                    param_e = param_padding
                    sichannel = int(np.ceil(param_e.shape[0]/32))
                    for c in range(sichannel):
                        for q in range(quant_write):
                            for i in range(param_e.shape[1]):
                                for j in range(param_e.shape[2]):
                                    if(quant_type[cnt] == 'BNN' or quant_type[cnt] == 'TBN'):
                                        hex_conv = convert_and_concatenate(_binary(param_e[(c+1)*32-32:(c+1)*32, i, j]))
                                        if j == param_e.shape[2]-1:
                                            f.write(f'0x{hex_conv}')
                                        else:
                                            f.write(f'0x{hex_conv}, ')
                                    elif(quant_type[cnt] == 'TNN'):
                                        hex_conv_bit0, hex_conv_bit1 = convert_and_concatenate_tnn(param_e[(c+1)*32-32:(c+1)*32, i, j])
                                        if(q == 0):
                                            if j == param_e.shape[2]-1:
                                                f.write(f'0x{hex_conv_bit0}')
                                            else:
                                                f.write(f'0x{hex_conv_bit0}, ')
                                        elif(q == 1):
                                            if j == param_e.shape[2]-1:
                                                f.write(f'0x{hex_conv_bit1}')
                                            else:
                                                f.write(f'0x{hex_conv_bit1}, ')
                                f.write(f'\n')
                        f.write(f'\n')

            elif 'fc' in name or 'linear' in name:
                f.write(f'{name}\n')
                f.write(f'layer_name: {layer_name}\n')
                f.write(f'input_channel: {param.shape[1]}\n')
                f.write(f'output_channel: {param.shape[0]}\n')
                if(quant_type[cnt] == 'BNN'):
                    f.write(f'quant_type: 0\n')
                elif(quant_type[cnt] == 'TBN'):
                    f.write(f'quant_type: 1\n')
                elif(quant_type[cnt] == 'TNN'):
                    f.write(f'quant_type: 2\n')
                f.write(f'input_thres: {input_thres[cnt]}\n')

                if(quant_type[cnt] == 'BNN' or quant_type[cnt] == 'TBN'):
                    param = _binary(param)
                    for param_e in param:
                        hex_conv = convert_and_concatenate(param_e)
                        segments = [hex_conv[i:i+8] for i in range(0, len(hex_conv), 8)]
                        segments.reverse()
                        for segment in segments:
                            f.write(f'0x{segment}\n')
                    f.write(f'\n')
                elif(quant_type[cnt] == 'TNN'):
                    # param = getattr(getattr(model, namen).weight_quantizer, 'forward')(param)
                    param = _ternary(param, weight_thres[cnt])
                    # print(name)
                    for param_e in param:
                        hex_conv_bit0, hex_conv_bit1 = convert_and_concatenate_tnn(param_e)
                        segments0 = [hex_conv_bit0[i:i+8] for i in range(0, len(hex_conv_bit0), 8)]
                        segments1 = [hex_conv_bit1[i:i+8] for i in range(0, len(hex_conv_bit1), 8)]
                        segments0.reverse()
                        segments1.reverse()
                        for j in range (len(segments0)):
                            f.write(f'0x{segments0[j]}\n')
                            f.write(f'0x{segments1[j]}\n')
                    f.write(f'\n')
            cnt+=1        
        return 0

