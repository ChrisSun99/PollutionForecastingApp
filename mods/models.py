# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

pytorch的神经网络模型
"""
import sys
import numpy as np
import torch
from torch import nn
sys.path.append('../')
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            print(x.shape)
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)

class ConvNet(nn.Module):
    """cnn模型"""
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(6144, 6144)
        self.fc2 = nn.Linear(6144, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class LSTM(nn.Module):
    """lstm模型"""

    def __init__(self, input_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = int(np.floor(input_size))
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.connect_0 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.connect_0(x)
        return x


def initialize_lstm_params(lstm):
    """
    初始化模型参数
    :param lstm: nn.LSTM(), 神经网络模型
    :return:
        lstm, nn.LSTM(), 参数初始化后的神经网络模型
    """
    lstm.connect_0.weight.data = torch.rand(1, lstm.hidden_size)  # attention: 注意shape是转置关系
    lstm.connect_0.bias.data = torch.rand(1)
    return lstm


class AlphaLayer(nn.Module):
    """alpha层"""

    def __init__(self, features_num):
        super(AlphaLayer, self).__init__()
        self.features_num = features_num
        self.output_size = 1

        self.connect_0 = nn.Linear(self.features_num, self.output_size)
        self.act_0 = nn.ReLU()

    def forward(self, x):
        x = self.connect_0(x)
        input_weight = self.act_0(x)
        return input_weight


def initialize_alpha_layer_params(alpha_layer):
    """
    初始化模型参数
    :param alpha_layer: nn.NN(), alpha层
    :return:
        alpha_layer, nn.NN(), 参数初始化后的神经网络模型
    """
    alpha_layer.connect_0.weight.data = torch.rand(alpha_layer.output_size,
                                                   alpha_layer.features_num)  # attention: 注意shape是转置关系
    alpha_layer.connect_0.bias.data = torch.rand(alpha_layer.output_size)
    return alpha_layer


class WeightsLayer(nn.Module):
    """权重层"""

    def __init__(self, weights_num, output_size):
        super(WeightsLayer, self).__init__()
        self.weights_num = weights_num
        self.output_size = output_size

        self.connect_0 = nn.Linear(self.weights_num, int(self.weights_num / 2))
        self.act_0 = nn.Tanh()
        self.connect_1 = nn.Linear(int(self.weights_num / 2), self.output_size)
        self.act_1 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.connect_0(x)
        x = self.act_0(x)
        x = self.connect_1(x)
        x = self.act_1(x)
        output_weight = self.softmax(x)
        return output_weight


def initialize_weights_layer_params(weights_layer):
    """
    初始化模型参数
    :param weights_layer: nn.NN(), weights_layer
    :return:
        weights_layer, nn.NN(), 参数初始化后的神经网络模型
    """
    weights_layer.connect_0.weight.data = torch.rand(int(weights_layer.weights_num / 2),
                                                     weights_layer.weights_num)  # attention: 注意shape是转置关系
    weights_layer.connect_0.bias.data = torch.rand(int(weights_layer.weights_num / 2))
    weights_layer.connect_1.weight.data = torch.rand(weights_layer.output_size,
                                                     int(weights_layer.weights_num / 2))  # attention: 注意shape是转置关系
    weights_layer.connect_1.bias.data = torch.rand(weights_layer.output_size)
    return weights_layer

# import json
# time = '2019-06-12'
# num = 2
# request_dict = {'time': time, 'num': num}
# request_js = json.dumps(request_dict)
# with open('../tmp/request.pkl', 'w') as f:
# 	json.dump(request_dict, f)
#
# request_dict_2 = json.loads(request_js)
#
# with open('../tmp/request.pkl', 'r') as f:
# 	request_dict_3 = json.load(f)
#
