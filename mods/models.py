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
from torch.autograd import Variable

sys.path.append('../')


class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, cell, num_layers, use_cuda):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = 1
        self.use_cuda = False
        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, dropout=0.0,
                               nonlinearity="tanh", batch_first=True, )
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, dropout=0.0, batch_first=True, )
        self.fc = nn.Linear(self.input_size, self.hidden_size)


class RNN(BaseModel):
    def __init__(self, input_size, hidden_size, output_size, cell, num_layers, use_cuda):
        super(RNN, self).__init__(input_size, hidden_size, output_size, cell, num_layers, use_cuda)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.num_layers * 1, batch_size, self.hidden_size))
        # if self.use_cuda:
        #     h0 = h0.cuda()
        rnn_output, hn = self.cell(x, h0)
        hn = hn.view(batch_size, self.hidden_size)
        fc_output = self.fc(hn)

        return fc_output


class GRU(BaseModel):
    def __init__(self, input_size, hidden_size, output_size, cell, num_layers, use_cuda):
        super(GRU, self).__init__(input_size, hidden_size, output_size, cell, num_layers, use_cuda)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.num_layers * 1, batch_size, self.hidden_size))
        # if self.use_cuda:
        #     h0 = h0.cuda()
        rnn_output, hn = self.cell(x, h0)
        hn = hn.view(batch_size, self.hidden_size)
        fc_output = self.fc(hn)

        return fc_output


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
