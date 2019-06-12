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

from mods.config_loader import config


class LSTM(nn.Module):
	"""lstm模型"""
	def __init__(self, input_size):
		super(LSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = int(np.floor(input_size * 2))
		self.num_layers = 2

		self.lstm = nn.LSTM(
			input_size = self.input_size,
			hidden_size = self.hidden_size,
			num_layers = self.num_layers,
			batch_first = True
		)

		self.connect_0 = nn.Linear(self.hidden_size, 1)
		# self.act_0 = nn.ReLU()
	
	def forward(self, x):
		x, _ = self.lstm(x)
		x = self.connect_0(x)
		# x = self.act_0(x)
		return x


class NN(nn.Module):
	"""神经网络模型"""
	# def __init__(self, input_size, hidden_size, output_size):
	# 	super(NN, self).__init__()
	# 	self.input_size = input_size
	# 	self.hidden_size = hidden_size
	# 	self.output_size = output_size
	#
	# 	self.connect_0 = nn.Linear(self.input_size, self.hidden_size[0])
	# 	self.act_0 = nn.Sigmoid()
	# 	self.connect_1 = nn.Linear(self.hidden_size[0], self.output_size)
	# 	self.act_1 = nn.ReLU()
	# 	# self.connect_2 = nn.Linear(self.hidden_size[1], self.output_size)
	# 	# self.act_2 = nn.ReLU()
	#
	# def forward(self, x):
	# 	x = self.connect_0(x)
	# 	x = self.act_0(x)
	# 	x = self.connect_1(x)
	# 	x = self.act_1(x)
	# 	# x = self.connect_2(x)
	# 	# x = self.act_2(x)
	# 	return x

	def __init__(self, input_size, hidden_size, output_size):
		super(NN, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size

		self.connect_0 = nn.Linear(self.input_size, self.output_size)
		self.act_0 = nn.ReLU()

	def forward(self, x):
		x = self.connect_0(x)
		x = self.act_0(x)
		return x


def initialize_lstm_params(lstm):
	"""
	初始化模型参数
	:param lstm: nn.LSTM(), 神经网络模型
	:param input_size: int, 输入维数
	:return:
		lstm, nn.LSTM(), 参数初始化后的神经网络模型
	"""
	lstm.connect_0.weight.data = torch.rand(1, lstm.hidden_size)  # attention: 注意shape是转置关系
	lstm.connect_0.bias.data = torch.rand(1)
	return lstm


def initialize_nn_params(nn):
	"""
	初始化神经网络模型参数
	:param nn: nn.NN(), 神经网络模型
	:param input_size: int, 输入维数
	:param hidden_size: list of ints, 中间隐含层维数
	:param output_size: int, 输出层维数
	:return:
		nn, nn.NN(), 参数初始化后的神经网络模型
	"""
	nn.connect_0.weight.data = torch.rand(nn.output_size, nn.input_size)
	nn.connect_0.bias.data = torch.rand(nn.output_size)
	# nn.connect_1.weight.data = torch.rand(nn.output_size, nn.hidden_size[0])
	# nn.connect_1.bias.data = torch.rand(nn.output_size)
	# nn.connect_2.weight.data = torch.rand(nn.output_size, nn.hidden_size[1])
	# nn.connect_2.bias.data = torch.rand(nn.output_size)
	return nn


import json
time = '2019-06-12'
num = 2
request_dict = {'time': time, 'num': num}
request_js = json.dumps(request_dict)
with open('../tmp/request.pkl', 'w') as f:
	json.dump(request_dict, f)

request_dict_2 = json.loads(request_js)

with open('../tmp/request.pkl', 'r') as f:
	request_dict_3 = json.load(f)

