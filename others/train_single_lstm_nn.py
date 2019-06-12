# -*- coding: utf-8 -*-
"""
Created on 2019/6/7 19:35
@author: luolei

训练lstm模型
"""
import json
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import numpy as np
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.build_samples_and_targets import build_train_samples_dict, build_train_targets_array
from mods.loss_criterion import criterion
from mods.models import LSTM, NN, initialize_lstm_params, initialize_nn_params


if __name__ == '__main__':
	# 设定参数
	target_column = config.conf['model_params']['target_column']
	use_cuda = config.conf['model_params']['train_use_cuda']
	lr = config.conf['model_params']['lr']
	epochs = config.conf['model_params']['epochs']
	batch_size = config.conf['model_params']['batch_size']
	pred_dim = config.conf['model_params']['pred_dim']

	# 载入训练样本和目标数据集
	train_samples_dict = build_train_samples_dict()
	train_targets_arr = build_train_targets_array()

	# 划分训练集
	X_train = train_samples_dict['pm25'].astype(np.float32)
	X_train = np.hstack((X_train, np.zeros([X_train.shape[0], pred_dim, 1]).astype(np.float32)))
	y_train = train_targets_arr.astype(np.float32)

	torch_dataset = Data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
	trainloader = DataLoader(torch_dataset, batch_size = batch_size, shuffle = True)
	if use_cuda:
		trainloader = [(tx.cuda(), ty.cuda()) for (tx, ty) in trainloader]

	# 初始化模型
	lstm_input_size = X_train.shape[2]
	lstm = LSTM(lstm_input_size)  # 样本的特征数

	# nn_input_size = pred_dim
	# nn_hidden_size = [int(nn_input_size / 2), int(nn_input_size / 2)]
	# nn_output_size = pred_dim
	# nn = NN(
	# 	input_size = nn_input_size,
	# 	hidden_size = nn_hidden_size,
	# 	output_size = nn_output_size
	# )

	lstm = initialize_lstm_params(lstm)
	# nn = initialize_nn_params(nn)

	if use_cuda:
		lstm = lstm.cuda()
		# nn = nn.cuda()

	# 设定优化器
	# optimizer = torch.optim.SGD(
	# 	[
	# 		{'params': lstm.parameters()},
	# 		{'params': nn.parameters()}
	# 	],
	# 	lr = lr
	# )
	optimizer = torch.optim.Adam(
		lstm.parameters(),
		lr = lr
	)

	# 进行模型训练
	loss_record = []
	for step in range(epochs):
		for train_x, train_y in trainloader:
			lstm_out = lstm(train_x)
			lstm_out = lstm_out[:, -pred_dim:, 0]
			# nn_out = nn(lstm_out)
			loss = criterion(lstm_out, train_y[:, :, 0])
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		loss_record.append(loss)
		print(step, loss)

	# 损失函数记录
	loss_record = [float(p.detach().cpu().numpy()) for p in loss_record]

	# 保存损失函数记录
	with open('../tmp/train_loss.pkl', 'w') as f:
		json.dump(loss_record, f)

	# 保存模型文件
	torch.save(lstm.state_dict(), '../tmp/lstm_state_dict_{}.pth'.format(target_column))
	# torch.save(nn.state_dict(), '../tmp/nn_state_dict_{}.pth'.format(target_column))

	# 保存模型结构参数
	model_struc_params = {
		'lstm': {
			'input_size': lstm.lstm.input_size
		}
		# 'nn': {
		# 	'input_size': nn.connect_0.in_features,
		# 	# 'hidden_size': [nn.connect_0.out_features, nn.connect_1.in_features],
		# 	# 'output_size': nn.connect_1.out_features
		# 	'hidden_size': [nn.connect_0.out_features],
		# 	'output_size': nn.output_size
		# }
	}

	with open('../tmp/model_struc_params.pkl', 'w') as f:
		json.dump(model_struc_params, f)