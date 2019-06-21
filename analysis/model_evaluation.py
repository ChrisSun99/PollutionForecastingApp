# -*- coding: utf-8 -*-
"""
Created on 2019/6/8 14:14
@author: luolei

模型评估
"""
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import numpy as np
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.models import LSTM
from mods.build_samples_and_targets import build_train_samples_dict, build_train_targets_array
from mods.model_evaluations import smape, mae, rmse, r2


if __name__ == '__main__':
	# 设定参数
	pred_dim = config.conf['model_params']['pred_dim']
	use_cuda = config.conf['model_params']['pred_use_cuda']
	target_column = config.conf['model_params']['target_column']

	# 载入训练好的模型
	with open('../tmp/model_struc_params.pkl', 'r') as f:
		model_struc_params = json.load(f)

	lstm_state_dict = torch.load('../tmp/lstm_state_dict_{}.pth'.format(target_column), map_location = 'cpu')
	# nn_state_dict = torch.load('../tmp/nn_state_dict_{}.pth'.format(target_column))

	lstm = LSTM(input_size = model_struc_params['lstm']['input_size'])
	lstm.load_state_dict(lstm_state_dict, strict = False)
	# nn = NN(
	# 	input_size = model_struc_params['nn']['input_size'],
	# 	hidden_size = model_struc_params['nn']['hidden_size'],
	# 	output_size = model_struc_params['nn']['output_size'],
	# )
	# nn.load_state_dict(nn_state_dict, strict = False)

	# 载入训练样本和目标数据集
	train_samples_dict = build_train_samples_dict()
	train_targets_arr = build_train_targets_array()

	# 划分训练集
	X_train = np.concatenate([train_samples_dict[col] for col in train_samples_dict.keys()], axis = 2).astype(np.float32)
	X_train = np.hstack((X_train, np.zeros([X_train.shape[0], pred_dim, X_train.shape[2]]).astype(np.float32)))
	y_train = train_targets_arr.astype(np.float32)

	# 训练效果
	y_train_raw = y_train[:, :, 0]

	X_train = torch.from_numpy(X_train)

	if use_cuda:
		lstm = lstm.cuda()
		# nn = nn.cuda()
		X_train = X_train.cuda()

	y = lstm(X_train)
	y_train_model = y[:, -pred_dim:, 0]
	# y_train_model = nn(y)

	y_train_model = y_train_model.detach().cpu().numpy()

	# 还原为真实值
	target_column = config.conf['model_params']['target_column']
	bounds = config.conf['model_params']['variable_bounds'][target_column]
	y_train_raw = y_train_raw * (bounds[1] - bounds[0]) + bounds[0]
	y_train_model = y_train_model * (bounds[1] - bounds[0]) + bounds[0]

	# 模型训练结果评估
	rmse_results, smape_results, mae_results, r2_results = [], [], [], []
	for i in range(y_train_raw.shape[1]):
		rmse_results.append(rmse(y_train_raw[:, i], y_train_model[:, i]))
		smape_results.append(smape(y_train_raw[:, i], y_train_model[:, i]))
		mae_results.append(mae(y_train_raw[:, i], y_train_model[:, i]))
		r2_results.append(r2(y_train_raw[:, i], y_train_model[:, i]))

	print('\n===========TRAINING EFFECTS==============')
	for step in [0, 3, 7, 11, 23, 47]: #, 47, 71]:
		print('{} hr: rmse {:4f}, smape {:4f}, mae {:4f}, r2 {:4f}'.format(
			step, rmse_results[step], smape_results[step], mae_results[step], r2_results[step])
		)
	print('=========================================')

	plt.figure(figsize = [8, 4])
	plt.subplot(1, 2, 1)
	sns.heatmap(y_train_raw)
	plt.subplot(1, 2, 2)
	sns.heatmap(y_train_model)
	
	# 查看训练loss曲线
	with open('../tmp/train_loss.pkl', 'r') as f:
		loss_record = json.load(f)
	plt.figure('loss curve', figsize = [4, 3])
	plt.plot(loss_record)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.tight_layout()
