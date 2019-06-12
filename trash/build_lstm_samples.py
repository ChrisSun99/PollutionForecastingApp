# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

构造lstm模型样本
"""
import pandas as pd
import numpy as np
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.data_filtering import savitzky_golay_filtering


def build_samples():
	"""
	构建全量样本集
	:return:
	"""
	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	
	# 滤波
	data_filtered = savitzky_golay_filtering(data)
	# data_filtered = data
	
	# 构建样本数据集
	seq_len = config.conf['model_params']['seq_len']
	selected_columns = config.conf['model_params']['selected_columns']
	
	data_array = np.array(data_filtered[selected_columns])
	
	# 数据扩充, 方便后面的lstm样本构造
	data_array = np.vstack((np.zeros([seq_len - 1, data_array.shape[1]]), data_array))
	
	samples = []
	for i in range(data_array.shape[0] - seq_len + 1):
		samples.append(data_array[i: i + seq_len, :])
	samples = np.array(samples)  # samples.shape = (samples_len, seq_len, features)
	
	return samples


def build_targets():
	"""
	构建全量目标数据集
	:return:
	"""
	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	
	# 滤波
	data_filtered = savitzky_golay_filtering(data)
	# data_filtered = data
	
	# 构建目标数据集
	seq_len = config.conf['model_params']['seq_len']
	target_column = config.conf['model_params']['target_column']
	
	target_array = np.array(data_filtered[target_column]).reshape(-1, 1)
	
	# 数据扩充
	target_array = np.vstack((target_array, np.zeros([seq_len - 1, 1])))
	
	targets = []
	for i in range(target_array.shape[0] - seq_len + 1):
		targets.append(target_array[i: i + seq_len, :])
	targets = np.array(targets)  # targets.shape = (samples_len, seq_len, 1)
	
	return targets


if __name__ == '__main__':
	samples = build_samples()
	targets = build_targets()
	
	print('samples shape: {}'.format(samples.shape))
	print('targets shape: {}'.format(targets.shape))

	

	
	


