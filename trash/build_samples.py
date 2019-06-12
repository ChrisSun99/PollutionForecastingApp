# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei


"""
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei


"""
import copy
import time
from scipy.ndimage.interpolation import shift
import numpy as np
import pandas as pd
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.data_filtering import savitzky_golay_filtering


def build_single_dim_manifold(time_series, embed_dim, lag, time_lag, direc = 1):
	"""
	构建一维时间序列嵌入流形
	:param time_lag: int, 对应的时间延迟
	:param direc: int, 平移方向，1为向下，-1为向上
	:param time_series: np.ndarray or pd.DataFrame, 一维时间序列, shape = (-1,)
	:param embed_dim: int, 嵌入维数
	:param lag: int, 嵌入延迟
	:return: manifold: np.ndarray, 嵌入流形数组, shape = (-1, embed_dim)
	"""
	time_series_copy = copy.deepcopy(time_series)
	manifold = []
	for dim in range(embed_dim):
		manifold.append(shift(shift(time_series_copy, direc * dim * lag), time_lag))
	manifold = np.array(manifold).T
	return manifold


def build_samples_data_frame(data):
	"""
	构建样本集
	:param data: pd.DataFrame, 数据表
	:return:
		data_new: pd.DataFrame, 构建的全量数据集，每个字段对应向量从前往后对应时间戳降序排列
	"""
	selected_columns = config.conf['model_params']['selected_columns']
	embed_lags = config.conf['model_params']['embed_lags']
	acf_lags = config.conf['model_params']['acf_lags']
	time_lags = config.conf['model_params']['time_lags']
	embed_dims = dict()
	for col in selected_columns:
		embed_dims[col] = int(np.floor(acf_lags[col] / embed_lags[col]))
		print('embed_dim for {} is {}'.format(col, embed_dims[col]))
	
	data_new = data[['time_stamp']]
	for col in selected_columns:
		samples = build_single_dim_manifold(data.loc[:, col], embed_dims[col], embed_lags[col], time_lags[col])
		columns = [col + '_{}'.format(i) for i in range(samples.shape[1])]
		samples = pd.DataFrame(samples, columns = columns)
		data_new = pd.concat([data_new, samples], axis = 1, sort = True)
	
	return data_new


def build_targets_data_frame(data):
	"""
	构建目标数据集
	:param data: pd.DataFrame, 数据表
	:return:
	"""
	target_column = config.conf['model_params']['target_column']
	embed_lag = 1
	seq_len = config.conf['model_params']['seq_len']
	
	data_new = data[['time_stamp']]
	samples = build_single_dim_manifold(data.loc[:, target_column], seq_len, embed_lag, direc = -1, time_lag = 0)
	columns = [target_column + '_{}'.format(i) for i in range(samples.shape[1])]
	samples = pd.DataFrame(samples, columns = columns)
	data_new = pd.concat([data_new, samples], axis = 1, sort = True)
	
	return data_new


def build_samples_arr(samples_df):
	"""
	构建全量samples数据集
	:param samples_df:
	:return:
	"""
	samples = np.array(samples_df)
	seq_len = config.conf['model_params']['seq_len']
	
	# 数据扩充, 方便后面的lstm样本构造
	data_array = np.vstack((np.zeros([seq_len - 1, samples.shape[1]]), samples))
	
	samples = []
	for i in range(data_array.shape[0] - seq_len + 1):
		samples.append(data_array[i: i + seq_len, :])
	samples = np.array(samples)  # samples.shape = (samples_len, seq_len, features)
	
	return samples


def build_targets_arr(targets_df):
	"""
	构建全量samples数据集
	:param samples_df:
	:return:
	"""
	targets = np.array(targets_df)
	seq_len = config.conf['model_params']['seq_len']
	
	# 数据扩充, 方便后面的lstm样本构造
	data_array = np.vstack((targets, np.zeros([seq_len - 1, targets.shape[1]])))
	
	targets = []
	for i in range(data_array.shape[0] - seq_len + 1):
		targets.append(data_array[i: i + seq_len, :])
	targets = np.array(targets)  # targets.shape = (samples_len, seq_len, 1)
	
	return targets


def build_samples():
	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	
	samples_df = build_samples_data_frame(data)
	samples = build_samples_arr(samples_df)
	
	return samples


def build_targets():
	# 载入数据
	data = pd.read_csv('../tmp/total_implemented_normalized_data.csv')
	
	targets_df = build_targets_data_frame(data)
	targets = build_targets_arr(targets_df)
	
	return targets


if __name__ == '__main__':
	samples = build_samples()
	targets = build_targets()
	
	print('samples shape: {}'.format(samples.shape))
	print('targets shape: {}'.format(targets.shape))

