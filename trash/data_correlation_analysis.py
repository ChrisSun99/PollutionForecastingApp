# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

数据相关性分析和可视化
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv

sys.path.append('../')

from mods.config_loader import config
from mods import time_delayed_correlation_analysis, pull_data_from_db

if __name__ == '__main__':
    # 载入数据
    result = pull_data_from_db.result
    NON_DER = ['aqi', ]
    df_new = time_delayed_correlation_analysis.df_derived_by_shift(result, 6, NON_DER)

    # Select data from table using SQL query.
    fields = pd.read_csv("../config/nameList.csv")
    data = [row for row in csv.reader(fields)]
    feature1 = data[0][0]
    feature2 = data[1][0]

    # 设定参数
    target_column = config.conf['model_params']['target_column']
    selected_columns = config.conf['model_params']['selected_columns']
    columns = [target_column] + selected_columns

    """
    可视化
    """

    colormap = plt.cm.RdBu
    plt.figure(figsize=(15, 10))
    plt.title(u'6 days', y=1.05, size=16)

    mask = np.zeros_like(df_new.corr())
    mask[np.triu_indices_from(mask)] = True

    svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=0.1, vmax=1.0,
                      square=True, cmap=colormap, linecolor='white', annot=True)

    plt.show()

# # 各时间序列作图
# plt.figure(figsize = [8, 7])
# for i in range(len(columns)):
# 	plt.subplot(len(columns), 1, i + 1)
# 	plt.plot(list(data[columns[i]]), linewidth = 1)
# 	plt.ylabel(columns[i])
# 	plt.xlim([0, len(data)])
# 	if i == len(columns) - 1:
# 		plt.xlabel('time (hr)')
# 		plt.tight_layout()

# # 各变量与目标点的相关散点图
# plt.figure(figsize = [6, 6])
# for i in range(len(selected_columns)):
# 	plt.scatter(data[target_column], data[selected_columns[i]], s = 1)
# 	plt.xlim([0.0, 1.0])
# 	plt.ylim([0.0, 1.0])
# plt.legend(['x: pm25, y: {}'.format(p) for p in selected_columns])
# plt.plot([0, 1], [0, 1], 'k--', linewidth = 1)
# plt.grid(True)
# plt.tight_layout()

# 同一序列中不同时间间隔点的相关性
# time_series = list(data[target_column])[:5000]
# time_intervals = list(range(0, 30, 6))
# for i in range(len(time_intervals)):
# 	plt.figure(figsize = [3, 3])
#
# 	if i == 0:
# 		series_a = time_series
# 		series_b = time_series
# 	else:
# 		series_a = time_series[: -time_intervals[i]]
# 		series_b = time_series[time_intervals[i]:]
# 	sns.kdeplot(series_a, series_b, cmap = "Blues", shade = True, shade_lowest = False)
# 	plt.legend(['r2_score: {:.2f}'.format(r2_score(series_a, series_b))], loc = 'upper right')
# 	plt.xlabel('value at t0')
# 	plt.ylabel('value at t0 + {} * dt'.format(time_intervals[i]))
# 	plt.tight_layout()
