"""
Created on 2019/6/20 16:00
@author: mengtisun

时滞分析及相关性分析
"""
import numpy as np
import sys
import json

sys.path.append('../')

from mods.build_samples import build_data_frame_for_correlation_analysis
from others.gen_req_data import gen_req_data


def get_normalized_samples(data):
    """列归一化"""
    data = data.copy()
    data.drop(['ptime', 'time_stamp'], axis = 1, inplace = True)
    for column in data.columns:
        min_value, max_value = data[column].min(), data[column].max()
        if min_value < max_value:
            data.loc[:, column] = data.loc[:, column].apply(lambda x: (x - min_value) / (max_value - min_value))
        else:
            data.loc[:, column] = data.loc[:, column].apply(lambda x: 1.0)
    return data


def peak_loc_and_value(ccf_values):
    """从一段ccf计算结果中找出峰值位置和对应的值"""
    mean, sigma = np.mean(ccf_values), np.power(np.var(ccf_values), 0.5)
    pos_corr_values, neg_corr_values = [p for p in ccf_values if (p > mean + 3.0 * sigma)], [p for p in ccf_values if
                                                                                             (p < mean - 3.0 * sigma)]
    if (len(pos_corr_values) == 0) & (len(neg_corr_values) == 0):
        peak_value = 0
        peak_loc = int((len(ccf_values) - 1) / 2)
    else:
        if len(pos_corr_values) >= len(neg_corr_values):
            peak_value = max(pos_corr_values) - mean
            peak_loc = ccf_values.index(max(pos_corr_values))
        else:
            peak_value = mean - min(neg_corr_values)
            peak_loc = ccf_values.index(min(neg_corr_values))
    
    return peak_loc, peak_value


def ccf(series_a, series_b, d):
    """
    在延迟为d上的互相关分析, 固定series_a, 移动series_b: d > 0向左移，d < 0向右移
    :param time_series_0: np.ndarray, 目标变量
    :param time_series_1: np.ndarray, 外生变量
    :param d: 延迟阶数
    :return:
    """
    # 数据截断
    if d > 0:
        series_a = series_a.flatten()[d:]
        series_b = series_b.flatten()[:-d]
    elif d == 0:
        series_a = series_a.flatten()
        series_b = series_b.flatten()
    elif d < 0:
        series_a = series_a.flatten()[:d]
        series_b = series_b.flatten()[-d:]

    mean_a, mean_b = np.mean(series_a), np.mean(series_b)
    numerator = np.sum((series_a - mean_a) * (series_b - mean_b))
    denominator = np.sqrt(
        np.sum(np.power((series_a - mean_a), 2))
    ) * np.sqrt(
        np.sum(np.power((series_b - mean_b), 2))
    )
    
    eps = 1e-6

    return numerator / (denominator + eps)


def time_delayed_correlation(half_range_len = 500):
    """各变量间时滞检测和对应的相关系数值计算"""
    total_ccf_results = {}
    columns = list(data.columns)
    for i in range(len(columns)):
        print('processing column {}'.format(columns[i]))
        total_ccf_results[columns[i]] = {}
        for j in range(i, len(columns)):
            # 相同变量间的ccf值
            if i == j:
                total_ccf_results[columns[i]][columns[j]] = [0, 1.0]
        
            # 不同变量间的ccf值
            else:
                cross_correlation_results = []
                for d in range(-half_range_len, half_range_len + 1):
                    cross_correlation_results.append(
                        [d, ccf(np.array(data[columns[i]]), np.array(data[columns[j]]), d)]
                    )
                peak_loc, peak_value = peak_loc_and_value([p[1] for p in cross_correlation_results])
                total_ccf_results[columns[i]][columns[j]] = [peak_loc, peak_value]
                
    return json.dumps(total_ccf_results)
    

if __name__ == "__main__":
    # Generate fake req data.
    start_time = '2017010101'
    end_time = '2017123123'
    req_data = gen_req_data(start_time, end_time)
    
    # Generate data for analysis.
    req_dict = json.loads(req_data)
    data = build_data_frame_for_correlation_analysis(req_dict['start_time'], req_dict['end_time'])
    
    # Data normalization.
    data = get_normalized_samples(data)
    
    # Correlation analysis.
    total_ccf_results = time_delayed_correlation()
    
