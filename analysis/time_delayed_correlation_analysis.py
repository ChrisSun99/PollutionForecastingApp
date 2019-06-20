"""
Created on 2019/6/20 16:00
@author: mengtisun

时滞分析及相关性分析
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sys
import json

sys.path.append('../')
from mods import build_samples_and_targets


def df_derived_by_shift(df, lag=0, NON_DER=[]):
    """

    :param df: pd.DataFrame, 待检验变量数据表
    :param lag: 读入request中的time shift
    :param NON_DER: 需要分析的特征值
    :return: dataFrame
    """
    df = df.copy()
    if not lag:
        return df
    cols = {}
    for i in range(1, lag + 1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i += 1
        df = pd.concat([df, dfn], axis=1, join_axes=[df.index])
    return df


def xcorr(x, y, normed=True, detrend=False, maxlags=10):
    """
    Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    :param x: 时间序列
    :param y: 时间序列
    :param normed: Optional; Returns inner products when normed=False
    :param detrend: Optional;
    :param maxlags: 允许的最大时滞，默认为10
    :return: Coefficient when normed=True
    """

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x))  # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))

    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y))  # this is the transformation function
        c = np.true_divide(c, n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c


if __name__ == "__main__":
    data = build_samples_and_targets.build_data_frame_for_correlation_analysis()
    data = data.fillna(0)
    with open('../tmp/request.pkl', 'r') as f:
        request_dict_3 = json.load(f)
    DER = request_dict_3["feature1"]
    NON_DER = request_dict_3["feature2"]

    df_new = df_derived_by_shift(data, 6, NON_DER)

    """
    相关性分析可视化
    """
    # Cross Correlation between two time series
    lags, c = xcorr(data[NON_DER], data[DER], maxlags=10)
    print(lags, c)

    # Cross Correlation between a time serie and the rest of the time series
    colormap = plt.cm.RdBu
    plt.figure(figsize=(15, 10))
    plt.title(u'6 days', y=1.05, size=16)

    mask = np.zeros_like(df_new.corr())
    mask[np.triu_indices_from(mask)] = True

    svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=0.1, vmax=1.0,
                      square=True, cmap=colormap, linecolor='white', annot=True)

    plt.show()
