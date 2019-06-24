import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sys
from mods import data_filtering

sys.path.append('../')


def df_derived_by_shift(df, lag=0, NON_DER=[]):
    """

    :param df: pd.DataFrame, 待检验变量数据表
    :param lag: 读入request中的time shift
    :param NON_DER: 需要分析的特征值
    :return:
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


if __name__ == "__main__":
    file_name = "../tmp/total_implemented_normalized_data.csv"
    data = pd.read_csv(file_name)

    data = data_filtering.savitzky_golay_filtering(data)
    NON_DER = ['aqi', ]
    df_new = df_derived_by_shift(data, 6, NON_DER)

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
