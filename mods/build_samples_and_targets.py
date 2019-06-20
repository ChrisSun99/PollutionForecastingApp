# -*- coding: utf-8 -*-
"""
Created on 2019/6/7 19:14
@author: luolei

构建lstm模型样本
"""
import time
import pandas as pd
import numpy as np
import copy
from scipy.ndimage.interpolation import shift
import category_encoders as ce
import sys

sys.path.append('../')

from mods.data_filtering import savitzky_golay_filtering
from mods.pull_data_from_db import pull_data_from_db
from mods.config_loader import config


def build_data_frame_for_correlation_analysis():
    """构建相关分析用样本"""
    db_config = config.conf['db_config']
    categorical_columns = config.conf['model_params']['categorical_columns']
    numerical_columns = config.conf['model_params']['numerical_columns']
    
    # 载入数据
    data = pull_data_from_db(db_config)

    # 时间戳
    data['time_stamp'] = data.loc[:, 'ptime'].apply(lambda x: int(time.mktime(time.strptime(str(int(float(x))), "%Y%m%d%H"))))

    # 分离类别数据
    data = data.drop(["_class", "_id", "city", "itime", "regionId"], axis = 1)
    data = data[['time_stamp'] + categorical_columns + numerical_columns]

    # 对类别数据进行One-Hot Encoding
    enc = ce.OneHotEncoder(return_df = True, handle_unknown = "ignore")
    encoded_categorical_data = pd.DataFrame(enc.fit_transform(data[categorical_columns]))
    
    # 对数值数据进行滤波
    filtered_numerical_data = savitzky_golay_filtering(data[numerical_columns].astype('float'))
    
    # 合并数据
    data = pd.concat([data[['time_stamp']], filtered_numerical_data, encoded_categorical_data], axis = 1, sort = False)

    return data


if __name__ == '__main__':
    data = build_data_frame_for_correlation_analysis()
