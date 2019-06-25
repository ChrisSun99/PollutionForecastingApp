# -*- coding: utf-8 -*-
"""
Created on 2019/6/7 19:14
@author: luolei

测试web服务接口
"""
import pandas as pd
from nose.tools import *
import urllib
import sys

sys.path.append('../')

from bin.app import *
from mods.build_samples import build_data_frame_for_correlation_analysis


def api_get(path, data):
    url = '%s?%s' % (path, urllib.parse.urlencode(data))
    print('-' * 100)
    print('请求接口: ' + url)
    c = app.test_client()
    response = c.get(url)
    res_obj = json.loads(response.data.decode('utf-8'))
    return res_obj


def api_post(path, data):
    print('-' * 100)
    print('请求接口: ' + path)
    c = app.test_client()
    response = c.post(path, json = data)
    print(response.data)
    res_obj = json.loads(response.data.decode('utf-8'))
    return res_obj


def api_hello(req_dict):
    return api_get('/test/', req_dict)


def api_correlation(req_dict):
    return api_post('/correlation/', req_dict)


class Test(object):
    def setup(self):
        # 通用函数
        data = pd.read_csv("../tmp/taiyuan_cityHour.csv")
        self.data_list = []
        for i in range(len(data)):
            self.data_list.append(list(data.loc[i, :]))
        self.data = {}
        self.data.update({'data': self.data_list})

    def test_api_hello(self):
        res = api_hello({})
        assert_equal(res['code'], 0)
        assert_equal(res['message'], 'test successfully')

    def test_api_correlation(self):
        """测试相关系数计算接口计算是否正常"""
        res = api_correlation(self.data)
        assert_equal(res['code'], 0)
        assert_equal(res['message'], "correlation correct.")

    def test_correlation_format(self):
        result_dict = json.loads(open('../tmp/total_ccf_results.json', 'r'))
        assert_is_instance(result_dict, dict)
        assert_equals(result_dict['aqi'], [0, 1.0])
        assert_equals(result_dict['co'][0], 500)
        assert_equals(result_dict['grade'][0], 500)
        assert_equals(result_dict['no2'][0], 502)
        assert_equals(result_dict['o3'][0], 0)
        assert_equals(result_dict['o3H8'], [500, 0])
        assert_equals(result_dict['pm10'][0], 500)
        assert_equals(result_dict['pm25'][0], 500)
        assert_equals(result_dict['sd'][0], 501)
        assert_equals(result_dict['so2'][0], 500)
        assert_equals(result_dict['temp'], [500, 0])
        assert_equals(result_dict['ws'], [500, 0])
        assert_equals(result_dict['weather_1'][0], 499)
        assert_equals(result_dict['weather_2'], 503)
        assert_equals(result_dict['weather_3'], 500)
        assert_equals(result_dict['weather_4'], 1000)
        assert_equals(result_dict['weather_5'], 996)
        assert result_dict['weather_6'][0] < 400
        assert result_dict['weather_7'][0] > 700
        assert result_dict['weather_8'][0] == 90


if __name__ == "__main__":
    test = Test()
    test.setup()
    test.test_api_hello()
    test.test_api_correlation()
