# -*- coding: utf-8 -*-
"""
Created on 2019/6/7 19:14
@author: luolei

测试web服务接口
"""
import pandas as pd
import json
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
    response = c.post(path, data=json.dumps(data))
    print(response.data)
    res_obj = json.loads(response.data.decode('utf-8'))
    return res_obj


def api_hello(req_dict):
    return api_get('/test/', req_dict)


def api_correlation(req_dict):
    return api_post('/correlation/', req_dict)


def generate_starttime_endtime():
    """

    :return: list of starttime and endtime
    """


class Test(object):
    def setup(self):
        # 通用函数
        request_dict = {'data': {'starttime': '2017010101', 'endtime': '2017010123'}}
        self.data = request_dict

    def test_api_hello(self):
        res = api_hello({})
        assert_equal(res['code'], 0)
        assert_equal(res['message'], 'test successfully')

    def test_api_correlation(self):
        """测试相关系数计算接口计算是否正常"""
        res = api_correlation(self.data)
        assert_equal(res['code'], 0)
        assert_equal(res['message'], "correlation correct")

    def test_correlation_format(self):
        with open('../tmp/total_ccf_results.json', 'r') as f:
            result_dict = json.load(f)
            assert_is_instance(result_dict, dict)
            assert_equals(result_dict['aqi']['aqi'], [0, 1.0])
            assert_equals(result_dict['aqi']['co'][0], 500)
            assert_equals(result_dict['aqi']['grade'][0], 500)
            assert_equals(result_dict['aqi']['no2'][0], 502)
            assert_equals(result_dict['aqi']['o3'], [500, 0])
            assert_equals(result_dict['aqi']['o3H8'], [500, 0])
            assert_equals(result_dict['aqi']['pm10'][0], 500)
            assert_equals(result_dict['aqi']['pm25'][0], 500)
            assert_equals(result_dict['aqi']['sd'][0], 501)
            assert_equals(result_dict['aqi']['so2'][0], 500)
            assert_equals(result_dict['aqi']['temp'], [500, 0])
            assert_equals(result_dict['aqi']['ws'], [500, 0])
            assert_equals(result_dict['aqi']['weather_1'][0], 499)
            assert_equals(result_dict['aqi']['weather_2'][0], 503)
            assert_equals(result_dict['aqi']['weather_3'][0], 500)
            assert_equals(result_dict['aqi']['weather_4'][0], 1000)
            assert_equals(result_dict['aqi']['weather_5'][0], 996)
            assert result_dict['aqi']['weather_6'][0] < 400
            assert result_dict['aqi']['weather_7'][0] > 700
            assert result_dict['aqi']['weather_8'][0] == 90


if __name__ == "__main__":
    test = Test()
    test.setup()
    # test.test_api_hello()
    # test.test_api_correlation()
    test.test_correlation_format()
