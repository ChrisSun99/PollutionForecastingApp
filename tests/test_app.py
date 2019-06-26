# -*- coding: utf-8 -*-
"""
Created on 2019/6/7 19:14
@author: luolei

测试web服务接口
"""

import datetime
from random import randrange
from datetime import datetime, timedelta
from nose.tools import *
from nose.tools import assert_equal, assert_almost_equal
import urllib
import sys
import collections

sys.path.append('../')

from bin.app import *


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


def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


def gen_starttime_endtime(start, end):
    """

    :return: list of starttime and endtime
    """
    d1 = datetime.strptime(start, '%Y%m%d%H')
    d2 = datetime.strptime(end, '%Y%m%d%H')
    starttime = random_date(d1, d2)
    endtime = random_date(starttime, d2)
    return [starttime.strftime('%Y%m%d%H'), endtime.strftime('%Y%m%d%H')]


def flatten(d, parent_key = '', sep = '_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Test(object):
    def setup(self):
        # 通用函数
        time_list = gen_starttime_endtime('2016010101', '2019010101')
        request_dict = {'data': {'starttime': time_list[0], 'endtime': time_list[1]}}
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
        """
        测试数据类型、数据形状、数据大小范围
        转置数据，重新比较测试相关性
        :return:
        """
        with open('../tmp/total_ccf_results.json', 'r') as f:
            result_dict = json.load(f)

            # 测试数据类型
            assert_is_instance(result_dict, dict)
            not_visited_keys = []
            for key in result_dict.keys():
                not_visited_keys.append(key)
            for key in result_dict.keys():
                assert_is_instance(result_dict[key], dict)
                if key in not_visited_keys:
                    for not_visited_key in not_visited_keys:
                        assert_is_instance(result_dict[key][not_visited_key], list)
                        assert_equal(len(result_dict[key][not_visited_key]), 2)
                        not_visited_keys.remove(not_visited_key)

            # 测试数据大小范围
            key_list = result_dict.keys()
            assert_less(['aqi', 'co', 'grade', 'no2', 'no2Ici',
                         'o3', 'o3H8', 'o3Ici', 'pm10', 'pm10Ici', 'pm25', 'pm25Ici', 'pp', 'ptime', 'so2',
                         'so2Ici', 'sd', 'temp', 'wd', 'weather_1', 'ws'], list(key_list))
            for key in key_list:
                assert len(result_dict[key]) <= len(key_list)
                assert len(result_dict[key]) >= 1

            flattened_dict = flatten(result_dict)

            def add(x):
                sum = len(x)
                for i in range(len(x)):
                    sum += i
                return sum

            assert_equal(len(flattened_dict), add(result_dict))

            # assert_equal(result_dict[key]['co'][0], 500)
            # assert_equal(result_dict[key]['grade'][0], 500)
            # assert_equal(result_dict[key]['no2'][0], 502)
            # assert_equal(result_dict[key]['o3'], [500, 0])
            # assert_equal(result_dict[key]['o3H8'], [500, 0])
            # assert_equal(result_dict[key]['pm10'][0], 500)
            # assert_equal(result_dict[key]['pm25'][0], 500)
            # assert_equal(result_dict[key]['sd'][0], 501)
            # assert_equal(result_dict[key]['so2'][0], 500)
            # assert_equal(result_dict[key]['temp'], [500, 0])
            # assert_equal(result_dict[key]['ws'], [500, 0])
            # assert_equal(result_dict[key]['weather_1'][0], 499)
            # assert_equal(result_dict[key]['weather_2'][0], 503)
            # assert_equal(result_dict[key]['weather_3'][0], 500)
            # assert_equal(result_dict[key]['weather_4'][0], 1000)
            # assert_equal(result_dict[key]['weather_5'][0], 996)
            # assert result_dict[key]['weather_6'][0] < 400
            # assert result_dict[key]['weather_7'][0] > 700
            # assert result_dict[key]['weather_8'][0] == 90


if __name__ == "__main__":
    test = Test()
    # test.setup()
    # test.test_api_hello()
    # test.test_api_correlation()
    test.test_correlation_format()
