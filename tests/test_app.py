

"""
测试用的web服务接口
"""
from nose.tools import *
import urllib
import sys
import unittest
import csv

sys.path.append('../')
from bin.app import *
from mods import build_samples


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
    response = c.post(path, data)
    print(response.data)
    res_obj = json.loads(response.data.decode('utf-8'))
    return res_obj


def api_hello(req_dict):
    return api_get('/test/', req_dict)


def api_correlation(req_dict):
    return api_post('/correlation/', req_dict)


class Test(object):
    def setup(self):
        #通用函数
        self.d = []
        with open("../tmp/taiyuan_cityHour.csv") as f:
            records = csv.DictReader(f)
            for row in records:
                self.d.append(row)
        self.dat = {}
        self.dat.update({'data' : self.d})
        with open('../tmp/data.json', 'w') as f:
            json.dump(self.dat, f)
        self.data = json.load(open('../tmp/data.json', 'rb'))

    def test_api_hello(self):
        res = api_hello({})
        assert_equals(res['code'], 0)
        assert_equals(res['message'], 'test successfully')

    def test_api_correlation(self):
        res = api_correlation(json.dumps(self.data))
        assert_equals(res['code'], 0)
        assert_equals(res['message'], "correlation correct.")

    def test_api_correlation_format(self):
        result_dict = json.loads(open('../tmp/total_ccf_results.json', 'r'))
        assert_is_instance(result_dict, dict)
        assert_is_instance(result_dict['aqi'], dict)
        aqi_dict = result_dict['aqi']
        assert_equals(aqi_dict['aqi'], [0, 1.0])
        assert_equals(aqi_dict['co'][0], 500)
        assert_equals(aqi_dict['grade'][0], 500)
        assert_equals(aqi_dict['no2'][0], 502)
        assert_equals(aqi_dict['o3'][0], 0)
        assert_equals(aqi_dict['o3H8'], [500, 0])
        assert_equals(aqi_dict['pm10'][0], 500)
        assert_equals(aqi_dict['pm25'][0], 500)
        assert_equals(aqi_dict['sd'][0], 501)
        assert_equals(aqi_dict['so2'][0], 500)
        assert_equals(aqi_dict['temp'], [500, 0])
        assert_equals(aqi_dict['ws'], [500, 0])
        assert_equals(aqi_dict['weather_1'][0], 499)
        assert_equals(aqi_dict['weather_2'], 503)
        assert_equals(aqi_dict['weather_3'], 500)
        assert_equals(aqi_dict['weather_4'], 1000)
        assert_equals(aqi_dict['weather_5'], 996)
        assert aqi_dict['weather_6'][0] < 400
        assert aqi_dict['weather_7'][0] > 700
        assert aqi_dict['weather_8'][0] == 90
        assert aqi_dict['weather_9'][0] < 80
        assert aqi_dict['weather_10'] == [500, 0]
        assert aqi_dict['weather_11'][0] < 80
        assert aqi_dict['weather_12'][0] < 80
        assert aqi_dict['weather_13'][0] < 600
        assert aqi_dict['weather_14'] == [500, 0]
        assert aqi_dict['wd'] == [500, 0]


if __name__ == "__main__":
    test = Test()
    test.setup()
    test.test_api_hello()
    test.test_api_correlation()
    test.test_api_correlation_format()
