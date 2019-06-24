from nose.tools import *
import pandas as pd
import json
import csv
import PIL

"""
测试用的web服务接口
"""
from nose.tools import *
import urllib
import sys

sys.path.append('../')
from bin.app import *
from lake.file import read as file_read


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
    response = c.post(path, json=data)
    print(response.data)
    res_obj = json.loads(response.data.decode('utf-8'))
    return res_obj


def api_hello(req_dict):
    return api_get('/test/', req_dict)


def api_correlation(req_dict):
    return api_post('/correlation/', req_dict)


def api_correlation_array(req_dict):
    return api_post('/correlation_array/', req_dict)


class Test(object):
    def setup(self):
        # 通用函数
        csvfile = open('../tmp/taiyuan_cityHour.csv', 'r')
        jsonfile = open('../tmp/data.json', 'w')
        fieldnames = (
        '_class', '_id', 'aqi', 'ci', 'city', 'co', 'coIci', 'grade', 'no2', 'no2Ici', 'o3', 'o3H8', 'o3Ici', 'pm10',
        'pm10Ici','pm25', 'pm25Ici', 'pp', 'ptime', 'so2', 'so2Ici', 'sd', 'temp', 'wd', 'weather', 'ws', 'itime', "regionId")
        reader = csv.DictReader(csvfile, fieldnames)
        for row in reader:
            json.dump(row, jsonfile)
            jsonfile.write('\n')
        self.data = []
        with open('../tmp/data.json', 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def test_api_hello(self):
        res = api_hello({})
        assert_equals(res['code'], 0)
        assert_equals(res['message'], "correlaton correct.")

    def test_api_correlation(self):
        res = api_correlation(self.data)
        assert_equals(res['code'], 0)
        assert_equals(res['message'], "correlaton correct.")

    def test_api_correlation_array(self):
        res = api_correlation_array(self.data)
        assert_equals(res['code'], 0)
        assert_equals(res['message'], "correlaton correct.")


if __name__ == "__main__":
    test = Test()
    test.setup()
    test.test_api_hello()
    test.test_api_correlation()
    test.test_api_correlation_array()
