import urllib
import json
from config.app import * # 已经编写好的 app


def api_get(path, data):
    # 定义get方式请求api方法
    url = '%s?%s' % (path, urllib.parse.urlencode(data))
    print('-' * 100)
    print('请求接口: ' + url)
    c = app.test_client()
    response = c.get(url)
    res_obj = json.loads(response.data.decode('utf-8'))
    return res_obj

def api_post(path, data):
    # 定义post方式请求api方法
    print('-' * 100)
    print('请求接口: ' + path)
    c = app.test_client()
    response = c.post(path, data = data)
    print(response.data)
    res_obj = json.loads(response.data.decode('utf-8'))
    return res_obj

def api_test(req_dict): # 调用test接口计算
    return api_get('/test/', req_dict)

def api_app_function(req_dict):
    # 调用app_function接口计算
    return api_post('/app_function/', req_dict)

test_result = api_test({})
params = {'key': 'value'}  # app_function 中的参数 app_function_result = api_app_function(params)