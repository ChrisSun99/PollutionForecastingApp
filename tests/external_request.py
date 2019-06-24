import json
import requests
from lake.file import read as file_read

unitDataList = json.loads(file_read('./tmp/total_ccf_results.json'))
data = {
    "aqi": 80.0,
    "ci": 1.607416749000549,
    "co": 0.5837500095367432,
    "coIci": 0.05837500095367432,
    "grade": 2.0,
    "no2": 26.0,
    "no2Ici": 0.12999999523162842,
    "o3": 66.0,
    "o3H8": 74.5,
    "o3Ici": 0.328125,
    "pm10": 110.0,
    "pm10Ici": 0.7350000143051147,
    "pm25": 25.0,
    "pm25Ici": 0.3316666781902313,
    "pp": "PM10",
    "ptime": 2016050516.0,
    "so2": 12.0,
    "so2Ici": 0.024250000715255737,
    "sd": 17.0,
    "temp": 25.0,
    "wd": 13.0,
    "weather": '晴',
    "ws": 2.0
}
r_get = requests.get('http://srv-pollution-interpolation.ai.dev.rktl.work/test/')
print(r_get.text)
r_post = requests.post('http://srv-pollution-interpolation.ai.dev.rktl.work/interp/', data = {'data': json.dumps(data)}) # 从外部请求接口有问题
print(r_post.text)
