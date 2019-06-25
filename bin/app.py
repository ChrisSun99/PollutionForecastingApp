#!flask/bin/python
import json
import pandas as pd
import csv
import time
import io
from mods.config_loader import config
from flask import Flask, request, jsonify, redirect, url_for
from mods.time_delayed_correlation_analysis import get_normalized_samples, time_delayed_correlation

config.set_logging()

import logging

_logger = logging.getLogger(__name__)

from flask import Flask, request

app = Flask(__name__)


@app.route('/test/')
def hello_world():
    return json.dumps({'code': 0, 'message': 'test successfully', 'data': 'Hello World!'})


@app.route('/correlation/', methods=['POST'])
def correlation():
    """
    进行相关性检验
    :return:
    """
    print('<<<<<< starting correlation analysis, /correlation/')

    try:
        print(type(request.data))
        data = json.loads(request.data)
        data = pd.DataFrame(data)
        data = get_normalized_samples(data)
        time_start = time.time()
        samples = time_delayed_correlation()
        _logger.info('time cost for correlation analysis: %s secs' % (time.time() - time_start))
        print('>>>>>> correlation SUCCEEDED')
        return json.dumps({'code': 0, 'message': 'correlation correct', 'data': samples})

    except Exception as e:
        _logger.exception(e)
        print('>>>>>> correlation FAILED')
        return json.dumps({'code': 1, 'message': 'correlation failed', 'data': 'Null'})


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8000, debug = False)  # app.run(host, port, debug, options)
