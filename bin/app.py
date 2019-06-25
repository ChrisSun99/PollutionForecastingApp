#!flask/bin/python
import json
import pandas as pd
from io import StringIO
import csv
import time
import io
from mods.config_loader import config
from flask import Flask, request, jsonify, redirect, url_for
from mods import time_delayed_correlation_analysis

config.set_logging()

import logging

_logger = logging.getLogger(__name__)

from flask import Flask, request

app = Flask(__name__)


@app.route('/success/<name>')
def success(name):
    return 'You selected: %s ' % name


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        feature1 = request.form['nm']
        feature2 = request.form['nmm']
        fieldnames = ['feature1', 'feature2']

        request_dict = {
            'feature1': feature1,
            'feature2': feature2
        }
        request_js = json.dumps(request_dict)
        with open('../tmp/request.pkl', 'w') as f:
            json.dump(request_dict, f)

        return redirect(url_for('success', name=feature1))
    else:
        feature2 = request.args.get('nmm')
        return redirect(url_for('success', name=feature2))


@app.route('/requests', methods=['GET'])
def get_request_by_query_parameter():
    return request.args.get('feature1')


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
        data = json.loads(request.data)
        print(data)
        data = time_delayed_correlation_analysis.get_normalized_samples(data)
        time_start = time.time()
        samples = time_delayed_correlation_analysis.time_delayed_correlation()
        _logger.info('time cost for correlation analysis: %s secs' % (time.time() - time_start))
        print('>>>>>> correlation SUCCEEDED')
        return json.dumps({'code': 0, 'message': 'correlation correct', 'data': samples})

    except Exception as e:
        _logger.exception(e)
        print('>>>>>> correlation FAILED')
        return json.dumps({'code': 1, 'message': 'correlation failed', 'data': 'Null'})


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8000, debug = False)  # app.run(host, port, debug, options)
