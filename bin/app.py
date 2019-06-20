#!flask/bin/python
import json
import csv
from flask import Flask, request, jsonify, redirect, url_for

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


if __name__ == '__main__':
    app.run(debug=False)  # app.run(host, port, debug, options)
