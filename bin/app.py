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

        with open('nameList.csv', 'w') as inFile:

            # dictionaries instead of having to add the csv manually.
            writer = csv.DictWriter(inFile, fieldnames=fieldnames)

            # writerow() will write a row in your csv file
            writer.writerow({'feature1': feature1, 'feature2': feature2})

        return redirect(url_for('success', name=feature1))
    else:
        feature2 = request.args.get('nmm')
        return redirect(url_for('success', name=feature2))


@app.route('/requests', methods=['GET'])
def get_request_by_query_parameter():
    return request.args.get('feature1')


# @app.route('/requests', methods=['POST'])
# def post_request():
#     return str(request.get_json())

# @app.route('/data', methods=['POST'])
# def f_data():
#     if request.method == "POST":
#         fields = [k for k in request.form]
#         values = [request.form[k] for k in request.form]
#         data = dict(zip(fields, values))
#     return jsonify(data)


if __name__ == '__main__':
    app.run(debug=False)  # app.run(host, port, debug, options)

