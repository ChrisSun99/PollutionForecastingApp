#!flask/bin/python
import json
from flask import Flask, request, jsonify, redirect, url_for

app = Flask(__name__)


@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        searchword = request.args.get('key', '')
        return redirect(url_for('success', name=user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name=user))


@app.route('/requests', methods=['GET'])
def get_request_by_query_parameter():
    return 'name=' + request.args.get('name')


@app.route('/requests', methods=['POST'])
def post_request():
    return str(request.get_json())


# @app.route('/data', methods=['POST'])
# def f_data():
#     if request.method == "POST":
#         fields = [k for k in request.form]
#         values = [request.form[k] for k in request.form]
#         data = dict(zip(fields, values))
#     return jsonify(data)


if __name__ == '__main__':
    app.run(debug=False)  # app.run(host, port, debug, options)

    client_input = post_request()

    input_params = {
        'client_input': {
            'name': client_input
        }
    }

    with open('../bin/client_input.pkl', 'w') as f:
        json.dump(input_params, f)



