#!flask/bin/python
from flask import Flask, request, jsonify, redirect, url_for

app = Flask(__name__)

@app.route('/success')
def success(name):
    return 'welcome %s' %name

@app.route('/login', methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success', name = user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name = user))




if __name__=='__main__':

    app.run(debug=False)  #app.run(host, port, debug, options)

