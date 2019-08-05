from flask import Flask, escape, request
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        payload = request.get_json()
        return payload
    else:
        return 'It works'
