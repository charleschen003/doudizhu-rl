import os
import sys

cur_dir, _ = os.path.split(os.path.abspath(__file__))
par_dir = os.path.abspath(os.path.join(cur_dir, '..'))
sys.path.insert(0, par_dir)

from flask import Flask, request
from server.core import Predictor

app = Flask(__name__)
ai = Predictor()


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        payload = request.get_json()
        debug = payload.pop('debug', False)
        res = ai.act(payload)
        app.logger.debug(res['msg'])
        if debug is False:
            res['msg'] = 'success'
        return res
    else:
        return 'It works'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
