import os
import sys
import logging
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy

cur_dir, _ = os.path.split(os.path.abspath(__file__))
par_dir = os.path.abspath(os.path.join(cur_dir, '..'))
sys.path.insert(0, par_dir)

import server.config as conf
from server.core import Predictor

ai = Predictor()
logging.basicConfig(filename=os.path.join(cur_dir, 'debug.log'),
                    level=logging.DEBUG)

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = conf.db_url

db = SQLAlchemy(app)


class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    human_up = db.Column(db.Boolean, nullable=False)
    human_lord = db.Column(db.Boolean, nullable=False)
    human_down = db.Column(db.Boolean, nullable=False)
    win_up = db.Column(db.Boolean, nullable=False)
    win_lord = db.Column(db.Boolean, nullable=False)
    win_down = db.Column(db.Boolean, nullable=False)


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


@app.route('/record', methods=['GET', 'POST'])
def record():
    if request.method == 'POST':
        payload = request.get_json()
        human = [payload['is_human'][str(i)] for i in range(3)]
        win = [payload['record'][str(i)] for i in range(3)]
        values = human + win
        columns = ['human_up', 'human_lord', 'human_down',
                   'win_up', 'win_lord', 'win_down']
        r = Record(**dict(zip(columns, values)))
        db.session.add(r)
        db.session.commit()
        return {}
    else:
        return 'It works, record'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
