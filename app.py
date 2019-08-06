import time
import json
import torch
import numpy as np
from envi import r, Env
from dqn import DQNFirst
from net import NetMoreComplicated
from flask import Flask, request

app = Flask(__name__)
app.logger.setLevel('INFO')

mock_env = Env(seed=0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ai = DQNFirst(NetMoreComplicated)
ai.policy_net.load('0805_1409_lord_4000')


def get_prob(role_id, cur_cards, history, left):
    size1, size2 = left[(role_id + 1 + 3) % 3], left[(role_id + 2 + 3) % 3]
    taken = np.hstack(list(history.values())).astype(np.int)
    cards = np.array(cur_cards, dtype=np.int)
    known = mock_env.cards2arr(np.hstack([taken, cards]))
    known = mock_env.batch_arr2onehot([known]).flatten()
    prob = mock_env.get_state_prob_manual(known, size1, size2)
    return prob


def parse_history(role_id, history):
    h0 = history[(role_id - 1 + 3) % 3]
    h1 = history[(role_id + 0 + 3) % 3]
    h2 = history[(role_id + 1 + 3) % 3]
    taken = h0 + h1 + h2
    return list(map(mock_env.cards2arr, [taken, h0, h1, h2]))


def face(role_id, cur_cards, history, left):
    """
    :return:  4 * 15 * 4 的数组，作为当前状态
    """
    # 已知数据
    handcards = mock_env.cards2arr(cur_cards)
    taken, h0, h1, h2 = parse_history(role_id, history)
    known = mock_env.batch_arr2onehot([handcards, taken, h0, h1, h2])
    prob = get_prob(role_id, cur_cards, history, left).reshape(2, 15, 4)
    state = np.concatenate((known, prob))
    return torch.tensor(state, dtype=torch.float).to(DEVICE)


def valid_actions(cur_cards, last, tensor=True):
    """
    :return:  batch_size * 15 * 4 的可行动作集合
    """
    cur_cards, last = list(map(mock_env.cards2arr, [cur_cards, last]))
    actions = r.get_moves(cur_cards, last)
    if tensor:
        return torch.tensor(mock_env.batch_arr2onehot(actions),
                            dtype=torch.float).to(DEVICE)
    else:
        return actions


def choose(state, actions):
    action = ai.greedy_action(state, actions)
    action = mock_env.onehot2arr(action)
    return [int(i) for i in mock_env.arr2cards(action)]


def response(payload):
    if payload['role_id'] != 1:
        return {'msg': '现在只支持地主', 'statue': False, 'data': []}
    start_time = time.time()
    payload['history'][1] = payload['history'].pop('1')
    payload['history'][2] = payload['history'].pop('2')
    payload['history'][0] = payload['history'].pop('0')
    payload['left'][1] = payload['left'].pop('1')
    payload['left'][2] = payload['left'].pop('2')
    payload['left'][0] = payload['left'].pop('0')
    last_taken = payload.pop('last_taken')
    state = face(**payload)
    actions = valid_actions(payload['cur_cards'], last_taken)
    data = choose(state, actions)
    end_time = time.time()
    app.logger.info('Response takes {:.2f}ms'
                    .format(1000 * (end_time - start_time)))
    return {'msg': 'success', 'statue': True, 'data': data}


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        payload = request.get_json()
        return json.dumps(response(payload))
    else:
        return 'It works'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10086, debug=False)
