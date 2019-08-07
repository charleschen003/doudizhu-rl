import time
import json
import torch
import numpy as np
from envi import r, Env
from dqn import DQNFirst
from net import NetCooperation
from flask import Flask, request

app = Flask(__name__)
# app.logger.setLevel('INFO')

mock_env = Env(seed=0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lord = DQNFirst(NetCooperation)
up = DQNFirst(NetCooperation)
down = DQNFirst(NetCooperation)
lord.policy_net.load('0807_1340_lord_4000')  # 原：0805_1409_lord_4000
up.policy_net.load('0807_1344_up_6000')
down.policy_net.load('0807_1344_down_6000')
AI = {0: up, 1: lord, 2: down}
NAME = {0: '地主上', 1: '地主', 2: '地主下'}
ai = None


def get_prob(role_id, cur_cards, history, left):
    size1, size2 = left[(role_id + 1 + 3) % 3], left[(role_id + 2 + 3) % 3]
    taken = np.hstack(list(history.values())).astype(np.int)
    cards = np.array(cur_cards, dtype=np.int)
    known = mock_env.cards2arr(np.hstack([taken, cards]))
    known = mock_env.batch_arr2onehot([known]).flatten()
    prob = mock_env.get_state_prob_manual(known, size1, size2)
    return prob


def parse_history(role_id, history, last_taken):
    h0 = history[(role_id - 1 + 3) % 3]
    h1 = history[(role_id + 0 + 3) % 3]
    h2 = history[(role_id + 1 + 3) % 3]
    b1 = last_taken[(role_id - 1 + 3) % 3]
    b2 = last_taken[(role_id - 2 + 3) % 3]
    taken = h0 + h1 + h2
    return list(map(mock_env.cards2arr, [taken, h0, h1, h2, b1, b2]))


def face(role_id, cur_cards, history, left, last_taken):
    """
    :return:  4 * 15 * 4 的数组，作为当前状态
    """
    # 已知数据
    handcards = mock_env.cards2arr(cur_cards)
    taken, h0, h1, h2, b1, b2 = parse_history(role_id, history, last_taken)
    known = mock_env.batch_arr2onehot([handcards, taken, h0, h1, h2, b1, b2])
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
    start_time = time.time()
    global ai
    if not payload['cur_cards']:
        return {'msg': '无手牌', 'status': False, 'data': []}
    debug = payload.pop('debug', False)
    for key in ['history', 'last_taken', 'left']:
        payload[key][0] = payload[key].pop('0')
        payload[key][1] = payload[key].pop('1')
        payload[key][2] = payload[key].pop('2')

    role_id = payload['role_id']
    ai = AI[role_id]
    name = NAME[role_id]
    state = face(**payload)
    last = payload['last_taken'][(role_id - 1 + 3) % 3]
    actions = valid_actions(payload['cur_cards'], last)

    data = choose(state, actions)
    end_time = time.time()
    msg = (('\n\t【{}】响应耗时{:.2f}ms\n'
            '\t【{}】桌上的牌：{}\n'
            '\t【{}】上家出牌：{}\n'
            '\t【{}】当前手牌：{}\n'
            '\t【{}】本次出牌：{}')
           .format(name, 1000 * (end_time - start_time),
                   name, payload['history'],
                   name, last,
                   name, payload['cur_cards'],
                   name, data))
    app.logger.debug(msg)
    res = {'msg': 'success', 'statue': True, 'data': data}
    if debug:
        res['msg'] = msg
    return res


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        payload = request.get_json()
        return json.dumps(response(payload))
    else:
        return 'It works'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
