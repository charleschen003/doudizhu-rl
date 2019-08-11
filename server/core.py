import time
import torch
import numpy as np
import server.config as conf
from envi import r, Env

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Predictor:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Predictor, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.mock_env = Env(seed=0)
        self.lord = self.up = self.down = None
        for role in ['lord', 'up', 'down']:
            if conf.net_dict[role]:
                ai = conf.dqn_dict[role](conf.net_dict[role])
                ai.policy_net.load(conf.model_dict[role])
                setattr(self, role, ai)
        self.id2ai = {0: self.up, 1: self.lord, 2: self.down}
        self.id2name = {0: '地主上', 1: '地主', 2: '地主下'}

    def get_prob(self, role_id, cur_cards, history, left):
        size1, size2 = left[(role_id + 1 + 3) % 3], left[(role_id + 2 + 3) % 3]
        taken = np.hstack(list(history.values())).astype(np.int)
        cards = np.array(cur_cards, dtype=np.int)
        known = self.mock_env.cards2arr(np.hstack([taken, cards]))
        known = self.mock_env.batch_arr2onehot([known]).flatten()
        prob = self.mock_env.get_state_prob_manual(known, size1, size2)
        return prob

    def parse_history(self, role_id, history, last_taken):
        h0 = history[(role_id - 1 + 3) % 3]
        h1 = history[(role_id + 0 + 3) % 3]
        h2 = history[(role_id + 1 + 3) % 3]
        b1 = last_taken[(role_id - 1 + 3) % 3]
        b2 = last_taken[(role_id - 2 + 3) % 3]
        taken = h0 + h1 + h2
        return list(map(self.mock_env.cards2arr, [taken, h0, h1, h2, b1, b2]))

    def face(self, role_id, cur_cards, history, left, last_taken):
        """
        :return:  6 * 15 * 4 的数组，作为当前状态
        """
        # 已知数据
        handcards = self.mock_env.cards2arr(cur_cards)
        taken, h0, h1, h2, b1, b2 = self.parse_history(role_id, history, last_taken)
        known = self.mock_env.batch_arr2onehot([handcards, taken, b1, b2])
        prob = self.get_prob(role_id, cur_cards, history, left).reshape(2, 15, 4)
        state = np.concatenate((known, prob))
        return torch.tensor(state, dtype=torch.float).to(DEVICE)

    def valid_actions(self, role_id, cur_cards, last_taken, **kwargs):
        """
        :return:  batch_size * 15 * 4 的可行动作集合
        """
        last = last_taken[(role_id - 1 + 3) % 3]
        if not last:
            last = last_taken[(role_id - 2 + 3) % 3]
        last_back = last
        cur_cards, last = list(map(self.mock_env.cards2arr, [cur_cards, last]))
        actions = r.get_moves(cur_cards, last)
        return last_back, torch.tensor(self.mock_env.batch_arr2onehot(actions),
                                       dtype=torch.float).to(DEVICE)

    def choose(self, role_id, state, actions):
        action = self.id2ai[role_id].greedy_action(state, actions)
        action = self.mock_env.onehot2arr(action)
        return [int(i) for i in self.mock_env.arr2cards(action)]

    def act(self, payload):  # TODO 判断使用哪个model
        """
        :param payload = {
                'role_id': 1,  # 0代表地主上家，1代表地主，2代表地主下家
                'last_taken': {  # 更改处
                    0: [],
                    1: [3, 4, 5, 6, 7, 8, 9],
                    2: [7,  8,  9, 10, 11, 12, 13],
                },
                'cur_cards': [15, 15, 14, 13, 13, 12, 11, 10,  9,  6,  6,  6,  4],  # 无需保持顺序
                'history': {  # 各家走过的牌的历史The environment
                    0: [],
                    1: [3, 4, 5, 6, 7, 8, 9],
                    2: [7,  8,  9, 10, 11, 12, 13],
                },
                'left': {  # 各家剩余的牌
                    0: 17,
                    1: 13,
                    2: 10,
                },
                'debug': False,  # 是否返回debug
            }
        :return:
        """
        start_time = time.time()

        if not payload['cur_cards']:
            return {'msg': '无手牌', 'status': False, 'data': []}
        for key in ['history', 'last_taken', 'left']:
            payload[key][0] = payload[key].pop('0')
            payload[key][1] = payload[key].pop('1')
            payload[key][2] = payload[key].pop('2')

        state = self.face(**payload)
        last, actions = self.valid_actions(**payload)
        action = self.choose(payload['role_id'], state, actions)

        end_time = time.time()
        msg = (('\n\t【{0}】响应耗时{1:.2f}ms\n'
                '\t【{0}】桌上的牌：{2}\n'
                '\t【{0}】上家出牌：{3}\n'
                '\t【{0}】当前手牌：{4}\n'
                '\t【{0}】本次出牌：{5}')
               .format(self.id2name[payload['role_id']],
                       1000 * (end_time - start_time), payload['history'],
                       last, payload['cur_cards'], action))
        res = {'msg': msg, 'status': True, 'data': action}
        return res
