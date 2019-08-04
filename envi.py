import sys
import config as conf
import torch
import random
import numpy as np
from config import DEVICE
from collections import Counter

sys.path.insert(0, conf.ENV_DIR)
import r
from env import Env as CEnv


class Env(CEnv):
    def __init__(self, debug=False, seed=None):
        if seed:
            super(Env, self).__init__(seed=seed)
        else:
            super(Env, self).__init__()
        self.taken = np.zeros((15,))
        self.left = np.array([17, 20, 17], dtype=np.int)
        self.old_cards = None
        self.debug = debug

    def reset(self):
        super(Env, self).reset()
        self.taken = np.zeros((15,))
        self.left = np.array([17, 20, 17])

    def _update(self, role, cards):
        self.left[role] -= len(cards)
        for card, count in Counter(cards - 3).items():
            self.taken[card] += count
        if self.debug:
            if role == 1:
                name = '地主'
                print('地主剩牌: {}'.format(self.cards2str(self.old_cards)))
                input()
            elif role == 0:
                name = '农民1'
            else:
                name = '农民2'
            print('{} 出牌： {}，分别剩余： {}'.format(
                name, self.cards2str(cards), self.left))
            input()

    def step_manual(self, onehot_cards):
        role = self.get_role_ID() - 1
        arr_cards = self.onehot2arr(onehot_cards)
        cards = self.arr2cards(arr_cards)

        self._update(role, cards)
        return super(Env, self).step_manual(cards)

    def step_auto(self):
        role = self.get_role_ID() - 1
        cards, r, _ = super(Env, self).step_auto()
        self._update(role, cards)
        return cards, r, _

    def step_random(self):
        actions = self.valid_actions(tensor=False)
        cards = self.arr2cards(random.choice(actions))
        self._update(self.get_role_ID() - 1, cards)
        return super(Env, self).step_manual(cards)

    @property
    def face(self):
        """
        :return:  4 * 15 * 4 的数组，作为当前状态
        """
        handcards = self.cards2arr(self.get_curr_handcards())
        known = self.batch_arr2onehot([handcards, self.taken])
        prob = self.get_state_prob().reshape(2, 15, 4)
        face = np.concatenate((known, prob))
        return torch.tensor(face, dtype=torch.float).to(DEVICE)

    def valid_actions(self, tensor=True):
        """
        :return:  batch_size * 15 * 4 的可行动作集合
        """
        self.old_cards = self.get_curr_handcards()
        handcards = self.cards2arr(self.old_cards)
        last_two = self.get_last_two_cards()
        if last_two[0]:
            last = last_two[0]
        elif last_two[1]:
            last = last_two[1]
        else:
            last = []
        last = self.cards2arr(last)
        actions = r.get_moves(handcards, last)
        if tensor:
            return torch.tensor(self.batch_arr2onehot(actions),
                                dtype=torch.float).to(DEVICE)
        else:
            return actions

    @classmethod
    def arr2cards(cls, arr):
        """
        :param arr: 15 * 4
        :return: ['A','A','A', '3', '3'] 用 [3,3,14,14,14]表示
            [3,4,5,6,7,8,9,10, J, Q, K, A, 2,BJ,CJ]
            [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        """
        res = []
        for idx in range(15):
            for _ in range(arr[idx]):
                res.append(idx + 3)
        return np.array(res, dtype=np.int)

    @classmethod
    def cards2arr(cls, cards):
        arr = np.zeros((15,), dtype=np.int)
        for card in cards:
            arr[card - 3] += 1
        return arr

    @classmethod
    def batch_arr2onehot(cls, batch_arr):
        res = np.zeros((len(batch_arr), 15, 4), dtype=np.int)
        for idx, arr in enumerate(batch_arr):
            for card_idx, count in enumerate(arr):
                if count > 0:
                    res[idx][card_idx][:int(count)] = 1
        return res

    @classmethod
    def onehot2arr(cls, onehot_cards):
        """
        :param onehot_cards: 15 * 4
        :return: (15,)
        """
        res = np.zeros((15,), dtype=np.int)
        for idx, onehot in enumerate(onehot_cards):
            res[idx] = sum(onehot)
        return res

    def cards2str(self, cards):
        res = [conf.DICT[i] for i in cards]
        return res
