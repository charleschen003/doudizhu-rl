import sys
import time
import json
import config as conf
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter, deque
import logging
import os

WORK_DIR, _ = os.path.split(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WORK_DIR, 'precompiled'))
import r
from env import Env as CEnv

lt = time.localtime(time.time())
BEGIN = '{:0>2d}{:0>2d}_{:0>2d}{:0>2d}'.format(
    lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min)


def fn():
    res = os.path.join(WORK_DIR, 'outs', '{}.log'.format(BEGIN))
    return res


logger = logging.getLogger('DDZ_RL')
logger.setLevel(logging.INFO)
logging.basicConfig(filename=fn(), filemode='w',
                    format='[%(asctime)s][%(name)s][%(levelname)s]:  %(message)s')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class Net(nn.Module):
    def __init__(self):
        # input shape: 5 * 15 * 4
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(5, 256, (1, 1), (1, 4))  # 256 * 15 * 1
        self.conv2 = nn.Conv2d(5, 256, (1, 2), (1, 4))
        self.conv3 = nn.Conv2d(5, 256, (1, 3), (1, 4))
        self.conv4 = nn.Conv2d(5, 256, (1, 4), (1, 4))
        self.convs = (self.conv1, self.conv2, self.conv3, self.conv4)  # 256 * 15 * 4
        self.conv_shunzi = nn.Conv2d(5, 256, (15, 1), 1)  # 256 * 1 * 4
        self.pool = nn.MaxPool2d((1, 4))  # 256 * 15 * 1
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * (15 + 4), 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, face, actions):
        """
        :param face: 当前状态  4 * 15 * 4
        :param actions: 所有动作 batch_size * 15 * 4
        :return:
        """
        if face.dim() == 3:
            face = face.unsqueeze(0).repeat((actions.shape[0], 1, 1, 1))
        actions = actions.unsqueeze(1)
        state_action = torch.cat((face, actions), dim=1)

        x = torch.cat([f(state_action) for f in self.convs], -1)
        x = self.pool(x)
        x = x.view(actions.shape[0], -1)

        x_shunzi = self.conv_shunzi(state_action).view(actions.shape[0], -1)
        x = torch.cat([x, x_shunzi], -1)

        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, name, folder=None):
        if folder is None:
            folder = os.path.join(WORK_DIR, 'models')
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, name)
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(self.state_dict(), path)

    def load(self, name, folder=None):
        if folder is None:
            folder = os.path.join(WORK_DIR, 'models')
        path = os.path.join(folder, name)
        map_location = 'cpu' if DEVICE.type == 'cpu' else 'gpu'
        static_dict = torch.load(path, map_location)
        self.load_state_dict(static_dict)
        self.eval()


class CQL:
    def __init__(self):
        super(CQL, self).__init__()
        self.epsilon = conf.EPSILON_HIGH
        self.replay_buffer = deque(maxlen=conf.REPLAY_SIZE)

        self.policy_net = Net().to(DEVICE)
        self.target_net = Net().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), 1e-4)

    def perceive(self, state, action, reward, next_state, next_action, done):
        self.replay_buffer.append((
            state, action, reward, next_state, next_action, done))
        if len(self.replay_buffer) < conf.BATCH_SIZE:
            return None

        # training
        samples = random.sample(self.replay_buffer, conf.BATCH_SIZE)
        s0, a0, r1, s1, a1, done = zip(*samples)
        s0 = torch.stack(s0)
        a0 = torch.stack(a0)
        r1 = torch.tensor(r1, dtype=torch.float).view(conf.BATCH_SIZE, -1).to(DEVICE)
        s1 = torch.stack(s1)
        a1 = torch.stack(a1)
        done = torch.tensor(done, dtype=torch.float).view(conf.BATCH_SIZE, -1).to(DEVICE)

        s1_reward = self.target_net(s1, a1).detach()
        y_true = r1 + (1 - done) * conf.GAMMA * s1_reward
        y_pred = self.policy_net(s0, a0)

        loss = nn.MSELoss()(y_true, y_pred)
        res = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return res

    def e_greedy_action(self, face, actions):
        """
        :param face: 当前状态  2 * 15 * 4
        :param actions: 所有动作 batch_size * 15 * 4
        :return: action: 选择的动作 15 * 4
        """
        q_value = self.policy_net(face, actions).detach()
        if random.random() <= self.epsilon:
            idx = np.random.randint(0, actions.shape[0])
        else:
            idx = torch.argmax(q_value).item()
        return actions[idx]

    def greedy_action(self, face, actions):
        """
        :param face: 当前状态  2 * 15 * 4
        :param actions: 所有动作 batch_size * 15 * 4
        :return: action: 选择的动作 15 * 4
        """
        q_value = self.policy_net(face, actions).detach()
        idx = torch.argmax(q_value).item()
        return actions[idx]

    def update_epsilon(self, episode):
        self.epsilon = conf.EPSILON_LOW + \
                       (conf.EPSILON_HIGH - conf.EPSILON_LOW) * \
                       np.exp(-1.0 * episode / conf.DECAY)

    def update_target(self, episode):
        if episode % conf.UPDATE_TARGET_EVERY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def lord_ai_play(total=3000, debug=False):
    env = Env(debug=debug)
    lord = CQL()
    max_win = -1
    win_rate_list = []
    total_loss, loss_times = 0, 0
    total_lord_win, total_farmer_win = 0, 0
    recent_lord_win, recent_farmer_win = 0, 0
    start_time = time.time()
    for episode in range(1, total + 1):
        print(episode)
        env.reset()
        env.prepare()
        done = False
        while not done:  # r == -1 地主赢， r == 1，农民赢
            # lord first
            state = env.face
            action = lord.e_greedy_action(state, env.valid_actions())
            _, done, _ = env.step_manual(action)
            if done:  # 地主结束本局，地主赢
                total_lord_win += 1
                recent_lord_win += 1
                reward = 100
            else:
                _, done, _ = env.step_auto()  # 下家
                if not done:
                    _, done, _ = env.step_auto()  # 上家
                if done:  # 农民结束本局，地主输
                    total_farmer_win += 1
                    recent_farmer_win += 1
                    reward = -100
                else:  # 未结束，无奖励
                    reward = 0
            if done:
                next_action = torch.zeros((15, 4), dtype=torch.float).to(DEVICE)
            else:
                next_action = lord.greedy_action(env.face, env.valid_actions())
            loss = lord.perceive(state, action, reward, env.face, next_action, done)
            if loss is not None:
                loss_times += 1
                total_loss += loss

        # print(env.left)

        if episode % 100 == 0:
            end_time = time.time()
            logger.info('Last 100 rounds takes {:.2f}seconds\n'
                        '\tLord recent 100 win rate: {:.2%}\n'
                        '\tLord total {} win rate: {:.2%}\n'
                        '\tMean Loss: {:.2f}\n\n'
                        .format(end_time - start_time,
                                recent_lord_win / 100,
                                episode, total_lord_win / episode,
                                total_loss / (loss_times + 0.001)))
            if recent_lord_win > max_win:
                max_win = recent_lord_win
                lord.policy_net.save('{}/{}_{}.bin'
                                     .format(BEGIN, episode, max_win))
            win_rate_list.append(recent_lord_win)
            total_loss, loss_times = 0, 0
            recent_lord_win, recent_farmer_win = 0, 0
            start_time = time.time()
        if episode % 1000 == 0:
            with open('outs/{}.json'.format(BEGIN), 'w') as f:
                json.dump(win_rate_list, f)
            lord.policy_net.save('{}/{}.bin'.format(BEGIN, episode))
        lord.update_epsilon(episode)
        lord.update_target(episode)


if __name__ == '__main__':
    lord_ai_play(debug=False)
