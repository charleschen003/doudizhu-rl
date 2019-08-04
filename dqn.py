import config as conf
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque


class DQNFirst:
    def __init__(self, net_cls):
        super(DQNFirst, self).__init__()
        self.epsilon = conf.EPSILON_HIGH
        self.replay_buffer = deque(maxlen=conf.REPLAY_SIZE)

        self.policy_net = net_cls().to(conf.DEVICE)
        self.target_net = net_cls().to(conf.DEVICE)
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
        r1 = torch.tensor(r1, dtype=torch.float).view(conf.BATCH_SIZE, -1)\
            .to(conf.DEVICE)
        s1 = torch.stack(s1)
        a1 = torch.stack(a1)
        done = torch.tensor(done, dtype=torch.float).view(conf.BATCH_SIZE, -1)\
            .to(conf.DEVICE)

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
