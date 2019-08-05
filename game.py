import os
import time
import json
import config as conf
import torch

BEGIN, logger = conf.get_logger()


class Game:
    def __init__(self, env_cls, nets_dict, dqns_dict, reward=None,
                 preload=None, seed=None, debug=False):
        assert not (nets_dict.keys() ^ dqns_dict.keys()), 'Net and DQN must match'
        self.lord_wins, self.down_wins, self.up_wins = [], [], []
        self.lord_total_loss = self.down_total_loss = self.up_total_loss = 0
        self.lord_loss_count = self.down_loss_count = self.up_loss_count = 0
        self.up_total_wins = self.lord_total_wins = self.down_total_wins = 0
        self.up_recent_wins = self.lord_recent_wins = self.down_recent_wins = 0

        self.env = env_cls(debug=debug, seed=seed)
        self.lord = self.down = self.up = None
        for role in ['lord', 'down', 'up']:
            if nets_dict.get(role) is not None:
                setattr(self, role, dqns_dict[role](nets_dict[role]))

        self.lord_s0 = self.down_s0 = self.up_s0 = None
        self.lord_a0 = self.down_a0 = self.up_a0 = None

        if reward is None:
            reward = {'lord': 100, 'down': 50, 'up': 50}
        self.r = reward

    def accumulate_loss(self, name, loss):
        assert name in {'up', 'down', 'lord'}
        if loss:
            if name == 'lord':
                self.lord_loss_count += 1
                self.lord_total_loss += loss
            elif name == 'down':
                self.down_loss_count += 1
                self.down_total_loss += loss
            else:
                self.up_loss_count += 1
                self.up_total_loss += loss

    def save_win_rates(self):
        self.lord_wins.append(self.lord_recent_wins)
        self.up_wins.append(self.up_recent_wins)
        self.down_wins.append(self.down_recent_wins)
        # 存一次胜率目录
        data = {'lord': self.lord_wins, 'down': self.down_wins, 'up': self.up_wins}
        path = os.path.join(conf.WIN_DIR, conf.name_dir(BEGIN))
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        path = '{}.json'.format(path)
        with open(path, 'w') as f:
            json.dump(data, f)
        # 清理最近胜率

    def reset_recent(self):
        self.lord_recent_wins = self.up_recent_wins = self.down_recent_wins = 0
        self.lord_total_loss = self.down_total_loss = self.up_total_loss = 0
        self.lord_loss_count = self.down_loss_count = self.up_loss_count = 0

    def lord_turn(self):
        if self.lord:  # 地主使用模型
            self.lord_s0 = self.env.face
            self.lord_a0 = self.lord.e_greedy_action(
                self.lord_s0, self.env.valid_actions())
            _, done, _ = self.env.step_manual(self.lord_a0)
        else:
            _, done, _ = self.env.step_auto()
        if not done:  # 本局未结束，下家得到0反馈
            if self.down and self.down_s0:
                face = self.env.face
                down_a1 = self.down.greedy_action(face, self.env.valid_actions())
                down_loss = self.down.perceive(
                    self.down_s0, self.down_a0, 0, face, down_a1, done)
                self.accumulate_loss('down', down_loss)
        else:  # 本局结束，地主胜利
            if self.down_s0:  # 非春天走完
                # 下家得到负反馈
                if self.down:
                    down_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                    down_loss = self.down.perceive(
                        self.down_s0, self.down_a0, -self.r['down'],
                        self.env.face, down_a1, done)
                    self.accumulate_loss('down', down_loss)
                if self.up:
                    # 上家得到负反馈
                    up_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                    up_loss = self.up.perceive(
                        self.up_s0, self.up_a0, -self.r['up'],
                        self.env.face, up_a1, done)
                    self.accumulate_loss('up', up_loss)
            if self.lord:
                # 自己得到正反馈
                lord_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                lord_loss = self.lord.perceive(
                    self.lord_s0, self.lord_a0, self.r['lord'],
                    self.env.face, lord_a1, done)
                self.accumulate_loss('lord', lord_loss)
            self.lord_total_wins += 1
            self.lord_recent_wins += 1
        return done

    def down_turn(self):
        # 随后下家开始
        if self.down:
            self.down_s0 = self.env.face
            self.down_a0 = self.down.e_greedy_action(
                self.down_s0, self.env.valid_actions())
            _, done, _ = self.env.step_manual(self.down_a0)
        else:
            _, done, _ = self.env.step_auto()
        if not done:  # 本局未结束，上家得到0反馈
            if self.up and self.up_s0:
                face = self.env.face
                up_a1 = self.up.greedy_action(face, self.env.valid_actions())
                up_loss = self.up.perceive(
                    self.up_s0, self.up_a0, 0, face, up_a1, done)
                self.accumulate_loss('up', up_loss)
        else:  # 本局结束，农民胜利
            # 上家得到正反馈
            if self.up:
                up_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                up_loss = self.up.perceive(
                    self.up_s0, self.up_a0, self.r['up'],
                    self.env.face, up_a1, done)
                self.accumulate_loss('up', up_loss)

            # 地主得到负反馈
            if self.lord:
                lord_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                lord_loss = self.lord.perceive(
                    self.lord_s0, self.lord_a0, -self.r['lord'],
                    self.env.face, lord_a1, done)
                self.accumulate_loss('lord', lord_loss)

            # 自己得到正反馈
            if self.down:
                down_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                down_loss = self.down.perceive(
                    self.down_s0, self.down_a0, self.r['down'],
                    self.env.face, down_a1, done)
                self.accumulate_loss('down', down_loss)
            self.down_recent_wins += 1
            self.down_total_wins += 1
        return done

    def up_turn(self):
        # 最后上家出牌
        if self.up:
            self.up_s0 = self.env.face
            self.up_a0 = self.up.e_greedy_action(
                self.up_s0, self.env.valid_actions())
            _, done, _ = self.env.step_manual(self.up_a0)  # 上家
        else:
            _, done, _ = self.env.step_auto()
        if not done:  # 本局未结束，地主得到0反馈
            if self.lord:
                face = self.env.face
                lord_a1 = self.lord.greedy_action(face, self.env.valid_actions())
                lord_loss = self.lord.perceive(
                    self.lord_s0, self.lord_a0, 0, face, lord_a1, done)
                self.accumulate_loss('lord', lord_loss)
        else:  # 本局结束，农民胜利
            # 地主得到负反馈
            if self.lord:
                lord_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                lord_loss = self.lord.perceive(
                    self.lord_s0, self.lord_a0, -self.r['lord'],
                    self.env.face, lord_a1, done)
                self.accumulate_loss('lord', lord_loss)
            # 下家得到正反馈
            if self.down:
                down_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                down_loss = self.down.perceive(
                    self.down_s0, self.down_a0, self.r['down'],
                    self.env.face, down_a1, done)
                self.accumulate_loss('down', down_loss)
            # 自己得到正反馈
            if self.up:
                up_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                up_loss = self.up.perceive(
                    self.up_s0, self.up_a0, self.r['up'],
                    self.env.face, up_a1, done)
                self.accumulate_loss('up', up_loss)
            self.up_total_wins += 1
            self.up_recent_wins += 1
        return done

    def play(self):
        self.env.reset()
        self.env.prepare()
        while True:  #
            done = self.lord_turn()
            if done:
                break
            done = self.down_turn()
            if done:
                break
            done = self.up_turn()
            if done:
                break

    def train(self, episodes, log_every=100, model_every=1000):
        start_time = time.time()
        for episode in range(1, episodes + 1):
            self.play()

            if episode % log_every == 0:
                end_time = time.time()
                message = (
                    'Reach at round {}, recent {} rounds takes {:.2f}seconds\n'
                    '\tUp   recent/total win: {:.2%}/{:.2%} [Mean loss: {:.2f}]\n'
                    '\tLord recent/total win: {:.2%}/{:.2%} [Mean loss: {:.2f}]\n'
                    '\tDown recent/total win: {:.2%}/{:.2%} [Mean loss: {:.2f}]\n'
                ).format(episode, log_every, end_time - start_time,
                         self.up_recent_wins / log_every, self.up_total_wins / episode,
                         self.up_total_loss / (self.up_loss_count + 1e-3),
                         self.lord_recent_wins / log_every, self.lord_total_wins / episode,
                         self.lord_total_loss / (self.lord_loss_count + 1e-3),
                         self.down_recent_wins / log_every, self.down_total_wins / episode,
                         self.down_total_loss / (self.down_loss_count + 1e-3))
                logger.info(message)
                self.save_win_rates()
                self.reset_recent()
                start_time = time.time()
            if episode % model_every == 0:
                for role in ['lord', 'down', 'up']:
                    ai = getattr(self, role)
                    if ai:
                        ai.policy_net.save(
                            '{}_{}_{}'.format(BEGIN, role, episode))

            for role in ['lord', 'down', 'up']:
                ai = getattr(self, role)
                if ai:
                    ai.update_epsilon(episode)
                    ai.update_target(episode)
