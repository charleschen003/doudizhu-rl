import os
import time
import json
import config as conf
import torch

BEGIN, logger, LOG_PATH = conf.get_logger()


class Game:
    def __init__(self, env_cls, nets_dict, dqns_dict, reward_dict=None,
                 train_dict=None, preload=None, seed=None, debug=False):
        if reward_dict is None:
            reward_dict = {'lord': 100, 'down': 50, 'up': 50}
        if train_dict is None:
            train_dict = {'lord': True, 'down': True, 'up': True}
        if preload is None:
            preload = {}
        assert not (nets_dict.keys() ^ dqns_dict.keys()), 'Net and DQN must match'

        self.lord_wins, self.down_wins, self.up_wins = [], [], []
        self.lord_total_loss = self.down_total_loss = self.up_total_loss = 0
        self.lord_loss_count = self.down_loss_count = self.up_loss_count = 0
        self.up_total_wins = self.lord_total_wins = self.down_total_wins = 0
        self.up_recent_wins = self.lord_recent_wins = self.down_recent_wins = 0

        self.env = env_cls(debug=debug, seed=seed)
        self.lord = self.down = self.up = None
        self.lord_train = self.down_train = self.up_train = False
        for role in ['lord', 'down', 'up']:
            if nets_dict.get(role):
                setattr(self, role, dqns_dict[role](nets_dict[role]))
                setattr(self, '{}_train'.format(role), train_dict[role])
                if preload.get(role):
                    getattr(self, role).target_net.load(preload.get(role))
                    getattr(self, role).policy_net.load(preload.get(role))

        self.lord_s0 = self.down_s0 = self.up_s0 = None
        self.lord_a0 = self.down_a0 = self.up_a0 = None
        self.reward_dict = reward_dict
        self.preload = preload
        self.train_dict = train_dict
        self.lord_max_wins = 0

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

    def save_win_rates(self, episode):
        self.lord_wins.append(self.lord_recent_wins)
        self.up_wins.append(self.up_recent_wins)
        self.down_wins.append(self.down_recent_wins)
        # 是否高于最高胜率
        if self.lord and self.up is None and self.down is None:
            if self.lord_recent_wins > self.lord_max_wins:
                self.lord_max_wins = self.lord_recent_wins
                self.lord.policy_net.save(
                    '{}_{}_{}'.format(BEGIN, episode, self.lord_max_wins))
        # 存一次胜率目录
        data = {'lord': self.lord_wins, 'down': self.down_wins, 'up': self.up_wins}
        path = os.path.join(conf.WIN_DIR, conf.name_dir(BEGIN))
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        path = '{}.json'.format(path)
        with open(path, 'w') as f:
            json.dump(data, f)

    def reset_recent(self):
        self.lord_recent_wins = self.up_recent_wins = self.down_recent_wins = 0
        self.lord_total_loss = self.down_total_loss = self.up_total_loss = 0
        self.lord_loss_count = self.down_loss_count = self.up_loss_count = 0

    def step(self, ai):
        assert ai in {'lord', 'down', 'up'}
        agent = getattr(self, ai)
        continue_train = getattr(self, '{}_train'.format(ai))
        if agent:  # 不是使用规则
            s0 = self.env.face
            if continue_train:  # 需要继续训练
                setattr(self, '{}_s0'.format(ai), s0)  # 更新状态s0
                action_f = agent.e_greedy_action
            else:
                action_f = agent.greedy_action
            a0 = action_f(s0, self.env.valid_actions())
            if continue_train:
                setattr(self, '{}_a0'.format(ai), a0)  # 更新动作a0
            _, done, _ = self.env.step_manual(a0)
        else:
            _, done, _ = self.env.step_auto()
        return done

    def feedback(self, ai, done, punish=False):
        assert ai in {'lord', 'up', 'down'}
        agent = getattr(self, ai)
        if agent and getattr(self, '{}_train'.format(ai)):  # 是需要继续训练的模型
            if done:
                reward = self.reward_dict[ai]
                if punish:
                    reward = -reward
            else:
                reward = 0
            s0 = getattr(self, '{}_s0'.format(ai))
            a0 = getattr(self, '{}_a0'.format(ai))
            s1 = self.env.face
            if done:
                a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
            else:
                a1 = agent.greedy_action(s1, self.env.valid_actions())
            loss = agent.perceive(s0, a0, reward, s1, a1, done)
            self.accumulate_loss(ai, loss)

    def lord_turn(self):
        done = self.step('lord')
        if not done:  # 本局未结束
            if self.down_a0 is not None:  # 如果下家曾经出过牌
                self.feedback('down', done)
        else:  # 本局结束，地主胜利
            if self.down_a0 is not None:  # 如果下家曾经出过牌（不是一次性走完）
                self.feedback('down', done, punish=True)  # 下家负反馈
                self.feedback('up', done, punish=True)  # 上家负反馈
            # 自己得到正反馈
            self.feedback('lord', done)
            self.lord_total_wins += 1
            self.lord_recent_wins += 1
        return done

    def down_turn(self):
        done = self.step('down')
        if not done:  # 本局未结束
            if self.up_a0 is not None:
                self.feedback('up', done)
        else:  # 本局结束，农民胜利
            self.feedback('up', done)
            self.feedback('lord', done, punish=True)
            self.feedback('down', done)
            self.down_recent_wins += 1
            self.down_total_wins += 1
        return done

    def up_turn(self):
        done = self.step('up')
        if not done:  # 本局未结束，地主得到0反馈
            self.feedback('lord', done)
        else:  # 本局结束，农民胜利
            self.feedback('lord', done, punish=True)  # 地主得到负反馈
            self.feedback('down', done)  # 下家得到正反馈
            self.feedback('up', done)  # 自己得到正反馈
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
        if not ((self.lord and self.lord_train)
                or (self.up and self.up_train)
                or (self.down and self.down_train)):
            print('No agent need train.')
            return
        print('Logged at {}'.format(LOG_PATH))
        messages = ''
        for role in ['up', 'lord', 'down']:
            m = '{}: {} based model.'.format(
                role, 'AI' if getattr(self, role) else 'Rule')
            if getattr(self, role):
                preload = self.preload.get(role)
                if preload:
                    m += ' With pretrained model {}.'.format(preload)
                else:
                    m += ' Without pretrained model.'
                if self.train_dict.get(role):
                    m += ' Continue training.'
            messages += '\n{}'.format(m)
        logger.info(messages + '\n------------------------------------')
        print(messages)
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
                self.save_win_rates(episode)
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

    @staticmethod
    def compete(env_cls, nets_dict, dqns_dict, model_dict, total=1000,
                print_every=100, debug=True):
        import collections
        assert not (nets_dict.keys() ^ dqns_dict.keys()), 'Net and DQN must match'
        assert not (nets_dict.keys() ^ model_dict.keys()), 'Net and Model must match'
        wins = collections.Counter()
        total_wins = collections.Counter()
        ai = {'up': None, 'lord': None, 'down': None}
        for role in ['up', 'lord', 'down']:
            if nets_dict.get(role) is not None:
                print('AI based {}.'.format(role))
                ai[role] = dqns_dict[role](nets_dict[role])
                ai[role].policy_net.load(model_dict[role])
            else:
                print('Rule based {}.'.format(role))

        env = env_cls(debug=debug)
        start_time = time.time()
        for episode in range(1, total + 1):
            if debug:
                print('\n-------------------------------------------')
            env.reset()
            env.prepare()
            done = False
            while not done:
                for role in ['lord', 'down', 'up']:
                    if ai[role]:
                        action = ai[role].greedy_action(env.face, env.valid_actions())
                        _, done, _ = env.step_manual(action)
                    else:
                        _, done, _ = env.step_auto()
                    if done:  # 地主结束本局，地主赢
                        wins[role] += 1
                        total_wins[role] += 1
                        break

            if episode % print_every == 0:
                end_time = time.time()
                message = ('Reach at {}, Last {} rounds takes {:.2f}seconds\n'
                           '\tUp  recent/total win rate: {:.2%}/{:.2%}\n'
                           '\tLord recent/total win rate: {:.2%}/{:.2%}\n'
                           '\tDown recent/total win rate: {:.2%}/{:.2%}\n')
                args = (episode, print_every, end_time - start_time,
                        wins['up'] / print_every, total_wins['up'] / episode,
                        wins['lord'] / print_every, total_wins['lord'] / episode,
                        wins['down'] / print_every, total_wins['down'] / episode)
                print(message.format(*args))
                wins = collections.Counter()
                start_time = time.time()
        return total_wins
