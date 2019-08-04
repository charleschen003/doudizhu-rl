import os
import time
import json
import config as conf
import torch
from envi import Env

BEGIN, logger = conf.get_logger()


def train(net_cls, dqn_cls, total=3000, debug=False):
    env = Env(debug=debug)
    lord = dqn_cls(net_cls)
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
                next_action = torch.zeros((15, 4), dtype=torch.float) \
                    .to(conf.DEVICE)
            else:
                next_action = lord.greedy_action(env.face, env.valid_actions())
            loss = lord.perceive(state, action, reward, env.face, next_action, done)
            if loss is not None:
                loss_times += 1
                total_loss += loss

        # print(env.left)
        if episode % 1 == 0:
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
                lord.policy_net.save('{}_{}_{}'.format(BEGIN, episode, max_win))
            win_rate_list.append(recent_lord_win)
            total_loss, loss_times = 0, 0
            recent_lord_win, recent_farmer_win = 0, 0
            start_time = time.time()
        if episode % 10 == 0:
            path = os.path.join(conf.WIN_DIR, conf.name_dir(BEGIN))
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            path = '{}.json'.format(path)
            with open(path, 'w') as f:
                json.dump(win_rate_list, f)

            lord.policy_net.save('{}_{}'.format(BEGIN, episode))
        lord.update_epsilon(episode)
        lord.update_target(episode)


if __name__ == '__main__':
    from net import NetFirst, NetComplicated
    from dqn import DQNFirst
    train(NetFirst, DQNFirst, total=3000, debug=False)
