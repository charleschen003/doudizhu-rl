import os
import time
import json
import config as conf
import torch
from envi import Env

BEGIN, logger = conf.get_logger()
BEGIN = 'feature_' + BEGIN

LORD_WINS, DOWN_WINS, UP_WINS = [], [], []
LORD_TOTAL_LOSS = DOWN_TOTAL_LOSS = UP_TOTAL_LOSS = 0
LORD_LOSS_COUNT = DOWN_LOSS_COUNT = UP_LOSS_COUNT = 0


def check_loss(name, loss):
    global LORD_TOTAL_LOSS, DOWN_TOTAL_LOSS, UP_TOTAL_LOSS
    global LORD_LOSS_COUNT, DOWN_LOSS_COUNT, UP_LOSS_COUNT
    assert name in {'up', 'down', 'lord'}
    if loss:
        if name == 'lord':
            LORD_LOSS_COUNT += 1
            LORD_TOTAL_LOSS += loss
        elif name == 'down':
            DOWN_LOSS_COUNT += 1
            DOWN_TOTAL_LOSS += loss
        else:
            UP_LOSS_COUNT += 1
            UP_TOTAL_LOSS += loss


def reset_loss():
    global LORD_TOTAL_LOSS, DOWN_TOTAL_LOSS, UP_TOTAL_LOSS
    global LORD_LOSS_COUNT, DOWN_LOSS_COUNT, UP_LOSS_COUNT
    LORD_TOTAL_LOSS = DOWN_TOTAL_LOSS = UP_TOTAL_LOSS = 0
    LORD_LOSS_COUNT = DOWN_LOSS_COUNT = UP_LOSS_COUNT = 0


def record(up, lord, down):
    global LORD_WINS, DOWN_WINS, UP_WINS
    LORD_WINS.append(lord)
    DOWN_WINS.append(down)
    UP_WINS.append(up)

    # 存一次胜率目录
    data = {'lord': LORD_WINS, 'down': DOWN_WINS, 'up': UP_WINS}
    path = os.path.join(conf.WIN_DIR, conf.name_dir(BEGIN))
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    path = '{}.json'.format(path)
    with open(path, 'w') as f:
        json.dump(data, f)


def train_from_scratch(net_cls, dqn_cls, total=3000, debug=False):
    global LORD_TOTAL_LOSS, DOWN_TOTAL_LOSS, UP_TOTAL_LOSS
    global LORD_LOSS_COUNT, DOWN_LOSS_COUNT, UP_LOSS_COUNT
    up_all = lord_all = down_all = 0
    up_recent = lord_recent = down_recent = 0
    env = Env(debug=debug, manual_peasant=True)
    lord = dqn_cls(net_cls)
    down = dqn_cls(net_cls)
    up = dqn_cls(net_cls)

    start_time = time.time()
    for episode in range(1, total + 1):
        env.reset()
        env.prepare()
        lord_s0 = down_s0 = up_s0 = None
        lord_a0 = down_a0 = up_a0 = None
        while True:  #
            # 地主先开始
            lord_s0 = env.face
            lord_a0 = lord.e_greedy_action(lord_s0, env.valid_actions())
            _, done, _ = env.step_manual(lord_a0)
            if not done:  # 本局未结束，下家得到0反馈
                if down_s0 is not None:
                    down_a1 = down.greedy_action(env.face, env.valid_actions())
                    down_loss = down.perceive(down_s0, down_a0, 0, env.face, down_a1, done)
                    check_loss('down', down_loss)
            else:  # 本局结束，地主胜利
                if down_s0 is not None:  # 非春天走完
                    # 下家得到负反馈
                    down_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                    down_loss = down.perceive(down_s0, down_a0, -50, env.face, down_a1, done)
                    check_loss('down', down_loss)
                    # 上家得到负反馈
                    up_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                    up_loss = up.perceive(up_s0, up_a0, -50, env.face, up_a1, done)
                    check_loss('up', up_loss)
                # 自己得到正反馈
                lord_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                lord_loss = lord.perceive(lord_s0, lord_a0, 100, env.face, lord_a1, done)
                check_loss('lord', lord_loss)
                lord_all += 1
                lord_recent += 1
                break

            # 随后下家开始
            down_s0 = env.face
            down_a0 = down.e_greedy_action(down_s0, env.valid_actions())
            _, done, _ = env.step_manual(down_a0)
            if not done:  # 本局未结束，上家得到0反馈
                if up_s0 is not None:
                    up_a1 = up.greedy_action(env.face, env.valid_actions())
                    up_loss = up.perceive(up_s0, up_a0, 0, env.face, up_a1, done)
                    check_loss('up', up_loss)
            else:  # 本局结束，农民胜利
                # 上家得到正反馈
                up_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                up_loss = up.perceive(up_s0, up_a0, 50, env.face, up_a1, done)
                check_loss('up', up_loss)

                # 地主得到负反馈
                lord_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                lord_loss = lord.perceive(lord_s0, lord_a0, -100, env.face, lord_a1, done)
                check_loss('lord', lord_loss)

                # 自己得到正反馈
                down_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                down_loss = down.perceive(down_s0, down_a0, 50, env.face, down_a1, done)
                check_loss('down', down_loss)
                down_recent += 1
                down_all += 1
                break

            # 最后上家出牌
            up_s0 = env.face
            up_a0 = up.e_greedy_action(up_s0, env.valid_actions())
            _, done, _ = env.step_manual(up_a0)  # 上家
            if not done:  # 本局未结束，地主得到0反馈
                lord_a1 = lord.greedy_action(env.face, env.valid_actions())
                lord_loss = lord.perceive(lord_s0, lord_a0, 0, env.face, lord_a1, done)
                check_loss('lord', lord_loss)
            else:  # 本局结束，农民胜利
                # 地主得到负反馈
                lord_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                lord_loss = lord.perceive(lord_s0, lord_a0, -100, env.face, lord_a1, done)
                check_loss('lord', lord_loss)
                # 下家得到正反馈
                down_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                down_loss = down.perceive(down_s0, down_a0, 50, env.face, down_a1, done)
                check_loss('down', down_loss)
                # 自己得到正反馈
                up_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                up_loss = up.perceive(up_s0, up_a0, 50, env.face, up_a1, done)
                check_loss('up', up_loss)
                up_recent += 1
                up_all += 1
                break

        # print(env.left)
        if episode % 100 == 0:
            end_time = time.time()
            logger.info('Last 100 rounds takes {:.2f}seconds\n'
                        '\tUp   recent/total win: {}/{} [{:.2f}]\n'
                        '\tLord recent/total win: {}/{} [{:.2f}]\n'
                        '\tDown recent/total win: {}/{} [{:.2f}]\n'
                        .format(end_time - start_time,
                                up_recent, up_all, UP_TOTAL_LOSS / (UP_LOSS_COUNT + 1e-3),
                                lord_recent, lord_all, LORD_TOTAL_LOSS / (LORD_LOSS_COUNT + 1e-3),
                                down_recent, down_all, DOWN_TOTAL_LOSS / (DOWN_LOSS_COUNT + 1e-3)))
            up_recent = lord_recent = down_recent = 0
            record(up_recent, lord_recent, down_recent)
            reset_loss()
            start_time = time.time()
        if episode % 10000 == 0:
            lord.policy_net.save('{}_lord_{}'.format(BEGIN, episode), 3)
            down.policy_net.save('{}_down_{}'.format(BEGIN, episode), 3)
            up.policy_net.save('{}_up_{}'.format(BEGIN, episode), 3)
        lord.update_epsilon(episode)
        lord.update_target(episode)
        down.update_epsilon(episode)
        down.update_target(episode)
        up.update_epsilon(episode)
        up.update_target(episode)


def train_from_model(models, net_cls, dqn_cls, total=3000, debug=False):
    global LORD_TOTAL_LOSS, DOWN_TOTAL_LOSS, UP_TOTAL_LOSS
    global LORD_LOSS_COUNT, DOWN_LOSS_COUNT, UP_LOSS_COUNT
    up_all = lord_all = down_all = 0
    up_recent = lord_recent = down_recent = 0
    env = Env(debug=debug, manual_peasant=True)
    lord = dqn_cls(net_cls)
    if models.get('lord'):
        print('Lord load pretrained model')
        lord.policy_net.load(models['lord'])
        lord.target_net.load(models['lord'])
    down = dqn_cls(net_cls)
    if models.get('down'):
        print('Down load pretrained model')
        down.policy_net.load(models['down'])
        down.target_net.load(models['down'])
    up = dqn_cls(net_cls)
    if models.get('up'):
        print('Up load pretrained model')
        up.policy_net.load(models['up'])
        up.target_net.load(models['up'])

    start_time = time.time()
    for episode in range(1, total + 1):
        env.reset()
        env.prepare()
        lord_s0 = down_s0 = up_s0 = None
        lord_a0 = down_a0 = up_a0 = None
        while True:  #
            # 地主先开始
            lord_s0 = env.face
            lord_a0 = lord.e_greedy_action(lord_s0, env.valid_actions())
            _, done, _ = env.step_manual(lord_a0)
            if not done:  # 本局未结束，下家得到0反馈
                if down_s0 is not None:
                    down_a1 = down.greedy_action(env.face, env.valid_actions())
                    down_loss = down.perceive(down_s0, down_a0, 0, env.face, down_a1, done)
                    check_loss('down', down_loss)
            else:  # 本局结束，地主胜利
                if down_s0 is not None:  # 非春天走完
                    # 下家得到负反馈
                    down_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                    down_loss = down.perceive(down_s0, down_a0, -50, env.face, down_a1, done)
                    check_loss('down', down_loss)
                    # 上家得到负反馈
                    up_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                    up_loss = up.perceive(up_s0, up_a0, -50, env.face, up_a1, done)
                    check_loss('up', up_loss)
                # 自己得到正反馈
                lord_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                lord_loss = lord.perceive(lord_s0, lord_a0, 100, env.face, lord_a1, done)
                check_loss('lord', lord_loss)
                lord_all += 1
                lord_recent += 1
                break

            # 随后下家开始
            down_s0 = env.face
            down_a0 = down.e_greedy_action(down_s0, env.valid_actions())
            _, done, _ = env.step_manual(down_a0)
            if not done:  # 本局未结束，上家得到0反馈
                if up_s0 is not None:
                    up_a1 = up.greedy_action(env.face, env.valid_actions())
                    up_loss = up.perceive(up_s0, up_a0, 0, env.face, up_a1, done)
                    check_loss('up', up_loss)
            else:  # 本局结束，农民胜利
                # 上家得到正反馈
                up_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                up_loss = up.perceive(up_s0, up_a0, 50, env.face, up_a1, done)
                check_loss('up', up_loss)

                # 地主得到负反馈
                lord_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                lord_loss = lord.perceive(lord_s0, lord_a0, -100, env.face, lord_a1, done)
                check_loss('lord', lord_loss)

                # 自己得到正反馈
                down_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                down_loss = down.perceive(down_s0, down_a0, 50, env.face, down_a1, done)
                check_loss('down', down_loss)
                down_recent += 1
                down_all += 1
                break

            # 最后上家出牌
            up_s0 = env.face
            up_a0 = up.e_greedy_action(up_s0, env.valid_actions())
            _, done, _ = env.step_manual(up_a0)  # 上家
            if not done:  # 本局未结束，地主得到0反馈
                lord_a1 = lord.greedy_action(env.face, env.valid_actions())
                lord_loss = lord.perceive(lord_s0, lord_a0, 0, env.face, lord_a1, done)
                check_loss('lord', lord_loss)
            else:  # 本局结束，农民胜利
                # 地主得到负反馈
                lord_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                lord_loss = lord.perceive(lord_s0, lord_a0, -100, env.face, lord_a1, done)
                check_loss('lord', lord_loss)
                # 下家得到正反馈
                down_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                down_loss = down.perceive(down_s0, down_a0, 50, env.face, down_a1, done)
                check_loss('down', down_loss)
                # 自己得到正反馈
                up_a1 = torch.zeros((15, 4), dtype=torch.float).to(conf.DEVICE)
                up_loss = up.perceive(up_s0, up_a0, 50, env.face, up_a1, done)
                check_loss('up', up_loss)
                up_recent += 1
                up_all += 1
                break

        # print(env.left)
        if episode % 100 == 0:
            end_time = time.time()
            logger.info('Last 100 rounds takes {:.2f}seconds\n'
                        '\tUp   recent/total win: {}/{} [{:.2f}]\n'
                        '\tLord recent/total win: {}/{} [{:.2f}]\n'
                        '\tDown recent/total win: {}/{} [{:.2f}]\n'
                        .format(end_time - start_time,
                                up_recent, up_all, UP_TOTAL_LOSS / (UP_LOSS_COUNT + 1e-3),
                                lord_recent, lord_all, LORD_TOTAL_LOSS / (LORD_LOSS_COUNT + 1e-3),
                                down_recent, down_all, DOWN_TOTAL_LOSS / (DOWN_LOSS_COUNT + 1e-3)))
            up_recent = lord_recent = down_recent = 0
            record(up_recent, lord_recent, down_recent)
            reset_loss()
            start_time = time.time()
        if episode % 10000 == 0:
            lord.policy_net.save('{}_lord_{}'.format(BEGIN, episode), 3)
            down.policy_net.save('{}_down_{}'.format(BEGIN, episode), 3)
            up.policy_net.save('{}_up_{}'.format(BEGIN, episode), 3)
        lord.update_epsilon(episode)
        lord.update_target(episode)
        down.update_epsilon(episode)
        down.update_target(episode)
        up.update_epsilon(episode)
        up.update_target(episode)


if __name__ == '__main__':
    from net import NetFirst, NetComplicated
    from dqn import DQNFirst
    train_from_model({'lord': '0804_1423_2400_49'},
                     NetComplicated, DQNFirst,
                     total=3000, debug=True)
