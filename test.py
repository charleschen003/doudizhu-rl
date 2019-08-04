import time
import dqn
import net
from envi import Env


def fight_with(policy, net_cls, dqn_cls, model, total, debug=True):
    assert policy in {'random', 'dhcp'}
    env = Env(debug=debug)
    print('Fight with {}'.format(policy))
    if policy == 'random':
        step = env.step_random
    else:
        step = env.step_auto
    lord = dqn_cls(net_cls)
    lord.policy_net.load(model)
    win_rate_list = []
    total_lord_win, total_farmer_win = 0, 0
    recent_lord_win, recent_farmer_win = 0, 0
    start_time = time.time()
    for episode in range(1, total + 1):
        env.reset()
        env.prepare()
        done = False
        while not done:  # r == -1 地主赢， r == 1，农民赢
            # lord first
            state = env.face
            action = lord.greedy_action(state, env.valid_actions())
            _, done, _ = env.step_manual(action)
            if done:  # 地主结束本局，地主赢
                total_lord_win += 1
                recent_lord_win += 1
            else:
                _, done, _ = step()  # 下家
                if not done:
                    _, done, _ = step()  # 上家
                if done:  # 农民结束本局，地主输
                    total_farmer_win += 1
                    recent_farmer_win += 1

        if episode % 100 == 0:
            end_time = time.time()
            print('Last 100 rounds takes {:.2f}seconds\n'
                  '\tLord recent 100 win rate: {:.2%}\n'
                  '\tLord total {} win rate: {:.2%}\n\n'
                  .format(end_time - start_time,
                          recent_lord_win / 100,
                          episode, total_lord_win / episode))
            win_rate_list.append(recent_lord_win)
            recent_lord_win, recent_farmer_win = 0, 0
            start_time = time.time()


def test_1():
    for policy in ['random', 'dhcp']:
        for model in ['0804_0112_2700_51', '0804_0112_3800_53',
                      '0804_0112_4500_57']:
            fight_with(policy, net.NetFirst, dqn.DQNFirst, model,
                       total=1000, debug=False)


def test_2():
    for policy in ['dhcp', 'random']:
        for model in ['0804_0245_2800_48', '0804_0245_3500_53',
                      '0804_0245_4600_57']:
            fight_with(policy, net.NetComplicated, dqn.DQNFirst, model,
                       total=1000, debug=False)


def test_3():
    for policy in ['dhcp', 'random']:
        for model in ['0804_1423_2400_49', '0804_1423_3700_54',
                      '0804_1423_6000']:
            fight_with(policy, net.NetComplicated, dqn.DQNFirst, model,
                       total=1000, debug=False)


if __name__ == '__main__':
    # TODO 根据model自行确定要调用的net 和dqn
    test_3()
