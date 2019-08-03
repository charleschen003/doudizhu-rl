import time
from core_v0 import Env, CQL


def fight_with(policy, total=200, debug=True):
    assert policy in {'random', 'dhcp'}
    env = Env(debug=debug)
    print('Fight with {}'.format(policy))
    if policy == 'random':
        step = env.step_random
    else:
        step = env.step_auto
    lord = CQL()
    lord.policy_net.load('0802_1036_78500_44.bin')
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
                reward = 100
            else:
                _, done, _ = step()  # 下家
                if not done:
                    _, done, _ = step()  # 上家
                if done:  # 农民结束本局，地主输
                    total_farmer_win += 1
                    recent_farmer_win += 1
                    reward = -100
                else:  # 未结束，无奖励
                    reward = 0

        if episode % 10 == 0:
            end_time = time.time()
            print('Last 10 rounds takes {:.2f}seconds\n'
                  '\tLord recent 10 win rate: {:.2%}\n'
                  '\tLord total {} win rate: {:.2%}\n\n'
                  .format(end_time - start_time,
                          recent_lord_win / 10,
                          episode, total_lord_win / episode))
            win_rate_list.append(recent_lord_win)
            recent_lord_win, recent_farmer_win = 0, 0
            start_time = time.time()


if __name__ == '__main__':
    fight_with('random', debug=False)
