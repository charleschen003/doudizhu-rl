import sys

sys.path.insert(0, 'precompiled')

import config_v0 as conf
from env import Env as CEnv


def random_play():
    # 地主胜率 50%
    env = CEnv()
    total_lord_win, total_farmer_win = 0, 0
    lord_win, farmer_win = 0, 0
    for episode in range(1, conf.EPISODE + 1):
        if episode % 100 == 0:
            print('-----------------------------'
                  'Lord win rate: \n'
                  '\tRecent 100：{:.2%}\n'
                  '\tTotal {}: {:.2%}\n\n'
                  .format(lord_win / 100, episode, total_lord_win / episode))
            lord_win, farmer_win = 0, 0
        env.reset()
        env.prepare()
        r = 0
        while r == 0:  # r == -1 地主赢， r == 1，农民赢
            # lord first
            action, r, _ = env.step_auto()
            print(action)
            if r == -1:  # 地主赢
                lord_win += 1
                total_lord_win += 1
            else:
                _, r, _ = env.step_auto()  # 下家
                if r == 0:
                    _, r, _ = env.step_auto()  # 上家
                if r == 1:  # 地主输
                    farmer_win += 1
                    total_farmer_win += 1
        input()


if __name__ == '__main__':
    random_play()
