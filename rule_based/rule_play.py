from envi import Env
from rule_based.utils.rule_based_model import RuleBasedModel


def rule_play():
    env = Env()
    rule = RuleBasedModel()
    total_lord_win, total_farmer_win = 0, 0
    for episode in range(1, 3000 + 1):
        # print(episode)
        env.reset()
        env.prepare()
        r = 0
        while r == 0:  # r == -1 地主赢， r == 1，农民赢
            # lord first
            r, _, _ = env.step_manual(rule.choose(env))
            if r == -1:  # 地主赢
                total_lord_win += 1
            else:
                h = env.get_curr_handcards()
                a, r, _ = env.step_auto()  # 下家
                print("Auto1", h, "//", a)
                if r == 0:
                    h = env.get_curr_handcards()
                    a, r, _ = env.step_auto()  # 上家
                    print("Auto2", h, "//", a)
                if r == 1:  # 地主输
                    total_farmer_win += 1
        print('\nLord win rate: {} / {} = {:.2%}\n\n'
              .format(total_lord_win, episode, total_lord_win / episode))


if __name__ == '__main__':
    rule_play()
