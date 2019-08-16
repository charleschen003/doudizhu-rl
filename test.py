from game import Game
from envi import Env, EnvComplicated, EnvCooperation, EnvCooperationSimplify
from net import NetComplicated, NetMoreComplicated, NetCooperation, NetCooperationSimplify
from dqn import DQNFirst


def e1():
    net_dict = {
        'lord': NetMoreComplicated,
    }
    dqn_dict = {
        'lord': DQNFirst,
    }
    model_dict = {
        'lord': '0805_1409_lord_4000',
    }
    wins = Game.compete(EnvComplicated, net_dict, dqn_dict, model_dict,
                        total=2000, print_every=100, debug=False)
    print(wins)


def e2():
    net_dict = {
        'lord': NetComplicated,
    }
    dqn_dict = {
        'lord': DQNFirst,
    }
    model_dict = {
        'lord': '0804_2022_lord_scratch3000',
    }
    wins = Game.compete(Env, net_dict, dqn_dict, model_dict,
                        total=1000, print_every=100, debug=False)
    print(wins)


def e_0806_1906_lord():
    net_dict = {
        'lord': NetMoreComplicated,
    }
    dqn_dict = {
        'lord': DQNFirst,
    }
    wins = {}
    for model in [3, 4]:
        model_dict = {
            'lord': '0806_1906_lord_{}000'.format(model),
        }
        win = Game.compete(EnvComplicated, net_dict, dqn_dict, model_dict,
                           total=1000, print_every=100, debug=False)
        wins[model] = win
    return wins


def e_0807_1340():
    net_dict = {
        'lord': None,
        'down': NetCooperation,
        'up': NetCooperation,
    }
    dqn_dict = {
        'lord': None,
        'down': DQNFirst,
        'up': DQNFirst,
    }
    model_dict = {
        'lord': None,
        'down': '0807_1344_down_3000',
        'up': '0807_1344_up_3000',
    }
    win = Game.compete(EnvCooperation, net_dict, dqn_dict, model_dict,
                       total=1000, print_every=100, debug=False)
    return win


def e0808():
    net_dict = {
        'lord': None,
        'down': NetCooperationSimplify,
        'up': NetCooperationSimplify,
    }
    dqn_dict = {
        'lord': None,
        'down': DQNFirst,
        'up': DQNFirst,
    }
    model_dict = {
        'lord': None,
        'down': '0808_0854_down_6000',
        'up': '0808_0854_up_6000',
    }
    win = Game.compete(EnvCooperationSimplify, net_dict, dqn_dict, model_dict,
                       total=1000, print_every=100, debug=False)
    return win


def e_ensemble():
    # 纯RL，58.7%，1000把
    from ensemble import Game
    net_dict = {
        'lord': None,
        'down': None,
        'up': None,
    }
    dqn_dict = {
        'lord': None,
        'down': None,
        'up': None,
    }
    model_dict = {
        'lord': None,
        'down': None,
        'up': None,
    }
    win = Game.ensemble_compete(EnvCooperationSimplify, net_dict, dqn_dict, model_dict,
                                total=1000, print_every=1, debug=False)
    return win


# if __name__ == '__main__':
res = e_ensemble()  # 246：胜利148
