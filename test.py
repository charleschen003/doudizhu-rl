from game import Game
from envi import Env, EnvComplicated, EnvCooperation
from net import NetComplicated, NetMoreComplicated, NetCooperation
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


# if __name__ == '__main__':
res = e_0807_1340()
