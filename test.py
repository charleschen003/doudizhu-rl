from game import Game


def e1():
    from envi import EnvComplicated
    from net import NetMoreComplicated
    from dqn import DQNFirst

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
    from envi import Env
    from net import NetComplicated
    from dqn import DQNFirst

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
    from envi import Env, EnvComplicated
    from net import NetComplicated, NetMoreComplicated
    from dqn import DQNFirst

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


if __name__ == '__main__':
    res = e_0806_1906_lord()
