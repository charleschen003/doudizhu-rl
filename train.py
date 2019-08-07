from game import Game

if __name__ == '__main__':
    from envi import EnvCooperation
    from net import NetCooperation
    from dqn import DQNFirst

    net_dict = {
        'lord': NetCooperation,
        'down': None,
        'up': None,
    }
    dqn_dict = {
        'lord': DQNFirst,
        'down': None,
        'up': None,
    }
    reward_dict = {
        'lord': 100,
        'down': None,
        'up': None,
    }
    train_dict = {
        'lord': True,
        'up': False,
        'down': False,
    }
    game = Game(EnvCooperation, net_dict, dqn_dict,
                reward_dict=reward_dict, train_dict=train_dict,
                debug=True)
    game.train(20, 5, 10)
