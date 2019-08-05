from game import Game

if __name__ == '__main__':
    from envi import EnvComplicated
    from net import NetMoreComplicated
    from dqn import DQNFirst
    net_dict = {
        'lord': None,
        'down': NetMoreComplicated,
        'up': NetMoreComplicated,
    }
    dqn_dict = {
        'lord': None,
        'down': DQNFirst,
        'up': DQNFirst,
    }
    reward_dict = {
        'lord': None,
        'down': 100,
        'up': 100,
    }
    train_dict = {
        'lord': False,
        'up': True,
        'down': True,
    }
    game = Game(EnvComplicated, net_dict, dqn_dict,
                reward_dict=reward_dict, train_dict=train_dict,
                debug=False)
    game.train(20, 5, 10)
