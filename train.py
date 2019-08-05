from game import Game

if __name__ == '__main__':
    from envi import EnvComplicated
    from net import NetMoreComplicated
    from dqn import DQNFirst
    net_dict = {
        'lord': NetMoreComplicated,
        'down': None,
        'up': None,
    }
    dqn_dict = {
        'lord': DQNFirst,
        'down': None,
        'up': None,
    }
    reward = {
        'lord': 100,
        'down': 50,
        'up': 50,
    }
    game = Game(EnvComplicated, net_dict, dqn_dict, reward, debug=False)
    game.train(10, 1, 10)
