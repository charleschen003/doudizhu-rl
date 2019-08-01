# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target q
EPSILON_HIGH = 0.5  # starting value of epsilon
EPSILON_LOW = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
DECAY = 200  # epsilon decay config
UPDATE_TARGET_EVERY = 20  # target-net参数更新频率


EPISODE = 3000  # Episode limitation

CARDS = range(3, 18)
STR = [str(i) for i in range(3, 11)] + ['J', 'Q', 'K', 'A', '2', '小', '大']
DICT = dict(zip(CARDS, STR))
