import os
import time
import torch
import logging
from datetime import datetime
from pytz import timezone

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target q
EPSILON_HIGH = 0.5  # starting value of epsilon
EPSILON_LOW = 0.01  # final value of epsilon
REPLAY_SIZE = 20000  # experience replay buffer size
BATCH_SIZE = 256  # size of minibatch
DECAY = int((8000 * (2 / 3)) / 5)  # epsilon decay config  1000 for 8000
UPDATE_TARGET_EVERY = 20  # target-net参数更新频率

CARDS = range(3, 18)
STR = [str(i) for i in range(3, 11)] + ['J', 'Q', 'K', 'A', '2', '小', '大']
DICT = dict(zip(CARDS, STR))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WORK_DIR, _ = os.path.split(os.path.abspath(__file__))

MODEL_DIR = os.path.join(WORK_DIR, 'models')
IMG_DIR = os.path.join(WORK_DIR, 'outs', 'images')
LOG_DIR = os.path.join(WORK_DIR, 'outs', 'logs')
WIN_DIR = os.path.join(WORK_DIR, 'outs', 'win_rates')
ENV_DIR = os.path.join(WORK_DIR, 'precompiled')


def name_dir(name):
    return os.path.join(*name.split('_', 2))


def get_logger():
    now_utc = datetime.now(timezone('Asia/Shanghai'))
    begin = now_utc.strftime("%m%d_%H%M")
    path = os.path.join(LOG_DIR, name_dir(begin))
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    path = '{}.log'.format(path)

    logger = logging.getLogger('DDZ_RL')
    logger.setLevel(logging.INFO)
    log_format = '[%(asctime)s][%(name)s][%(levelname)s]:  %(message)s'
    logging.basicConfig(filename=path, filemode='w', format=log_format)
    return begin, logger
