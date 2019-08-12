from dqn import DQNFirst
from net import NetCooperationSimplify

# db_url = 'sqlite:///tmp.db'
db_url = 'mysql+pymysql://lyq:lyqhhh@localhost/ddz'

net_dict = {
    'lord': NetCooperationSimplify,
    'up': NetCooperationSimplify,
    'down': NetCooperationSimplify,
}

model_dict = {
    'lord': '/data/models/lord_3500_59.pt',  # 原：0805_1409_lord_4000
    'up': '/data/models/up_4000.pt',
    'down': '/data/models/down_4000.pt',
}

dqn_dict = {
    'lord': DQNFirst,
    'up': DQNFirst,
    'down': DQNFirst,
}
