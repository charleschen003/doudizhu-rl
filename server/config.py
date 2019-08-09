from dqn import DQNFirst
from net import NetCooperationSimplify

# db_url = 'sqlite:///tmp.db'
db_url = 'mysql://lyq:lyqhhh@localhost/ddz'

net_dict = {
    'lord': NetCooperationSimplify,
    'up': NetCooperationSimplify,
    'down': NetCooperationSimplify,
}

model_dict = {
    'lord': '0808_0852_lord_3500_59',  # 原：0805_1409_lord_4000
    'up': '0808_0854_up_4000',
    'down': '0808_0854_down_4000',
}

dqn_dict = {
    'lord': DQNFirst,
    'up': DQNFirst,
    'down': DQNFirst,
}
