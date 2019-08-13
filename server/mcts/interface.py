from __future__ import absolute_import
import sys
import os
sys.path.insert(0, os.path.join('..'))

from mcts.tree_policy import tree_policy
from mcts.default_policy import default_policy
from mcts.backup import backup
from mcts.tree import Node, State
from mcts.get_moves import get_moves
from mcts.get_bestchild import get_bestchild_
import time


def mcts(payload):
    root = Node(None, None)
    my_id = (payload['role_id'] + 2) % 3

    next_id = (payload['role_id'] + 1) % 3
    next_next_id = (payload['role_id'] + 2) % 3
    my_card_ = payload['hand_card'][payload['role_id']]
    my_card_.sort()
    my_card = card_list_to_dict(card_to_list(change_card_form_reversal(my_card_)))
    next_card_ = payload['hand_card'][next_id]
    next_card_.sort()
    next_card = card_list_to_dict(card_to_list(change_card_form_reversal(next_card_)))
    next_next_card_ = payload['hand_card'][next_next_id]
    next_next_card_.sort()
    next_next_card = card_list_to_dict(card_to_list(change_card_form_reversal(next_next_card_)))
    last_move_, last_p_ = get_last_move(payload['role_id'], next_id, next_next_id, payload['last_taken'])
    last_move = change_card_form_reversal(last_move_)
    last_p = (last_p_ + 2) % 3
    moves_num = len(get_moves(my_card, last_move))
    state = State(my_id, my_card, next_card, next_next_card, last_move, -1, moves_num, None, last_p)
    root.set_state(state)

    computation_budget = 1000
    for i in range(computation_budget):
        expand_node = tree_policy(root, my_id)
        reward = default_policy(expand_node, my_id)
        backup(expand_node, reward)
    best_next_node = get_bestchild_(root)
    move = best_next_node.get_state().action

    return move


def change_card_form_reversal(before):
    #  e.g.[3, 3, 3, 4, 4, 4, 14, 15, 16, 17] -> [3, 3, 3, 4, 4, 4, 1, 2, 14 ,15]
    card = before.copy()
    for i, j in enumerate(before):
        if j == 14:
            card[i] = 1
        if j == 15:
            card[i] = 2
        if j == 16:
            card[i] = 14
        if j == 17:
            card[i] = 15
    return card


def card_to_list(before):
    #  e.g. [3, 3, 3, 4, 4, 4, 1, 2] -> [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    # index = [str(i) for i in range(3, 14)] + ['1', '2', '14', '15']
    tem = [0] * 15
    for card in before:
        tem[card - 1] += 1
    tem = tem[2:-2] + tem[:2] + tem[-2:]
    return tem


def card_list_to_dict(card_list):
    #  e.g. [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0] -> ['3':3, '4':3, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, '12':0, '13':0, '1':1, '2':1, '14':0, '15':0]
    card_name = [str(i) for i in range(3, 14)] + ['1', '2', '14', '15']
    card_dict = dict(zip(card_name, card_list))
    return card_dict


def get_last_move(role_id, next_id, next_next_id, last_taken):
    my_taken = last_taken[role_id]
    next_taken = last_taken[next_id]
    next_next_taken = last_taken[next_next_id]
    if len(next_next_taken) != 0:
        return next_next_taken, next_next_id
    if len(next_taken) != 0:
        return next_taken, next_id
    return my_taken, role_id


pl = {
    'role_id': 0,  # 0代表地主上家，1代表地主，2代表地主下家
    'last_taken': {  # 更改处
        0: [],
        1: [7, 8, 9, 10, 11],
        2: [],
    },
    'cur_cards': [4, 4, 5, 5, 5, 6, 7, 9, 10, 11, 11, 12, 13, 14, 15, 15, 17],  # 无需保持顺序
    'history': {  # 各家走过的牌的历史The environment
        0: [],
        1: [7, 8, 9, 10, 11],
        2: [],
    },
    'left': {  # 各家剩余的牌
        0: 17,
        1: 15,
        2: 17,
    },
    'debug': False,  # 是否返回debug
    'hand_card': {
        0: [4, 4, 5, 5, 5, 6, 7, 9, 10, 11, 11, 12, 13, 14, 15, 15, 17],
        1: [3, 3, 4, 6, 6, 9, 12, 12, 13, 13, 14, 14, 15, 15, 16],
        2: [3, 3, 4, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 13, 14],
    }
}