import numpy as np
from game.engine import Card
from mcts.get_moves import get_moves
from copy import copy

class Node(object):
    def __init__(self, parent, state):
        self.parent = parent
        self.children = []
        # 胜利
        self.reward = 0
        # 总局数
        self.visit = 0
        self.state = state

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_children(self):
        return self.children

    def add_child(self, sub_node):
        self.children.append(sub_node)

    def is_all_expand(self):
        if len(self.children) < self.state.moves_num:
            return False
        return True

    def expand(self):
        if self.state.try_flag == 0:
            valid_moves = get_moves(self.state.my_card, self.state.last_move)
            for move in valid_moves:
                self.state.init_untried_actions(move)
            self.state.try_flag = 1

        moves_num = len(self.state.untried_actions)
        i = np.random.choice(moves_num)
        untried_move = self.state.untried_actions[i].copy()
        while self.state.is_buchu(untried_move) and self.state.last_pid == self.state.my_id:
            i = np.random.choice(moves_num)
            untried_move = self.state.untried_actions[i].copy()

        new_state = self.get_state().get_next_state_with_random_choice(untried_move)
        del self.state.untried_actions[i]
        sub_node = Node(self, new_state)
        self.add_child(sub_node)
        return sub_node


class State(object):
    def __init__(self, my_id, my_card, next_card, next_next_card, last_move, winner, moves_num, action, last_p):
        self.my_id = my_id
        self.my_card = my_card
        self.next_card = next_card
        self.next_next_card = next_next_card
        self.last_move = last_move
        self.winner = winner
        self.moves_num = moves_num
        self.action = action
        self.last_pid = last_p
        self.untried_actions = []
        self.try_flag = 0

    def init_untried_actions(self, move):
        self.untried_actions.append(move)

    def compute_reward(self, my_id):
        if my_id == 0:
            if self.winner == my_id:
                return 1
            else:
                return 0
        else:
            if self.winner != 0:
                return 1
            else:
                return 0

    def get_next_state_with_random_choice(self, untried_move):

        #  下家变自家，下下家变下家，自家变下下家
        valid_moves = get_moves(self.my_card, self.last_move)
        moves_num = len(valid_moves)
        i = np.random.choice(moves_num)
        tmp = valid_moves[i].copy()
        if untried_move is not None:
            tmp = untried_move
        while self.is_buchu(tmp) and self.last_pid == self.my_id:
            i = np.random.choice(moves_num)
            tmp = valid_moves[i].copy()
        move = []
        next_next_card = self.my_card.copy()
        for k in Card.all_card_name:
            move.extend([int(k)] * tmp.get(k, 0))
            next_next_card[k] -= tmp.get(k, 0)

        my_id = (self.my_id + 1) % 3
        my_card = self.next_card.copy()
        next_card = self.next_next_card.copy()
        #  判断出完牌游戏是否结束
        winner = self.my_id
        for lis in next_next_card.values():
            if lis != 0:
                winner = -1
                break
        last_move = move.copy()
        last_p = self.my_id
        #  如果选择不出， 下家的last_move等于自家的last_move
        if len(move) == 0:
            last_p = self.last_pid
            last_move = self.last_move.copy()
        if len(move) == 0 and self.last_pid == my_id:
            last_move = []
        valid_moves_ = get_moves(my_card, last_move)
        moves_num_ = len(valid_moves_)
        next_state = State(my_id, my_card, next_card, next_next_card, last_move, winner, moves_num_, move, last_p)
        return next_state

    @staticmethod
    def is_buchu(move):
        for k in Card.all_card_name:
            if move.get(k) != 0:
                return False
        return True
