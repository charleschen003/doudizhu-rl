# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:53:29 2019

@author: 刘文景
"""
from envi import r
import numpy as np
import time
import random


def hash_card(card):  # 例：输入[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 输出：'020000000000000' 被hash_actions调用
    # print("hash_card输入:",card)
    card_str = ""
    for i in card:
        card_str += str(i)
    return card_str


def hash_actions(actions):  # 例：输入[[1,2],[3,4],[5,6]] 输出：'12 34 56 ' 调用hash_card函数
    # print("hash_actions输入:",actions)
    actions_str = ""
    for i in actions:
        actions_str += hash_card(i) + " "
    # print("hash_actions输出:",actions_str)
    return actions_str


def after_move_cards(cards, player, action):  # 输入当前所有人手牌状态 执行出牌动作动作的玩家 出的牌 返回出牌后的手牌状态
    """
        例:
        输入：
        cards=[[1,2,3],[4,5,6],[7,8,9]]
        after_move_cards(cards,0,[1,0,0])
        输出：
        [[0, 2, 3], [4, 5, 6], [7, 8, 9]]
    """
    c = cards[:]  # !!!!!注意这里是深复制
    # print("after_move_cards:输入：cards：",c,"player：",player,"action:",action)
    t = np.array(c[player])
    t -= np.array(action)
    c[player] = list(t)
    # print("after_move_cards:输出：",c)
    return c


def get_moves_new(hand, last_move):  # 对于俊get_moves函数的一点小修正
    # print("get_moves_new： hand:",hand,"last_move:",last_move)
    if hand == [0] * 15:  # 即如果手牌为空 那么应对任何牌返回的动作集皆是空集
        # print("结果：空")
        return []
    else:
        # print("结果：",get_moves(hand,last_move))
        return r.get_moves(hand, last_move)


class GameStateBase:  # 游戏状态父类

    def __init__(self, parent, to_move, actions):
        self.parent = parent  # parent:记录父结点
        self.to_move = to_move  # to_move:该回合要行动的玩家
        self.actions = actions  # actions：该回合可能的行动

    def is_chance(self):
        return self.to_move == "CHANCE"  # 若该回合的to_move为特殊标记“CHANCE”，则说明该结点是机会结点

    def visualization(self):  # 用来观看所生成的所有局面
        if self.is_terminal:
            print("胜利玩家：", (self.to_move - 1) % 3, "初始发牌状态:", self.initial_cards)
            print("出牌过程:", self.information_set[1:], "\n")
        else:
            for child in self.children:
                self.children[child].visualization()


class ChanceGameState(GameStateBase):  # 机会结点（初始发牌） 继承GameStateBase
    def __init__(self, actions, first_to_move, last_move=[0] * 15):
        """
            actions传入的是发牌之后所有有可能的手牌状态    
                格式[ [第一种可能的每人手牌状态] , [第二种可能的每人手牌状态] , …… , [最后一种可能的每人手牌状态]]
                其中每一种可能的每人手牌状态的格式是： [ [位置1的初始手牌] , [位置2的初始手牌] , [位置3的初始手牌] ] 
                其中[位置1的初始手牌]的格式是一个 15维list
            first_to_move传入的是chance node之后第一个行动的玩家是谁
            last_move为chance node之前上一个有效出牌 默认为没有（即下一个玩家first_to_move自由出牌）
            
        """
        # print("chancenode actions：",actions)
        super().__init__(parent=None, to_move="CHANCE", actions=actions)  # to_move 为特殊名“CHANCE” actions为所有发牌的可能

        ###################   构造孩子结点部分   ###################################### 
        self.children = {  # 孩子结点是一个字典，key是可能的每人手牌情况，value是对应的孩子结点（PlayerMoveGameState类的实例）
            hash_actions(cards): PlayerMoveGameState(
                # 参数分别是 parent , to_move, actions_history , initial_cards , cards , actions , last_action
                self, first_to_move, [], cards, cards, get_moves_new(cards[first_to_move], last_move), last_move,
                "CHANCE"
            ) for cards in self.actions
        }
        ######################################################################################################

        self.information_set = actions  # 初始信息集（CHANCE NODE的信息集并没有实际意义）

        self.is_terminal = False

        self.chance_prob = 1. / len(self.children)  # 设置发牌产生的每种结果的可能性 

    def sample_one(self):
        return random.choice(list(self.children.values()))


class PlayerMoveGameState(GameStateBase):  # 玩家行动结点 继承GameStateBase

    def __init__(self, parent, to_move, actions_history, initial_cards, cards, actions, last_valid_action,
                 last_valid_action_pid):
        """
            parent 为该结点的父母结点 即本状态的上一个状态
            to_move 为本回合要行动的角色
            actions_history 为出牌历史
            initial_cards 为初始发牌状态
            cards 为当前回合所有人的手牌状态
            actions 为当前回合出牌角色可以做出的所有行动（可以出的所有牌组）
            last_valid_action 为上一次有效出牌（即忽视“要不起”）
            last_valid_action_pid 为上一次有效出牌的角色
        """

        super().__init__(parent=parent, to_move=to_move, actions=actions)

        self.actions_history = actions_history
        self.initial_cards = initial_cards
        self.cards = cards
        self.last_valid_action_pid = last_valid_action_pid
        self.last_valid_action = last_valid_action

        ###################   判断是否该结点是否是终结状态 并且计算效用值   ###############################
        actor = 0
        self.is_terminal = False
        for c in self.cards:
            if np.array(c).sum() == 0:  # 有某个玩家（actor）已经出完了牌
                self.is_terminal = True  # 该结点是终结结点
                if actor == 0:  # 地主胜利
                    self.utility = 1
                else:  # 农民胜利
                    self.utility = -1
            actor += 1

        ##############################################################################################################

        def next_react_actions(self, action):  # 当该结点选择动作action时，下一个结点所有的可能动作
            if action != [0] * 15:  # 若本回合没有选择过牌 则下一家应该应对本回合所出的牌
                return get_moves_new(self.cards[(self.to_move + 1) % 3], action)
            else:  # 若本回合选择过牌
                if (self.to_move + 1) % 3 == self.last_valid_action_pid:  # 假如下一家是上一个产生有效发牌的人 则下一家是自由出牌
                    return get_moves_new(self.cards[(self.to_move + 1) % 3], [0] * 15)
                else:  # 假如下一家不是上一个产生有效发牌的人 则下一家需要应对上一个有效发牌
                    return get_moves_new(self.cards[(self.to_move + 1) % 3], self.last_valid_action)

        def next_react_last_valid_action(self, action):  # 当该结点选择动作action时，下一个结点所记录的上一个有效动作
            if action != [0] * 15:  # 若本回合没有选择过牌 则下一家应该应对本回合的牌
                return action
            else:  # 若本回合选择过牌
                if (self.to_move + 1) % 3 == self.last_valid_action_pid:  # 假如下一家是上一个产生有效发牌的人 则下一家是自由出牌
                    return [0] * 15
                else:  # 假如下一家不是上一个产生有效发牌的人 则下一家需要应对上一个有效发牌
                    return self.last_valid_action

        def next_react_last_valid_action_pid(self, action):  # 当该结点选择动作action时，下一个结点所记录的上一个有效动作的执行者
            if action != [0] * 15:  # 若本回合没有选择过牌 则下一家应该应对本回合的牌
                return self.to_move
            else:  # 若本回合选择过牌
                return self.last_valid_action_pid

        ###################   构造孩子结点部分   ######################################

        if self.is_terminal == False:  # 不是终端结点 才可构造接下来的孩子结点
            self.children = {
                hash_card(a): PlayerMoveGameState(
                    self,  # 参数parent
                    (self.to_move + 1) % 3,  # 参数to_move 下一个要行动的玩家
                    self.actions_history + [a],  # 参数actions_history 原来的历史+本次的行动
                    initial_cards,  # 参数initial_cards  代表的是最开始发牌（指的是chance node）的时候 的每个人的手牌状态
                    after_move_cards(self.cards, to_move, a),  # 参数cards  代表的是该行动结束之后 下一回合的手牌状态
                    next_react_actions(self, a),  # 参数actions 下一个玩家在行动a之后可能的行动
                    next_react_last_valid_action(self, a),  # 参数 last_valid_action
                    next_react_last_valid_action_pid(self, a),  # 参数last_valid_action_pid
                ) for a in self.actions
            }
        else:
            self.children = {}
        ########################################################################### 

        ###################   构造信息集部分   ###################################### 
        # 构造信息集（针对的是当前玩家to_move） 该结点（状态）所处在的信息集
        # 构造信息集的第一项 ini_card 指的是本回合行动的玩家最初的手牌
        if self.to_move == 0:  # 当前回合是玩家0行动（所以构造的是针对玩家0的信息集）
            ini_card = self.initial_cards[0]  # 初始发牌时第0位玩家的手牌
        elif self.to_move == 1:
            ini_card = self.initial_cards[1]
        else:
            ini_card = self.initial_cards[2]
        # 信息集格式：[最初时该玩家的手牌ini_card, 行动历史1 即actions_history的第一项， 行动历史2 即actions_history的第二项，……]
        self.information_set = [ini_card]
        for history in self.actions_history:
            self.information_set += [history]

        # print("信息集：",self.information_set)
        ###########################################################################     


def init_sigma(node, output=None):  # 初始化策略：输入一个结点（一般是根节点） 然后输出从该结点开始直到最深 所有信息集的初始策略（随机策略）
    output = dict()  # 创建空字典 字典内的元素还是字典

    def init_sigma_recursive(node):
        output[hash_actions(node.information_set)] = {hash_card(action): 1. / len(node.actions) for action in
                                                      node.actions}  # 构造该结点的针对当前信息集的策略
        for k in node.children:
            init_sigma_recursive(node.children[k])  # 按深度遍历 构造每个结点的信息集的策略

    if not node.is_chance():  # 如果该结点不是chance node 正常遍历
        init_sigma_recursive(node)
    else:  # 如果该结点是chance node 则只遍历该结点的孩子结点（策略只针对非chande node的node）
        for action in node.actions:
            output.update(init_sigma(node.children[hash_actions(action)]))
    return output  # 格式是字典 key是信息集 value是该信息集下的策略（策略也是一个字典 key是某种行动 value是该行动的概率）


def init_empty_node_maps(node, output=None):  # 初始化结点并指向0值 输出output格式：字典 key是信息集 value也是一个字典（key是动作 value是0 待更新）
    output = dict()

    def init_empty_node_maps_recursive(node):
        output[hash_actions(node.information_set)] = {hash_card(action): 0. for action in node.actions}
        for k in node.children:
            init_empty_node_maps_recursive(node.children[k])

    if not node.is_chance():  # 如果该结点不是chance node 正常遍历
        init_empty_node_maps_recursive(node)
    else:  # 如果该结点是chance node 则只遍历该结点的孩子结点
        for action in node.actions:
            output.update(init_empty_node_maps(node.children[hash_actions(action)]))
    return output


class CounterfactualRegretMinimizationBase:

    def __init__(self, root, chance_sampling=False):
        self.root = root
        self.sigma = init_sigma(root)  # 格式：字典 key是信息集 value也是一个字典（key是动作 value是对应动作的选择概率（初始为随机选择））
        self.cumulative_regrets = init_empty_node_maps(
            root)  # 一开始都是0 格式：字典 key是信息集（hash后的） value也是一个字典（key是动作（hash后的） value是0 待更新）
        # self.cumulative_sigma = init_empty_node_maps(root)    # 一开始都是0 格式同上
        # self.nash_equilibrium = init_empty_node_maps(root)    # 一开始都是0 格式同上   # 在 __value_of_the_game_state_recursive会用到
        self.chance_sampling = chance_sampling

    def _update_sigma(self, information_set):  # 利用cfr算法更新策略：information_set是信息集（不是Chance Node）
        # print("调用_update_sigma")
        i = hash_actions(information_set)  # 信息集hash化
        # print("_update_sigma中cumulative_regrets[i].values：",self.cumulative_regrets[i].values())
        rgrt_sum = sum(
            filter(lambda x: x > 0, self.cumulative_regrets[i].values()))  # 返回self.cumulative_regrets[i].values()中正数之和
        for a in self.cumulative_regrets[i]:  # a为对应该信息集下的某种动作
            before_change = self.sigma[i][a]
            # print("原来的某策略：",self.sigma[i][a])
            self.sigma[i][a] = max(self.cumulative_regrets[i][a], 0.) / rgrt_sum if rgrt_sum > 0 else 1. / len(
                self.cumulative_regrets[i].keys())
            after_change = self.sigma[i][a]
            if abs(after_change - before_change) > 1e-3:
                print("策略修正 ", after_change - before_change)
            # print("_update_sigma后的某策略：",self.sigma[i][a])

    def _cumulate_cfr_regret(self, information_set, action, regret):
        # print("调用_cumulate_cfr_regret，regret：",regret)
        i = hash_actions(information_set)
        act = hash_card(action)
        self.cumulative_regrets[i][act] += regret

    def _cfr_utility_recursive(self, state, reach_a, reach_b, reach_c):  # 迭代调用返回该结点的虚拟效用(counterfactual utility)
        # reach_i就相当于第i个玩家 到当前结点为止做出过的所有决定的概率的积
        children_states_utilities = {}

        if state.is_terminal:  # 如果当前结点是终结结点 直接返回效用值
            return state.utility

        # 如果当前节点是发牌结点（chance），要考虑是否sampling 若采样某一种发牌情况，则计算该情况cfr效用；若不采样，则计算所有发牌情况的平均效用
        if state.to_move == "CHANCE":
            if self.chance_sampling:
                # if node is a chance node, lets sample one child node and proceed normally
                return self._cfr_utility_recursive(state.sample_one(), reach_a, reach_b, reach_c)  # samole_one 函数暂时缺失
            else:
                chance_outcomes = {state.children[hash_actions(action)] for action in
                                   state.actions}  # 格式：集合 该state在不同动作的情况下产生的所有子结点的集合
                return state.chance_prob * sum(
                    [self._cfr_utility_recursive(outcome, reach_a, reach_b, reach_c) for outcome in chance_outcomes])

        # 如果是游戏中间状态结点 计算该结点cfr效用 （sum up all utilities for playing actions in our game state） 
        value = 0.
        for action in state.actions:
            sigma_info = self.sigma[hash_actions(state.information_set)]
            act = hash_card(action)
            child_reach_a = reach_a * (
                sigma_info[act] if state.to_move == 0 else 1)  # 如果该回合是第一个玩家做决定 那么在a玩家的做决定策略上乘上做该次决定的概率
            child_reach_b = reach_b * (
                sigma_info[act] if state.to_move == 1 else 1)  # 如果该回合是第二个玩家做决定 那么在b玩家的做决定策略上乘上做该次决定的概率
            child_reach_c = reach_c * (
                sigma_info[act] if state.to_move == 2 else 1)  # 如果该回合是第三个玩家做决定 那么在b玩家的做决定策略上乘上做该次决定的概率

            # 将被选择的孩子结点视作根节点 计算该孩子结点的效用（value as if child state implied by chosen action was a game tree root）
            child_state_utility = self._cfr_utility_recursive(state.children[act], child_reach_a, child_reach_b,
                                                              child_reach_c)
            # 将上函数得到的结果乘以选择该行动的概率（即对针对该行动的策略）增加到该结点的value值上
            value += sigma_info[act] * child_state_utility
            # values for chosen actions (child nodes) are kept here
            children_states_utilities[act] = child_state_utility

        # cfr_reach是相对于该结点的玩家来说 其余的玩家做出的选择的概率乘积
        # reach是相对于该结点的玩家做出的选择的概率乘积
        if state.to_move == 0:
            (cfr_reach, reach) = (reach_b * reach_c, reach_a)
        elif state.to_move == 1:
            (cfr_reach, reach) = (reach_a * reach_c, reach_b)
        else:  # 该节点行动的是第三个玩家
            (cfr_reach, reach) = (reach_a * reach_b, reach_c)

        for action in state.actions:
            """
            # 对不同的玩家身份，得到的效用是不同的（地主赢就代表了农民输） 所以应该做一些正负的转换
            # 但是对此存疑 因为该模式取于二人零和博弈模型 所以效用的设定 存在了疑问
            """

            act = hash_card(action)
            if state.to_move == 0:  # 当前行动的是地主
                action_cfr_regret = cfr_reach * (children_states_utilities[act] - value)
            else:  # 当前行动的是农民  (效用和地主的效用是相反的)
                action_cfr_regret = -1 * cfr_reach * (children_states_utilities[act] - value)

            # 计算self.cumulative_regrets[state.inf_set()][action] += action_cfr_regret
            self._cumulate_cfr_regret(state.information_set, action, action_cfr_regret)

        if self.chance_sampling:
            # update sigma according to cumulative regrets - we can do it here because we are using chance sampling
            # and so we only visit single game_state from an information set (chance is sampled once)
            self._update_sigma(state.information_set)
        return value


class VanillaCFR(CounterfactualRegretMinimizationBase):

    def __init__(self, root):
        super().__init__(root=root, chance_sampling=False)

    def run(self, iterations=1):
        for _ in range(0, iterations):
            print("第", _ + 1, "轮开始", end=" ")
            time_start = time.time()

            self._cfr_utility_recursive(self.root, 1, 1, 1)
            # since we do not update sigmas in each information set while traversing, we need to traverse the tree to perform to update it now
            self.__update_sigma_recursively(self.root)

            time_end = time.time()
            print('本轮结束 总用时：', time_end - time_start)

    def __update_sigma_recursively(self, node):
        # stop traversal at terminal node
        if node.is_terminal:
            return
        # 忽略chance node 
        if not node.is_chance():  # 如果该结点不是CHANCE node
            self._update_sigma(node.information_set)
        # go to subtrees
        for k in node.children:
            self.__update_sigma_recursively(node.children[k])


class ChanceSamplingCFR(CounterfactualRegretMinimizationBase):

    def __init__(self, root):
        super().__init__(root=root, chance_sampling=True)

    def run(self, iterations=1):
        for _ in range(0, iterations):
            print("第", _ + 1, "轮开始")
            time_start = time.time()
            self._cfr_utility_recursive(self.root, 1, 1, 1)
            time_end = time.time()
            print('本轮结束 总用时：', time_end - time_start)


# 余冠一
def deal(card_n, remainder, cards=[[0] * 15] * 3):  # 为了生成所有发牌情况
    """
        card_ni 代表第i位玩家的手牌数
        remainder 代表当前能发出的所有牌（相对于残局 即意味着 全部的牌减去已经打出的牌）
    """
    if sum(remainder) != sum(card_n):  # 如果剩余可发的牌和所有人的手牌数不等 则报错:
        raise RuntimeError('deal手牌设置出错')

    if sum(remainder) == 0:
        return [cards]

    output = []
    for i in range(0, 15):  # 代表remainder向量的第i位
        if remainder[i] > 0:
            r = remainder[:]
            r[i] = 0
            for num1 in range(min(remainder[i], card_n[0]) + 1):
                for num2 in range(min(remainder[i] - num1, card_n[1]) + 1):
                    num3 = remainder[i] - num1 - num2
                    if num3 <= card_n[2]:
                        n = card_n[:]
                        n[0] -= num1
                        n[1] -= num2
                        n[2] -= num3
                        c = [dc[:] for dc in cards]
                        c[0][i] += num1
                        c[1][i] += num2
                        c[2][i] += num3
                        output.extend(deal(n, r, c))
            break
    return output


def initiate_game(person, card, first_to_move, last_move=[0] * 15):
    cards_dealings = deal(person, card)
    testgame = ChanceGameState(cards_dealings, first_to_move, last_move)
    hahaha = VanillaCFR(testgame)
    hahaha.run(10)
    return hahaha.sigma


def choose(information_set, sigma):
    """
        information_set 格式形如 '010000000000000 100000000000000 '
    """
    probability = np.array(list(sigma[information_set].values()))
    return np.random.choice(list(sigma[information_set].keys()), p=probability.ravel())


def card_change(yq_card):
    wj_card = [0] * 15
    for i in yq_card:
        wj_card[i - 3] += 1
    return wj_card


"""测试用例            
payload1 = {
    'role_id': 1,  # 0代表地主上家，1代表地主，2代表地主下家
    'last_taken': {  # 更改处
        0: [],
        1: [],
        2: [],
    },
    'cur_cards': [3,3],  # 无需保持顺序
    'history': {  # 各家走过的牌的历史The environment
        0: [4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11],
        1: [12,12,12,12,13,13,13,13],
        2: [14,14,14,14,15,15,15,15,16,17],
    },
    'left': {  # 各家剩余的牌
        0: 2,
        1: 2,
        2: 2,
    },
    'debug': False,  # 是否返回debug
}
"""


def final_card(payload):
    first_to_move = (payload['role_id'] - 1) % 3
    id = payload['role_id']
    if payload['last_taken'][(id - 1) % 3] == []:  # 上家为空
        if payload['last_taken'][(id - 2) % 3] == []:  # 上上家为空
            last_move = [0] * 15
        else:
            last_move = card_change(payload['last_taken'][(id - 2) % 3])
    else:
        last_move = card_change(payload['last_taken'][(id - 1) % 3])
    person = [payload['left'][1], payload['left'][2], payload['left'][0]]
    card = list(np.array([4] * 13 + [1, 1]) - np.array(
        card_change(payload['history'][0] + payload['history'][1] + payload['history'][2])))
    sigma = initiate_game(person, card, first_to_move, last_move)
    information_set = hash_card(card_change(payload['cur_cards'])) + " "
    return choose(information_set, sigma)

# final_card(payload1)
