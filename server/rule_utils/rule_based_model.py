import numpy as np

from server.rule_utils.card import action_space
from server.rule_utils.decomposer import Decomposer
from server.rule_utils.evaluator import cards_value, dapai

card_list = [
    "3", "4", "5", "6",
    "7", "8", "9", "10",
    "J", "Q", "K", "A",
    "2", "*", "$"
]


# 返回当前局面上剩余最大的单牌（没考虑炸弹）
def maxcard(other_hand, l, ll):
    max = 0
    # for i, j in enumerate(other_hand):
    #     if j != 0:
    #         max = i
    for i in other_hand[l]:
        if i > max:
            max = i
    for j in other_hand[ll]:
        if j > max:
            max = j
    return max - 2


def choose(payload3):
    # 获得手牌
    hand_card = payload3['cur_cards']

    # 拆牌器和引擎用了不同的编码 1 -> A, B -> *, R -> $
    trans_hand_card = [card_list[i - 3] for i in hand_card]
    # 获得上家出牌
    role_id = payload3['role_id']
    lrole_id = (role_id - 1 + 3) % 3  # 上家ID
    llrole_id = (role_id - 2 + 3) % 3  # 上上家ID
    last_pid = lrole_id  # 上一个有效出牌
    tlast_move = payload3['last_taken'][last_pid]
    if not tlast_move:
        last_pid = llrole_id
        tlast_move = payload3['last_taken'][last_pid]
    last_move = [card_list[i - 3] for i in tlast_move]
    # last_move = [card_list[i] for i in range(15) for _ in range(state.last_move[i])]
    # 拆牌
    D = Decomposer()
    combs, fine_mask = D.get_combinations(trans_hand_card, last_move)
    # 根据对手剩余最少牌数决定每多一手牌的惩罚
    # left_crads = [sum(p.get_hand_card()) for p in self.game.players]
    # min_oppo_crads = min(left_crads[1], left_crads[2]) if self.player_id == 0 else left_crads[0]
    min_oppo_crads = min(payload3['left'][lrole_id], payload3['left'][llrole_id])
    round_penalty = 17 - 12 * min_oppo_crads / 20  # 惩罚值调整为与敌人最少手牌数负线性相关

    if not last_move:
        if role_id == 0:  # 地主
            round_penalty += 7
        elif role_id == 1:  # 地主下家
            round_penalty += 5
        else:  # 地主上家
            round_penalty += 3

    if role_id == 2 and not last_move:  # 队友没要地主牌
        round_penalty += 5
    if role_id == 1 and not last_move:  # 地主没要队友牌
        round_penalty -= 8

    # 寻找最优出牌
    best_move = None
    max_value = -np.inf
    for i in range(len(combs)):
        # 手牌总分
        total_value = sum([cards_value[x] for x in combs[i]])
        # small_num = 0
        # for j in range(0, len(combs[i])):
        #     if j > 0 and action_space[j][0] not in ["2", "R", "B"]:
        #         small_num += 1
        # total_value -= small_num * round_penalty
        small_num = hand_card[-1] + hand_card[-2] + hand_card[-3]
        small_num = (len(combs[i]) - small_num)  # 如果一手牌为小牌, 需要加上惩罚值, 所以要统计小牌数量
        total_value -= small_num * round_penalty

        # 手里有火箭和另一手牌
        if len(combs[i]) == 3 and combs[i][0] == 0 or len(combs[i]) == 2:
            if cards_value[combs[i][-1]] == 12 or cards_value[combs[i][-2]] == 12:
                print('*****rule  火箭直接走')
                return [0] * 13 + [1, 1], None

        # 下家农民手里只有一张牌,送队友走
        # if role_id == 1 and sum(self.game.players[2].get_hand_card()) == 1 and not last_move:
        if role_id == 1 and payload3['left'][2] == 1 and not last_move:
            for i, j in enumerate(hand_card):
                if j != 0:
                    tem = [0] * 15
                    tem[i] = 1
                    print('******rule  下家农民手里只有一张牌,送队友走')
                    return tem, None

        # 队友出大牌能走就压
        if role_id == 2 and len(combs[i]) == 3 and combs[i][0] == 0:
            if action_space[combs[i][1]] in dapai and (fine_mask is None or fine_mask[i, 1] == True):
                print('******rule  队友出大牌能走就压')
                best_move = combs[i][1]
                break
            elif action_space[combs[i][2]] in dapai and (fine_mask is None or fine_mask[i, 2] == True):
                print('******rule  队友出大牌能走就压')
                best_move = combs[i][2]
                break

        # 队友出大牌走不了就不压
        if role_id == 2 and last_pid == 1 and sorted(last_move) in dapai:
            print('******rule  队友出大牌走不了就不压')
            best_move = 0
            break

        for j in range(0, len(combs[i])):
            # Pass 得分
            if combs[i][j] == 0 and min_oppo_crads > 8:
                if total_value > max_value:
                    max_value = total_value
                    best_move = 0
                    # print('pass得分',max_value,end='   //   ')
            # 出牌得分
            elif combs[i][j] > 0 and (fine_mask is None or fine_mask[i, j] == True):  # 枚举非pass且fine_mask为True的出牌
                # 特判只有一手
                if len(combs[i]) == 1 or len(combs[i]) == 2 and combs[i][0] == 0:
                    max_value = np.inf
                    best_move = combs[i][-1]
                    break

                move_value = total_value - cards_value[combs[i][j]] + round_penalty

                # 手里有当前最大牌和另一手牌
                if len(combs[i]) == 3 and combs[i][0] == 0 or len(combs[i]) == 2:
                    if combs[i][j] > maxcard(payload3['hand_card'], lrole_id, llrole_id) and combs[i][j] <= 15:
                        move_value += 100

                # 地主只剩一张牌时别出单牌
                # if role_id != 0 and sum(self.game.players[0].get_hand_card()) == 1:
                if role_id != 0 and payload3['left'][0] == 1:
                    if combs[i][j] <= maxcard(payload3['hand_card'], lrole_id, llrole_id):
                        move_value -= 100

                # 农民只剩一张牌时别出单牌
                if role_id == 0 and (payload3['left'][1] == 1 or payload3['left'][2] == 1):
                    if combs[i][j] <= maxcard(payload3['hand_card'], lrole_id, llrole_id):
                        move_value -= 100

                if move_value > max_value:
                    max_value = move_value
                    best_move = combs[i][j]
        if best_move is None:
            best_move = 0

    # 最优出牌
    best_cards = action_space[best_move]
    move = [best_cards.count(x) for x in card_list]
    # print('出牌得分', max_value)
    # 输出选择的牌组
    # print("\nbest comb: ")
    # for m in best_comb:
    #     print(action_space[m], cards_value[m])
    # 输出 player i [手牌] // [出牌]
    # print("Player {}".format(role_id), ' ', Card.visual_card(hand_card), end=' // ')
    # print(Card.visual_card(move))
    return move


if __name__ == "__main__":
    payload = {
        'role_id': 1,  # 0代表地主上家，1代表地主，2代表地主下家
        'last_taken': {  # 更改处
            0: [],
            1: [9, 9, 9, 6],
            2: [],
        },
        'cur_cards': [17, 16, 15, 14, 14, 12, 10],  # 无需保持顺序
        'history': {  # 各家走过的牌的历史The environment
            0: [],
            1: [5, 5, 5, 4, 4, 3, 3, 3, 3, 9, 9, 9, 6],
            2: [11, 11, 11, 8, 8],
        },
        'left': {  # 各家剩余的牌
            0: 17,
            1: 7,
            2: 12,
        },
        'hand_card': {
            0: [15, 14, 13, 13, 12, 10, 10, 9, 8, 8, 7, 7, 7, 6, 6, 6, 4],
            1: [17, 16, 15, 14, 14, 12, 10],
            2: [15, 15, 14, 13, 13, 12, 12, 11, 10, 7, 5, 4],
        },
        'debug': False,  # 是否返回debug
    }
    import time

    start = time.time()
    print(choose(payload))
    end = time.time()
    print(end - start)
