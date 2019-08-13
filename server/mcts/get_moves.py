import mcts.r
from mcts.evaluator import cards_value
import numpy as np

        # test r
# all_cards = [4]*13 + [1,1]
# print(all_cards)
#
# handcards = [0,0,0,0,0,1,1,3,3,3,3,3,1,1,1]
# no_cards = [0]*15
# result = r.get_moves(handcards,no_cards)
# print(result)


# 根据当前手牌和上家的牌返回所有可能出的牌
# input：
#     handcards : dict e.g. 4455566 = {'3': 0, '4': 2, '5': 3, '6': 2, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0, '13': 0, '1': 0, '2': 0, '14': 0, '15': 0}
#     lastcards: list e.g. 33 = [3,3]
# ouput:
#     list of dict

sidaihuojian = []
t = [0]*13+[1,1]
for i in range(13):
    tt = t[:]
    tt[i] = 4
    sidaihuojian.append(tt)

sandaihuojian = []
for i in range(11):
    ttt = t[:]
    ttt[i] = 3
    ttt[i+1] = 3
    sandaihuojian.append(ttt)


#  print(sidaihuojian)
def get_moves(handcards, lastcards):
    if not lastcards:
        lastcards = []
    index = [str(i) for i in range(3,14)] + ['1','2','14','15']
    rhandcards = list(handcards.values())
    tem = dict(zip(index, [0]*15))
    for l in lastcards:
        # print('last cards',l)
        tem[str(l)] += 1
    rlastcards = list(tem.values())

    moves = []
    rmoves = mcts.r.get_moves(rhandcards, rlastcards)

    length = len(rmoves)
    if length > 10:
        # print('-----pruning-----')
        values = []
        handnum = sum(rhandcards)
        rrmoves = []
        for m in rmoves:
            if m in sidaihuojian or m in sandaihuojian:
                continue
            rrmoves.append(m)
            values.append(cards_value[tuple(m)]- 0.1 * (handnum - sum(m)))
        sorted_index = sorted(range(len(values)), key=lambda i: values[i])
        for k in range(int(length/3+1)):
            moves.append(dict(zip(index, rrmoves[sorted_index[k]])))
            moves.append(dict(zip(index, rrmoves[sorted_index[-k-1]])))

    else:
        for m in rmoves:
            moves.append(dict(zip(index, m)))

    return moves


# test function:
# index = [str(i) for i in range(3, 14)] + ['1', '2', '14', '15']
# aa = [0, 2, 3, 2] + [0]*11
# a = dict(zip(index, aa))
# # hand_card = {'3': 2, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '10': 1, '11': 1, '12': 2, '13': 0, '1': 0,  '2': 1, '14': 0, '15': 0}
# # hand_card = {'3': 1, '4': 1, '5': 1, '6': 3, '7': 3, '8': 3, '9': 0, '10': 0, '11': 2, '12': 0, '13': 0, '1': 0,  '2': 1, '14': 0, '15': 0}
# hand_card = {'3': 2, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '10': 1, '11': 1, '12': 2, '13': 0, '1': 0,  '2': 2, '14': 0, '15': 0}
# b = []
# moves = get_moves(hand_card, b)
# print(moves)

