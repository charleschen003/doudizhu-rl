# https://www.jianshu.com/p/9fb001daedcf
from mcts.card import action_space_category

char2val = {
    "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14,
    "2": 15, "*": 16, "$": 17
}
char2index = {
    "3": 1, "4": 2, "5": 3, "6": 4,
    "7": 5, "8": 6, "9": 7, "10": 8,
    "J": 9, "Q": 10, "K": 11, "A": 12,
    "2": 13, "*": 14, "$": 15
}

cards_value = {}
for c in range(len(action_space_category)):
    for a in action_space_category[c]:
        cards = [0] * 15
        for i in a:
            cards[char2index[i]-1] += 1
        v = None
        if c == 0:
            v = 0
        elif c <= 3:  # 1单牌, 2对子, 3三条
            v = char2val[a[0]] - 10  # maxCard - 10
            if c == 2 and v > 0:
                v *= 1.5  # positive + 50%
            if c == 3 and v > 0:
                v *= 2  # positive + 100%
        elif c == 4:  # 4炸弹
            v = 9  # 固定9分
        elif c <= 6:  # 5三带一, 6三带二
            v = char2val[a[0]] - 10  # maxCard - 10
            if v > 0:
                v *= 1.5  # 带牌比三条加得少
        elif c <= 9:  # 7顺子, 8连对, 9飞机
            v = max(0, (char2val[a[-1]] - 10) / 2)  # max(0, (maxCard - 10) / 2)
        elif c == 10:  # 10飞机带小
            main_len = len(a) // 4 * 3
            v = max(0, (char2val[a[-1]] - 10) / 2)  # max(0, (maxCard - 10) / 2)
            for i in range(main_len, len(a)):
                if char2val[a[i]] > 10:
                    v += char2val[a[i]] - 10  # 带牌为正加上
        elif c == 11:  # 11飞机带大
            main_len = len(a) // 5 * 3
            v = max(0, (char2val[a[-1]] - 10) / 2)  # max(0, (maxCard - 10) / 2)
            for i in range(main_len, main_len + main_len // 3):
                if char2val[a[i]] > 10:
                    v += 1.5 * (char2val[a[i]] - 10)  # 带牌为正加上
        elif c == 12:  # 12火箭
            v = 12
        elif c <= 14:  # 13四带二只, 14四带二对
            v = char2val[a[0]] - 10  # maxCard - 10
        assert v is not None
        cards_value[tuple(cards)] = v

# print(action_space_category)
# print(cards_value)