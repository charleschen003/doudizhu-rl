# https://www.jianshu.com/p/9fb001daedcf
from server.rule_utils.card import action_space_category

char2val = {
    "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14,
    "2": 15, "*": 16, "$": 17
}

dapai = []
i = 9
for i in range(11,15):
    dapai.append(sorted(action_space_category[1][i]))
for i in range(8,13):
    dapai.append(sorted(action_space_category[2][i]))
    dapai.append(sorted(action_space_category[3][i]))
for i in range(112,182):
    dapai.append(sorted(action_space_category[5][i]))
for i in range(96,156):
    dapai.append(sorted(action_space_category[6][i]))
for a in action_space_category[7]:
    if len(a) >= 7:
        dapai.append(sorted(a))
for i in [4,8,9,10,11,13,14]:
    for a in action_space_category[i]:
        dapai.append(sorted(a))


cards_value = []
for c in range(len(action_space_category)):
    for a in action_space_category[c]:
        v = None
        if c == 0:
            v = 0
        elif c <= 3:  # 1单牌, 2对子, 3三条
            v = char2val[a[0]] - 10  # maxCard - 10
            if c == 2 and v > 0:
                if a == ['2','2']:
                    v *= 1.2
                elif a == ['A','A']:
                    v *= 1.3
                else:
                    v *= 1.4  # positive + 50%
            if c == 3 and v > 0:
                if a == ['2','2','2']:
                    v *= 1
                elif a == ['A','A','A']:
                    v *= 1.5
                else:
                    v *= 1.8  # positive + 100%
        elif c == 4:  # 4炸弹
            if a == ['2','2','2','2']:
                v = 7
            else:
                v = 9  # 固定9分
        elif c <= 6:  # 5三带一, 6三带二
            v = char2val[a[0]] - 10  # maxCard - 10
            if v > 0:
                if a[:3] == ['2','2','2']:
                    v *= 1
                elif a == ['A','A','A']:
                    v *= 1.3
                else:
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
            if v > 0:
                if a[:4] == ['2','2','2','2']:
                    v *= 1
                elif a[:4] == ['A','A','A','A']:
                    v *= 1.2
                else:
                    v *= 1.5  # 带牌比三条加得少
        assert v is not None

        cards_value.append(v)