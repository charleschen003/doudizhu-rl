import json
import requests

url = 'http://127.0.0.1:5000/'
server_url = 'http://117.78.4.26:5000'
payload1 = {
    'role_id': 1,  # 0代表地主上家，1代表地主，2代表地主下家
    'last_taken': {  # 更改处
        0: [],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [7, 8, 9, 10, 11, 12, 13],
    },
    'cur_cards': [15, 15, 14, 13, 13, 12, 11, 10, 9, 6, 6, 6, 4],  # 无需保持顺序
    'history': {  # 各家走过的牌的历史The environment
        0: [],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [7, 8, 9, 10, 11, 12, 13],
    },
    'left': {  # 各家剩余的牌
        0: 17,
        1: 13,
        2: 10,
    },
    'debug': False,  # 是否返回debug
}
payload2 = {
    'role_id': 1,  # 0代表地主上家，1代表地主，2代表地主下家
    'last_taken': {  # 更改处
        0: [],
        1: [15, 15],
        2: [],
    },
    'cur_cards': [16, 14, 13, 12, 12, 11, 11, 10, 9, 7, 6, 6, 4, 4, 4, 4],  # 无需保持顺序
    'history': {  # 各家走过的牌的历史The environment
        0: [8, 8],
        1: [5, 5, 15, 15],
        2: [7, 7],
    },
    'left': {  # 各家剩余的牌
        0: 15,
        1: 16,
        2: 15,
    },
    'debug': True,  # 是否返回debug
}

# res = requests.post(server_url, json=payload1)
# print(json.loads(res.content))
#
# res = requests.post(server_url, json=payload2)
# print(json.loads(res.content))
record = {
    'is_human': {
        '0': 1,
        '1': 0,
        '2': 0,
    },
    'record': {
        '0': 1,
        '1': 0,
        '2': 0,
    }
}

payload3 = {
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
for i in range(10):
    print(i)
    res = requests.post('http://40.115.138.207:5000/', json=payload3)
end = time.time()
print(end - start)
