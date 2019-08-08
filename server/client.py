import json
import requests

url = 'http://127.0.0.1:5001/'
server_url = 'http://117.78.4.26:5001'
payload1 = {
    'role_id': 1,  # 0代表地主上家，1代表地主，2代表地主下家
    'last_taken': {  # 更改处
        0: [],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [7,  8,  9, 10, 11, 12, 13],
    },
    'cur_cards': [15, 15, 14, 13, 13, 12, 11, 10,  9,  6,  6,  6,  4],  # 无需保持顺序
    'history': {  # 各家走过的牌的历史The environment
        0: [],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [7,  8,  9, 10, 11, 12, 13],
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
    'cur_cards': [16, 14, 13, 12, 12, 11, 11, 10,  9,  7,  6,  6,  4,  4,  4,  4],  # 无需保持顺序
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

res = requests.post(server_url, json=payload1)
print(json.loads(res.content))

res = requests.post(server_url, json=payload2)
print(json.loads(res.content))
