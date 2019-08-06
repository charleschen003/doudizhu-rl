import json
import requests

url = 'http://117.78.4.26:5000/'

payload1 = {
    'role_id': 1,  # 0代表地主上家，1代表地主，2代表地主下家
    'last_taken': [11],  # 上家出的牌
    'cur_cards': [15, 15, 14, 13, 13, 11, 10, 7, 7, 7, 3],  # 无需保持顺序
    'history': {  # 各家走过的牌的历史The environment
        0: [10],
        1: [3, 3, 3, 4, 4, 5, 11],
        2: [5, 5, 6, 6, 6, 9],
    },
    'left': {  # 各家剩余的牌
        0: 16,
        1: 13,
        2: 11,
    }
}

payload2 = {
    'role_id': 1,  # 0代表地主上家，1代表地主，2代表地主下家
    'last_taken': [],  # 上家出的牌
    'cur_cards': [17, 15, 14, 13, 12, 12, 11, 10, 6, 6, 4, 4],  # 无需保持顺序
    'history': {  # 各家走过的牌的历史The environment
        0: [],
        1: [7, 7, 7, 3, 8, 8, 8, 8],
        2: [15, 15, 15, 4],
    },
    'left': {  # 各家剩余的牌
        0: 17,
        1: 12,
        2: 13,
    }
}

for data in [payload1, payload2]:
    res = requests.post(url, json=data)
    print(json.loads(res.content))
