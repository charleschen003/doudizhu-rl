import numpy as np
import requests
import random
import math
import json

"""
{
requests:[
    {
        own: [0, 1, 2, 3, 4] // 自己最初拥有哪些牌
        history: [[0, 1, 2]/* 上上家 */, []/* 上家 */], // 总是两项，每一项都是数组，分别表示上上家和上家出的牌，空数组表示跳过回合或者还没轮到他。
        publiccard: [0, 1, 2, 3, 4] // 自己最初拥有哪些牌
    },
    {
        "history": [[0, 1, 2]/* 上上家 */, []/* 上家 */], // 总是两项，每一项都是数组，分别表示上上家和上家出的牌，空数组表示跳过回合。
    },
    {...}
]
}
"""


def get_id(o):
    # 判断自己是什么身份，地主0 or 农民甲1 or 农民乙2
    player_role = 0
    if len(o[0]) == 0:
        if len(o[1]) != 0:
            player_role = 1
    else:
        player_role = 2
    return player_role


full_input = json.loads(input())
req = full_input["requests"]

data = {}
if len(req) == 1:
    use_info = full_input["requests"][0]
    player_id = get_id(use_info["history"])
    data['own'] = use_info["own"]
    data['pid'] = player_id
data['hist0'] = req[-1]['history'][0]
data['hist1'] = req[-1]['history'][1]

url = 'http://104.215.150.235:8080/choose_action/'
print(data)
response = requests.post(url, data)
js = response.json()
strlist = js['ans']
ans = [int(c) for c in strlist]
print(json.dumps({
    "response": ans
}))
