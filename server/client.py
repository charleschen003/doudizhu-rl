import requests
url = 'http://127.0.0.1:5000/'

handcards = [3, ]

res = requests.post(url, json={'1': 2, '3': 4})
print(res.content)
