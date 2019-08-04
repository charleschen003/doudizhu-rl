import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

sns.set(color_codes=True)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot(fn):
    with open('outs/{}.json'.format(fn)) as f:
        data = json.load(f)
    y = np.array(data) / 100
    plt.plot(y, alpha=0.3)
    sm = gaussian_filter1d(y, sigma=3)
    plt.plot(sm)
    plt.title('地主胜率走势')
    plt.xlabel('训练总百次数')
    plt.ylabel('过去100次AI地主胜率')
    # plt.savefig('images/{}.svg'.format(fn), format='svg')
    plt.show()


if __name__ == '__main__':
    plot('0804_0112')
