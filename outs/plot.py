import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import config as conf

sns.set(color_codes=True)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams["figure.figsize"] = [9.6, 4.8]  # 设置大小，默认为[6.4, 4.8]


def plot(fn):
    path = os.path.join(conf.WIN_DIR, conf.name_dir(fn))
    path = '{}.json'.format(path)
    with open(path) as f:
        data = json.load(f)
    title = '未知'
    if isinstance(data, list):
        title = '地主胜率走势'
        y = np.array(data) / 100
        plt.plot(y, alpha=0.3)
        sm = gaussian_filter1d(y, sigma=3)
        plt.plot(sm)
    elif isinstance(data, dict):
        title = '胜率走势'
        for i, (k, v) in enumerate(data.items()):
            y = np.array(v) / 100
            plt.plot(y, alpha=0.3,color='C{}'.format(i))
            sm = gaussian_filter1d(y, sigma=5)
            plt.plot(sm, label=k, color='C{}'.format(i))
    plt.title(title)
    plt.xlabel('训练总百次数')
    plt.ylabel('过去100次AI地主胜率')
    plt.legend()

    path = os.path.join(conf.IMG_DIR, conf.name_dir(fn))
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    path = '{}.svg'.format(path)
    plt.savefig(path, format='svg')
    print('Saved at {}'.format(path))


if __name__ == '__main__':
    plot('0808_0918')
