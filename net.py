import os
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import config as conf


class Net(nn.Module, ABC):
    def save(self, name, max_split=2):
        path = os.path.join(conf.MODEL_DIR, conf.name_dir(name, max_split))
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        path = '{}.pt'.format(path)
        torch.save(self.state_dict(), path)

    def load(self, name=None, abspath=None, max_split=2):
        if abspath:
            path = abspath
        else:
            path = os.path.join(conf.MODEL_DIR, conf.name_dir(name, max_split))
            path = '{}.pt'.format(path)
        map_location = 'cpu' if conf.DEVICE.type == 'cpu' else None
        static_dict = torch.load(path, map_location)
        self.load_state_dict(static_dict)
        self.eval()
        print("Loaded model from {}.".format(path))


class NetFirst(Net):
    def __init__(self):
        # input shape: 5 * 15 * 4
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(5, 256, (1, 1), (1, 4))  # 256 * 15 * 1
        self.conv2 = nn.Conv2d(5, 256, (1, 2), (1, 4))
        self.conv3 = nn.Conv2d(5, 256, (1, 3), (1, 4))
        self.conv4 = nn.Conv2d(5, 256, (1, 4), (1, 4))
        self.convs = (self.conv1, self.conv2, self.conv3, self.conv4)  # 256 * 15 * 4
        self.pool = nn.MaxPool2d((1, 4))  # 256 * 15 * 1
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 15, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, face, actions):
        """
        :param face: 当前状态  4 * 15 * 4
        :param actions: 所有动作 batch_size * 15 * 4
        :return:
        """
        if face.dim() == 3:
            face = face.unsqueeze(0).repeat((actions.shape[0], 1, 1, 1))
        actions = actions.unsqueeze(1)
        state_action = torch.cat((face, actions), dim=1)

        x = torch.cat([f(state_action) for f in self.convs], -1)
        x = self.pool(x)
        x = x.view(actions.shape[0], -1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetComplicated(Net):
    def __init__(self):
        # input shape: 5 * 15 * 4
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(5, 256, (1, 1), (1, 4))  # 256 * 15 * 1
        self.conv2 = nn.Conv2d(5, 256, (1, 2), (1, 4))
        self.conv3 = nn.Conv2d(5, 256, (1, 3), (1, 4))
        self.conv4 = nn.Conv2d(5, 256, (1, 4), (1, 4))
        self.convs = (self.conv1, self.conv2, self.conv3, self.conv4)  # 256 * 15 * 4
        self.conv_shunzi = nn.Conv2d(5, 256, (15, 1), 1)  # 256 * 1 * 4
        self.pool = nn.MaxPool2d((1, 4))  # 256 * 15 * 1
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * (15 + 4), 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, face, actions):
        """
        :param face: 当前状态  face_deep(根据env固定) * 15 * 4
        :param actions: 所有动作 batch_size * 15 * 4
        :return:
        """
        if face.dim() == 3:
            face = face.unsqueeze(0).repeat((actions.shape[0], 1, 1, 1))
        actions = actions.unsqueeze(1)
        state_action = torch.cat((face, actions), dim=1)

        x = torch.cat([f(state_action) for f in self.convs], -1)
        x = self.pool(x)
        x = x.view(actions.shape[0], -1)

        x_shunzi = self.conv_shunzi(state_action).view(actions.shape[0], -1)
        x = torch.cat([x, x_shunzi], -1)

        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetMoreComplicated(NetComplicated):
    def __init__(self):
        # input shape: 8 * 15 * 4
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(8, 256, (1, 1), (1, 4))  # 256 * 15 * 1
        self.conv2 = nn.Conv2d(8, 256, (1, 2), (1, 4))
        self.conv3 = nn.Conv2d(8, 256, (1, 3), (1, 4))
        self.conv4 = nn.Conv2d(8, 256, (1, 4), (1, 4))
        self.convs = (self.conv1, self.conv2, self.conv3, self.conv4)  # 256 * 15 * 4
        self.conv_shunzi = nn.Conv2d(8, 256, (15, 1), 1)  # 256 * 1 * 4
        self.pool = nn.MaxPool2d((1, 4))  # 256 * 15 * 1
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * (15 + 4), 256)
        self.fc2 = nn.Linear(256, 1)
