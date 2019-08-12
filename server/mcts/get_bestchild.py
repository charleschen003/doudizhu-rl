import numpy as np


def UCB1(node, c=0.7):
    visit = np.array([n.visit for n in node.children])
    reward = np.array([n.reward for n in node.children])
    values = reward/visit + c * np.sqrt(2*np.log(node.visit) / visit)
    index = np.where(values == np.max(values))
    return np.array(node.children)[index]


def UCB2(node, c=0.7):
    visit = np.array([n.visit for n in node.children])
    reward = np.array([n.reward for n in node.children])
    values = reward / visit + c * np.sqrt(2 * np.log(node.visit) / visit)
    index = np.where(values == np.min(values))
    return np.array(node.children)[index]


def get_bestchild(node, my_id):
    if node.state.my_id == my_id:
        nodes = UCB1(node)
        if len(nodes) == 1:
            return nodes[0]
        else:
            return np.random.choice(nodes)
    else:
        nodes = UCB2(node)
        if len(nodes) == 1:
            return nodes[0]
        else:
            return np.random.choice(nodes)

def get_bestchild_(node):
    visit = np.array([n.visit for n in node.children])
    reward = np.array([n.reward for n in node.children])
    values = reward / visit
    index = np.where(values == np.max(values))
    nodes = np.array(node.children)[index]
    if len(nodes) == 1:
        return nodes[0]
    else:
        return np.random.choice(nodes)