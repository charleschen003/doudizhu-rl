def backup(node, reward):
    while node is not None:
        node.visit += 1
        node.reward += reward
        node = node.parent
