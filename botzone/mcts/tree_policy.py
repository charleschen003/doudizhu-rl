from mcts.get_bestchild import get_bestchild


def tree_policy(node, my_id):
    while node.state.winner == -1:
        if node.is_all_expand():
            node = get_bestchild(node, my_id)
        else:
            sub_node = node.expand()
            return sub_node
    return node


