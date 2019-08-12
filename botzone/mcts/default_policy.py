from copy import copy


def default_policy(node, my_id):
    current_state = node.get_state()
    #  随机出牌直到游戏结束
    while current_state.winner == -1:
        current_state = current_state.get_next_state_with_random_choice(None)
    final_sate_reward = current_state.compute_reward(my_id)
    return final_sate_reward
