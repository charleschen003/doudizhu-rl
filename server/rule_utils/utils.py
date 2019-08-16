import server.rule_utils.card as card
from server.rule_utils.card import action_space
import numpy as np
from collections import Counter


action_space_single = action_space[1:16]
action_space_pair = action_space[16:29]
action_space_triple = action_space[29:42]
action_space_quadric = action_space[42:55]


def counter_subset(list1, list2):
    c1, c2 = Counter(list1), Counter(list2)
    for (k, n) in c1.items():
        if n > c2[k]:
            return False
    return True


def get_mask_onehot60(cards, action_space, last_cards):
    # 1 valid; 0 invalid
    mask = np.zeros([len(action_space), 60])
    if cards is None:
        return mask
    if len(cards) == 0:
        return mask
    for j in range(len(action_space)):
        if counter_subset(action_space[j], cards):
            mask[j] = card.Card.char2onehot60(action_space[j])
    if last_cards is None:
        return mask
    if len(last_cards) > 0:
        for j in range(1, len(action_space)):
            if np.sum(mask[j]) > 0 and not card.CardGroup.to_cardgroup(action_space[j]).\
                    bigger_than(card.CardGroup.to_cardgroup(last_cards)):
                mask[j] = np.zeros([60])
    return mask
