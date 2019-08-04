# https://github.com/qq456cvb/doudizhu-C
from rule_utils.card import Card, action_space, CardGroup, augment_action_space_onehot60, augment_action_space, clamp_action_idx
from rule_utils.utils import get_mask_onehot60
import numpy as np
import sys
from env import get_combinations_nosplit, get_combinations_recursive


class Decomposer:
    def __init__(self, num_actions=(100, 21)):
        self.num_actions = num_actions

    def get_combinations(self, curr_cards_char, last_cards_char):
        if len(curr_cards_char) > 10:
            card_mask = Card.char2onehot60(curr_cards_char).astype(np.uint8)
            mask = augment_action_space_onehot60
            a = np.expand_dims(1 - card_mask, 0) * mask
            invalid_row_idx = set(np.where(a > 0)[0])
            if len(last_cards_char) == 0:
                invalid_row_idx.add(0)

            valid_row_idx = [i for i in range(len(augment_action_space)) if i not in invalid_row_idx]

            mask = mask[valid_row_idx, :]
            idx_mapping = dict(zip(range(mask.shape[0]), valid_row_idx))

            # augment mask
            # TODO: known issue: 555444666 will not decompose into 5554 and 66644
            combs = get_combinations_nosplit(mask, card_mask)
            combs = [([] if len(last_cards_char) == 0 else [0]) + [clamp_action_idx(idx_mapping[idx]) for idx in comb] for
                     comb in combs]

            if len(last_cards_char) > 0:
                idx_must_be_contained = set(
                    [idx for idx in valid_row_idx if CardGroup.to_cardgroup(augment_action_space[idx]). \
                        bigger_than(CardGroup.to_cardgroup(last_cards_char))])
                combs = [comb for comb in combs if not idx_must_be_contained.isdisjoint(comb)]
                fine_mask = np.zeros([len(combs), self.num_actions[1]], dtype=np.bool)
                for i in range(len(combs)):
                    for j in range(len(combs[i])):
                        if combs[i][j] in idx_must_be_contained:
                            fine_mask[i][j] = True
            else:
                fine_mask = None
        else:
            mask = get_mask_onehot60(curr_cards_char, action_space, None).reshape(len(action_space), 15, 4).sum(-1).astype(
                np.uint8)
            valid = mask.sum(-1) > 0
            cards_target = Card.char2onehot60(curr_cards_char).reshape(-1, 4).sum(-1).astype(np.uint8)
            # do not feed empty to C++, which will cause infinite loop
            combs = get_combinations_recursive(mask[valid, :], cards_target)
            idx_mapping = dict(zip(range(valid.shape[0]), np.where(valid)[0]))

            combs = [([] if len(last_cards_char) == 0 else [0]) + [idx_mapping[idx] for idx in comb] for comb in combs]

            if len(last_cards_char) > 0:
                valid[0] = True
                idx_must_be_contained = set(
                    [idx for idx in range(len(action_space)) if valid[idx] and CardGroup.to_cardgroup(action_space[idx]). \
                        bigger_than(CardGroup.to_cardgroup(last_cards_char))])
                combs = [comb for comb in combs if not idx_must_be_contained.isdisjoint(comb)]
                fine_mask = np.zeros([len(combs), self.num_actions[1]], dtype=np.bool)
                for i in range(len(combs)):
                    for j in range(len(combs[i])):
                        if combs[i][j] in idx_must_be_contained:
                            fine_mask[i][j] = True
            else:
                fine_mask = None
        return combs, fine_mask
