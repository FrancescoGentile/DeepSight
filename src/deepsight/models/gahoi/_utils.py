##
##
##

from typing import Literal

import torch

from deepsight.typing import Tensor


def get_interaction_mask(
    obj_to_interactions: list[list[int]] | None,
    num_entity_classes: int,
    num_interaction_classes: int,
) -> Tensor[Literal["E V"], bool] | None:
    if obj_to_interactions is None:
        return None

    if len(obj_to_interactions) != num_entity_classes:
        raise ValueError(
            "The number of lists in `obj_to_interactions` must be equal to "
            "the number of object classes, but got "
            f"{len(obj_to_interactions)} lists and "
            f"{num_entity_classes} object classes."
        )

    mask = torch.ones(num_entity_classes, num_interaction_classes, dtype=torch.bool)
    for obj_class, interactions in enumerate(obj_to_interactions):
        mask[obj_class, interactions] = False

    return mask
