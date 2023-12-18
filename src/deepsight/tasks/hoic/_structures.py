##
##
##

from dataclasses import dataclass
from typing import Annotated, Self

import torch
from torch import Tensor

from deepsight.structures.vision import BoundingBoxes, Image
from deepsight.typing import Moveable


@dataclass(frozen=True, slots=True)
class Sample(Moveable):
    """A Human-Object Interaction Classification sample.

    Attributes:
        image: The image containing the scene with the entities and interactions.
        entity_boxes: The bounding boxes of the entities in the image.
        entity_labels: The class indices of the entities. The length of this
            tensor is equal to the number of bounding boxes in `entities`.
    """

    image: Image
    entity_boxes: BoundingBoxes
    entity_labels: Annotated[Tensor, "N", int]

    @property
    def device(self) -> torch.device:
        return self.image.device

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        if self.device == torch.device(device):
            return self

        return self.__class__(
            self.image.to(device, non_blocking=non_blocking),
            self.entity_boxes.to(device, non_blocking=non_blocking),
            self.entity_labels.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class Annotations(Moveable):
    """The ground-truth annotations for a Human-Object Interaction (HOI) sample.

    Attributes:
        interaction_indices: The indices of the interactions between the entities.
            This is a tensor of shape `(I, 2)`, where `I` is the number of interactions.
            The first row contains the indices of the subject entities (i.e., the human)
            and the second row contains the indices of the object entities (if
            human-human interactions are included, the object entity can also
            be a human).
        interaction_labels: The probabilities of the interactions. The shape of this
            tensor is `(I, C)` where `I` is the number of interactions and `C` is the
            number of interaction classes. The `(i, j)`-th element of this tensor is
            the probability that the `i`-th interaction is of class `j`. Note that an
            interaction can be of multiple classes (thus, the sum of the probabilities
            of each interaction can be greater than 1.0).
    """

    interaction_indices: Annotated[Tensor, "I 2", int]
    interaction_labels: Annotated[Tensor, "I C", float]

    @property
    def device(self) -> torch.device:
        return self.interaction_indices.device

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        if self.device == torch.device(device):
            return self

        return self.__class__(
            self.interaction_indices.to(device, non_blocking=non_blocking),
            self.interaction_labels.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class Predictions(Moveable):
    """The output for a Human-Object Interaction (HOI) sample.

    Attributes:
        interaction_indices: The indices of the interactions between the entities.
            This is a tensor of shape `(I, 2)`, where `I` is the number of interactions.
            The first row contains the indices of the subject entities (i.e., the human)
            and the second row contains the indices of the object entities (if
            human-human interactions are included, the object entity can also
            be a human).
        interaction_labels: The probabilities of the interactions. The shape of this
            tensor is `(I, C)` where `I` is the number of interactions and `C` is the
            number of interaction classes. The `(i, j)`-th element of this tensor is
            the probability that the `i`-th interaction is of class `j`. Note that an
            interaction can be of multiple classes (thus, the sum of the probabilities
            of each interaction can be greater than 1.0).
    """

    interaction_indices: Annotated[Tensor, "I 2", int]
    interaction_labels: Annotated[Tensor, "I C", float]

    @property
    def device(self) -> torch.device:
        return self.interaction_indices.device

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        if self.device == torch.device(device):
            return self

        return self.__class__(
            self.interaction_indices.to(device, non_blocking=non_blocking),
            self.interaction_labels.to(device, non_blocking=non_blocking),
        )
