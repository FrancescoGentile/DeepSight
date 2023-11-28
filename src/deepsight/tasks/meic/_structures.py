##
##
##

from dataclasses import dataclass
from typing import Annotated

import torch
from torch import Tensor
from typing_extensions import Self

from deepsight.structures.vision import BoundingBoxes, Image
from deepsight.typing import Moveable


@dataclass(frozen=True, slots=True)
class Sample(Moveable):
    """A Multi-Entity Interaction (MEI) sample with object detection annotations.

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
    """The output for a Multi-Entity Interaction (MEI) sample.

    Attributes:
        interactions: The ground-truth interactions between the entities. This is a
            binary tensor of shape `(N, H)`, where `N` is the number of entities and
            `H` is the number of interactions. The `(i, j)`-th element of this tensor
            is `True` if the `i`-th entity is involved in the `j`-th interaction.
        interaction_labels: The probabilities of the interactions. The shape of this
            tensor is `(H, C)` where `H` is the number of interactions and `C` is the
            number of interaction classes. The `(i, j)`-th element of this tensor is
            the probability that the `i`-th interaction is of class `j`. Note that an
            interaction can be of multiple classes (thus, the sum of the probabilities
            of each interaction can be greater than 1.0).
        binary_interactions: The ground-truth human-object and human-human interactions.
            This is a tensor of shape `(3, E)` where `E` is the number of interactions.
            Each row of this tensor is a triplet `(i, j, k)` where `i` is the index of
            the human entity, `j` is the index of the target entity (human or object),
            and `k` is the index of the multi-entity interaction that involves both
            entities.
    """

    interactions: Annotated[Tensor, "N H", bool]
    interaction_labels: Annotated[Tensor, "H C", float]
    binary_interactions: Annotated[Tensor, "3 E", bool]

    @property
    def device(self) -> torch.device:
        return self.interactions.device

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        if self.device == torch.device(device):
            return self

        return self.__class__(
            interactions=self.interactions.to(device, non_blocking=non_blocking),
            interaction_labels=self.interaction_labels.to(
                device, non_blocking=non_blocking
            ),
            binary_interactions=self.binary_interactions.to(
                device, non_blocking=non_blocking
            ),
        )


@dataclass(frozen=True, slots=True)
class Predictions(Moveable):
    """The output for a Multi-Entity Interaction (MEI) sample.

    Attributes:
        interactions: The ground-truth interactions between the entities. This is a
            binary tensor of shape `(N, H)`, where `N` is the number of entities and
            `H` is the number of interactions. The `(i, j)`-th element of this tensor
            is `True` if the `i`-th entity is involved in the `j`-th interaction.
        interaction_labels: The probabilities of the interactions. The shape of this
            tensor is `(H, C)` where `H` is the number of interactions and `C` is the
            number of interaction classes. The `(i, j)`-th element of this tensor is
            the probability that the `i`-th interaction is of class `j`. Note that an
            interaction can be of multiple classes (thus, the sum of the probabilities
            of each interaction can be greater than 1.0).
        binary_interactions: The ground-truth human-object and human-human interactions.
            This is a tensor of shape `(2, E)` where `E` is the number of interactions.
            Each row of this tensor is a pair `(i, j)` where `i` is the index of the
            human entity and `j` is the index of the target entity (human or object).
        binary_interaction_labels: The probabilities of the binary interactions. The
            shape of this tensor is `(E, C)`, where `E` is the number of binary
            interactions and `C` is the number of interaction classes. The `(i, j)`-th
            element of this tensor is the probability that the `i`-th binary interaction
            is of class `j`. Note that a binary interaction can be of multiple classes
            (thus, the sum of the probabilities of each binary interaction can be
            greater than 1.0).
    """

    interactions: Annotated[Tensor, "N H", bool]
    interaction_labels: Annotated[Tensor, "H C", float]
    binary_interactions: Annotated[Tensor, "2 E", int]
    binary_interaction_labels: Annotated[Tensor, "E C", float]

    @property
    def device(self) -> torch.device:
        return self.interactions.device

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        if self.device == torch.device(device):
            return self

        return self.__class__(
            interactions=self.interactions.to(device, non_blocking=non_blocking),
            interaction_labels=self.interaction_labels.to(
                device, non_blocking=non_blocking
            ),
            binary_interactions=self.binary_interactions.to(
                device, non_blocking=non_blocking
            ),
            binary_interaction_labels=self.binary_interaction_labels.to(
                device, non_blocking=non_blocking
            ),
        )
