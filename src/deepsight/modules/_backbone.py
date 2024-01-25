##
##
##

import abc
from collections.abc import Iterable
from dataclasses import dataclass

from deepsight.structures import BatchedImages

from ._module import Module


class Backbone(Module, abc.ABC):
    """Base class for vision backbones.

    A vision backbones takes in input a batch of images (or features) and returns
    the extracted features.
    """

    @dataclass(frozen=True)
    class StageInfo:
        """Information about a stage of the backbone.

        Attributes:
            name: The name of the stage.
            channels: The number of output channels.
        """

        name: str
        out_channels: int

    @abc.abstractmethod
    def get_stages_info(self) -> tuple[StageInfo, ...]:
        """Get information about the stages of the backbone.

        !!! note

            The stages are returned in the same order as they are processed by the
            backbone.
        """
        ...

    @abc.abstractmethod
    def __call__(
        self,
        images: BatchedImages,
        *,
        return_stages: Iterable[str | int] = (-1,),
    ) -> tuple[BatchedImages, ...]:
        """Extract the features from the given images.

        Args:
            images: The batch of images to process.
            return_stages: The stages from which to return the features. A stage can be
                specified either by its name or by its index (indexing can be negative
                to count from the end).

        Returns:
            The extracted features for each stage specified in `return_stages`.
        """
        ...
