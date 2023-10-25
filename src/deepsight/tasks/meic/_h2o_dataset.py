##
##
##

from pathlib import Path

from deepsight.tasks import Dataset, Split

from ._structures import Annotations, Predictions, Sample


class H2ODataset(Dataset[Sample, Annotations, Predictions]):
    """A Multi-Entity Interaction dataset extracted from H2O."""

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

    def __init__(self, path: Path | str, split: Split) -> None:
        self.path = Path(path)
        self.split = split

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def num_interaction_classes(self) -> int:
        """The number of interaction classes."""
        raise NotImplementedError

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[Sample, Annotations, Predictions]:
        raise NotImplementedError
