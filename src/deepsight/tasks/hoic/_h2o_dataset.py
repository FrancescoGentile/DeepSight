##
##
##

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

import torch
from typing_extensions import NotRequired

from deepsight.structures import BoundingBoxes, BoundingBoxFormat, Image
from deepsight.tasks import Dataset, Split

from ._structures import Annotations, Predictions, Sample


class H2ODataset(Dataset[Sample, Annotations, Predictions]):
    """The Human-to-Human-or-Object (H2O) Interaction Dataset."""

    def __init__(
        self,
        path: Path | str,
        split: Literal[Split.TRAIN, Split.TEST],
    ) -> None:
        """Initializes a new H2O dataset.

        Args:
            path: The path to the dataset. This should be a directory containing
                the following files:
                - `images/`: A directory containing the images. This directory
                    should be further split into a `train/` and `test/` directory.
                - `{split}.json`: A JSON file for each split containing the
                    annotations for the samples in the split.
                - `categories.json`: A JSON file containing the names of the entity
                    classes.
                - `verbs.json`: A JSON file containing the names of the interaction
                    classes.
            split: The split of the dataset to load. At the moment, only the
                `train` and `test` splits are supported.
        """
        super().__init__()

        self._path = Path(path)
        self._split = split
        self._samples = self._get_samples()

        entity_classes = self._get_entity_classes()
        self._entity_class_to_id = {name: i for i, name in enumerate(entity_classes)}

        interaction_classes = self._get_interaction_classes()
        self._interaction_class_to_id = {
            name: i for i, name in enumerate(interaction_classes)
        }

    # ---------------------------------------------------------------------- #
    # Properties
    # ---------------------------------------------------------------------- #

    @property
    def num_entity_classes(self) -> int:
        """The number of entity classes."""
        return len(self._entity_class_to_id)

    @property
    def num_interaction_classes(self) -> int:
        """The number of interaction classes."""
        return len(self._interaction_class_to_id)

    @property
    def human_class_id(self) -> int:
        """The class ID for human entities."""
        return self._entity_class_to_id["person"]

    # ---------------------------------------------------------------------- #
    # Magic Methods
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Sample, Annotations, Predictions]:
        file_sample = self._samples[index]

        image_path = self._path / f"images/{self._split}/{file_sample['id']}.jpg"
        image = Image.open(image_path)

        coords = [e["bbox"] for e in file_sample["entities"]]
        entities = BoundingBoxes(coords, BoundingBoxFormat.XYXY, True, image.size)

        entity_labels = [e["category"] for e in file_sample["entities"]]
        entity_labels = torch.as_tensor(entity_labels)

        interaction_to_idx: dict[tuple[int, int], int] = {}
        for action in file_sample["actions"]:
            subject = action["subject"]
            object_ = action.get("target", action.get("instrument", None))

            # we only care about interactions
            if object_ is None:
                continue

            interaction = (subject, object_)
            if interaction not in interaction_to_idx:
                interaction_to_idx[interaction] = len(interaction_to_idx)

        interactions = torch.as_tensor(list(interaction_to_idx.keys())).transpose_(0, 1)
        interaction_labels = torch.zeros(
            (len(interaction_to_idx), self.num_interaction_classes),
            dtype=torch.float,
        )

        for action in file_sample["actions"]:
            subject = action["subject"]
            object_ = action.get("target", action.get("instrument", None))

            if object_ is None:
                continue

            interaction = (subject, object_)
            interaction_idx = interaction_to_idx[interaction]
            interaction_class_id = self._interaction_class_to_id[action["verb"]]
            interaction_labels[interaction_idx, interaction_class_id] = 1.0

        sample = Sample(image, entities, entity_labels)
        annotations = Annotations(interactions, interaction_labels)
        target = Predictions(interactions, interaction_labels)

        return sample, annotations, target

    # ---------------------------------------------------------------------- #
    # Private Methods
    # ---------------------------------------------------------------------- #

    def _get_samples(self) -> list[FileSample]:
        with open(self._path / f"{self._split}.json") as file:
            data = json.load(file)

        # FIXME: should we exclude samples without entities?
        # exclude samples without entities
        return [s for s in data if len(s["entities"]) > 0]

    def _get_entity_classes(self) -> list[str]:
        with open(self._path / "categories.json") as file:
            return json.load(file)

    def _get_interaction_classes(self) -> list[str]:
        with open(self._path / "verbs.json") as file:
            return json.load(file)


# -------------------------------------------------------------------------- #
# Data
# -------------------------------------------------------------------------- #


class Entity(TypedDict):
    bbox: list[float]
    category: str


class Actions(TypedDict):
    subject: int
    target: NotRequired[int]
    instrument: NotRequired[int]
    verb: str


class FileSample(TypedDict):
    id: str
    entities: list[Entity]
    actions: list[Actions]
