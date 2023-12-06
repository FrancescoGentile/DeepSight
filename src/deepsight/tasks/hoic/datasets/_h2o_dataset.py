##
##
##

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import torch

from deepsight import utils
from deepsight.core import Dataset
from deepsight.structures.vision import BoundingBoxes, BoundingBoxFormat, Image
from deepsight.tasks.hoic import Annotations, Predictions, Sample
from deepsight.transforms.vision import Transform
from deepsight.typing import Configs, Configurable, PathLike


class H2ODataset(Dataset[Sample, Annotations, Predictions], Configurable):
    """The Human-to-Human-or-Object (H2O) Interaction Dataset."""

    def __init__(
        self,
        path: PathLike,
        split: Literal["train", "test"],
        max_samples: int | None = None,
        transform: Transform | None = None,
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
            split: The split of the dataset to load. At the moment, only the `train` and
                `test` splits are supported.
            max_samples: The maximum number of samples to load. If `None`, all samples
                are loaded. Must be greater than 0.
            transform: An optional transform to apply to the images and bounding boxes.
                At the moment, only transforms that do not remove entities are
                supported.
        """
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be greater than 0.")

        self._path = Path(path)
        self._split = split
        self._transform = transform

        self._samples = self._get_samples(self._path, self._split)
        if max_samples is not None:
            self._samples = self._samples[:max_samples]

        entity_classes = self._get_entity_classes(self._path)
        self._entity_class_to_id = {name: i for i, name in enumerate(entity_classes)}

        interaction_classes = self._get_interaction_classes(self._path)
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
    # Public Methods
    # ---------------------------------------------------------------------- #

    def get_object_valid_interactions(
        self,
        splits: Iterable[Literal["train", "test"]],
    ) -> list[list[int]]:
        """Return for each entity class the interactions in which it can participate.

        !!! warning

            Differently from HICO-DET, the H2O dataset does not provide the list of
            interaction classes that are valid for a given object, thus this method
            computes this list from the dataset annotations.

        Args:
            splits: The splits to consider when computing the valid interactions.

        Returns:
            A list of lists. The list at index `i` contains the interaction
            classes in which the entity class with ID `i` can participate.
        """
        valid_interactions = [set() for _ in range(self.num_entity_classes)]

        for split in splits:
            samples = self._get_samples(self._path, split)
            for sample in samples:
                for action in sample["actions"]:
                    object_ = action.get("target", action.get("instrument", None))

                    if object_ is None:
                        continue

                    object_class = sample["entities"][object_]["category"]
                    object_class_id = self._entity_class_to_id[object_class]
                    interaction_class_id = self._interaction_class_to_id[action["verb"]]

                    valid_interactions[object_class_id].add(interaction_class_id)

        return [list(interactions) for interactions in valid_interactions]

    def get_configs(self, recursive: bool) -> Configs:
        configs: Configs = {"split": self._split}
        if recursive and self._transform is not None:
            configs["transform"] = utils.get_configs(self._transform, recursive)

        return configs

    # ---------------------------------------------------------------------- #
    # Magic Methods
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Sample, Annotations, Predictions]:
        file_sample = self._samples[index]

        image_path = self._path / f"images/{self._split}/{file_sample['id']}.jpg"
        image = Image.open(image_path, mode=Image.Mode.RGB)

        coords = [e["bbox"] for e in file_sample["entities"]]
        entity_boxes = BoundingBoxes(coords, BoundingBoxFormat.XYXY, True, image.size)

        entity_labels = torch.as_tensor(
            [self._entity_class_to_id[e["category"]] for e in file_sample["entities"]]
        )

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

        if len(interaction_to_idx) == 0:
            interactions = torch.empty((0, 2), dtype=torch.long)
        else:
            interactions = torch.as_tensor(list(interaction_to_idx.keys()))  # (I, 2)

        interaction_labels = torch.zeros(
            (len(interaction_to_idx), self.num_interaction_classes), dtype=torch.float
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

        if self._transform is not None:
            image, entity_boxes = self._transform(image, entity_boxes)
            if len(entity_boxes) != len(entity_labels):
                raise NotImplementedError("Not all entities were kept after transform.")

        sample = Sample(image, entity_boxes, entity_labels)
        annotations = Annotations(interactions, interaction_labels)
        target = Predictions(interactions, interaction_labels)

        return sample, annotations, target

    def __str__(self) -> str:
        return "H2O Dataset"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(split={self._split}, num_samples={len(self)})"
        )

    # ---------------------------------------------------------------------- #
    # Private Methods
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _get_samples(path: Path, split: str) -> list[FileSample]:
        with open(path / f"{split}.json") as file:
            data = json.load(file)

        return [s for s in data if len(s["entities"]) > 0]

    @staticmethod
    def _get_entity_classes(path: Path) -> list[str]:
        with open(path / "categories.json") as file:
            return json.load(file)

    @staticmethod
    def _get_interaction_classes(path: Path) -> list[str]:
        with open(path / "verbs.json") as file:
            return json.load(file)


# -------------------------------------------------------------------------- #
# Data classes
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
