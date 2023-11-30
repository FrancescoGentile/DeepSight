##
##
##

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

import torch

from deepsight import utils
from deepsight.core import Dataset
from deepsight.ops.geometric import coalesce
from deepsight.structures.vision import BoundingBoxes, BoundingBoxFormat, Image
from deepsight.tasks.meic import Annotations, Predictions, Sample
from deepsight.transforms.vision import Transform
from deepsight.typing import Configs, Configurable, PathLike


class H2ODataset(Dataset[Sample, Annotations, Predictions], Configurable):
    """A Multi-Entity Interaction dataset extracted from H2O."""

    # ----------------------------------------------------------------------- #
    # Constructor
    # ----------------------------------------------------------------------- #

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
                - `mei_{split}.json`: A JSON file for each split containing the
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

        self._samples = self._get_samples()
        if max_samples is not None:
            self._samples = self._samples[:max_samples]

        entity_classes = self._get_entity_classes()
        self._entity_class_to_id = {name: i for i, name in enumerate(entity_classes)}

        interaction_classes = self._get_interaction_classes()
        self._interaction_class_to_id = {
            name: i for i, name in enumerate(interaction_classes)
        }

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

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
        """The class ID of the human entity."""
        return self._entity_class_to_id["person"]

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def get_configs(self, recursive: bool) -> Configs:
        configs: Configs = {"split": self._split}

        if recursive and self._transform is not None:
            configs["transform"] = utils.get_configs(self._transform, recursive)

        return configs

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Sample, Annotations, Predictions]:
        file_sample = self._samples[index]

        image_path = self._path / f"images/{self._split}/{file_sample['id']}.jpg"
        image = Image.open(image_path)

        coords = [e["bbox"] for e in file_sample["entities"]]
        entity_boxes = BoundingBoxes(coords, BoundingBoxFormat.XYXY, True, image.size)
        entity_labels = torch.as_tensor(
            [self._entity_class_to_id[e["category"]] for e in file_sample["entities"]]
        )

        interaction_to_idx: dict[tuple[int, ...], int] = {}
        idx_to_label: dict[int, list[str]] = {}
        idx_to_binary: dict[int, set[tuple[int, int]]] = {}
        for interaction in file_sample["interactions"]:
            key = tuple(interaction["entity_indices"])
            if key not in interaction_to_idx:
                idx = len(interaction_to_idx)
                interaction_to_idx[key] = idx
                idx_to_label[idx] = [interaction["verb"]]
                idx_to_binary[idx] = {
                    (i[0], i[1]) for i in interaction["binary_interactions"]
                }
            else:
                idx = interaction_to_idx[key]
                idx_to_label[idx].append(interaction["verb"])
                idx_to_binary[idx].update(
                    (i[0], i[1]) for i in interaction["binary_interactions"]
                )

        interactions = torch.zeros(
            len(entity_boxes), len(interaction_to_idx), dtype=torch.bool
        )
        interaction_labels = torch.zeros(
            len(interaction_to_idx), self.num_interaction_classes, dtype=torch.float
        )

        binary_interactions = []
        for entities, idx in interaction_to_idx.items():
            interactions[list(entities), idx] = True
            for label in idx_to_label[idx]:
                interaction_labels[idx, self._interaction_class_to_id[label]] = 1.0

            for i, j in idx_to_binary[idx]:
                binary_interactions.append((i, j, idx))

        if len(binary_interactions) > 0:
            binary_interactions = torch.as_tensor(binary_interactions).transpose_(0, 1)
            binary_indices, binary_labels = coalesce(
                indices=binary_interactions[:2],
                values=interaction_labels[binary_interactions[2]],
                reduce="max",
            )
        else:
            binary_interactions = torch.empty((3, 0), dtype=torch.long)
            binary_indices = torch.empty((2, 0), dtype=torch.long)
            binary_labels = torch.zeros(
                (0, self.num_interaction_classes), dtype=torch.float
            )

        if self._transform is not None:
            image, entity_boxes = self._transform(image, entity_boxes)
            if len(entity_boxes) != len(entity_labels):
                raise NotImplementedError("Not all entities were kept after transform.")

        sample = Sample(image, entity_boxes, entity_labels)
        annotations = Annotations(interactions, interaction_labels, binary_interactions)
        target = Predictions(
            interactions, interaction_labels, binary_indices, binary_labels
        )

        return sample, annotations, target

    def __str__(self) -> str:
        return "H2O Multi-Entity Dataset"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(split={self._split}, num_samples={len(self)})"
        )

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _get_samples(self) -> list[FileSample]:
        with open(self._path / f"mei_{self._split}.json") as f:
            samples = json.load(f)

        return [s for s in samples if len(s["entities"]) > 0]

    def _get_entity_classes(self) -> list[str]:
        with open(self._path / "categories.json") as file:
            return json.load(file)

    def _get_interaction_classes(self) -> list[str]:
        with open(self._path / "verbs.json") as file:
            return json.load(file)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


class Entity(TypedDict):
    category: str
    bbox: list[float]


class Interaction(TypedDict):
    verb: str
    entity_indices: list[int]
    binary_interactions: list[list[int]]


class FileSample(TypedDict):
    id: str
    entities: list[Entity]
    interactions: list[Interaction]
