##
##
##

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

import torch

from deepsight.core import Dataset
from deepsight.structures.vision import BoundingBoxes, BoundingBoxFormat, Image
from deepsight.tasks.hoic import Annotations, Predictions, Sample
from deepsight.transforms.vision import Transform
from deepsight.typing import PathLike


class HICODETDataset(Dataset[Sample, Annotations, Predictions]):
    """The HICO-DET dataset."""

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
        super().__init__()

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
        interaction_classes.remove("no_interaction")
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

    def get_object_valid_interactions(self) -> list[list[int]]:
        """Return for each entity class the interactions in which it can participate.

        Returns:
            A list of lists. The list at index `i` contains the interaction
            classes in which the entity class with ID `i` can participate.
        """
        file = self._path / "actions.json"
        if not file.exists():
            raise RuntimeError("actions.json does not exist.")

        with open(file) as f:
            data = json.load(f)

        object_valid_interactions = [[] for _ in range(self.num_entity_classes)]
        for action in data:
            if action["vname"] == "no_interaction":
                continue

            object_idx = self._entity_class_to_id[action["nname"]]
            interaction_idx = self._interaction_class_to_id[action["vname"]]

            object_valid_interactions[object_idx].append(interaction_idx)

        return object_valid_interactions

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
        entity_boxes = BoundingBoxes(coords, BoundingBoxFormat.XYXY, False, image.size)

        entity_labels = torch.as_tensor(
            [self._entity_class_to_id[e["category"]] for e in file_sample["entities"]]
        )

        interaction_to_idx: dict[tuple[int, int], int] = {}
        for action in file_sample["actions"]:
            interaction = (action["subject"], action["target"])
            if action["verb"] == "no_interaction":
                continue

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
            if action["verb"] == "no_interaction":
                continue

            interaction = (action["subject"], action["target"])
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

    # ---------------------------------------------------------------------- #
    # Private Methods
    # ---------------------------------------------------------------------- #

    def _get_samples(self) -> list[FileSample]:
        with open(self._path / f"{self._split}.json") as file:
            data = json.load(file)

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
    target: int
    verb: str


class FileSample(TypedDict):
    id: str
    entities: list[Entity]
    actions: list[Actions]
