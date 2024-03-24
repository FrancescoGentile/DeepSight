# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Self

import torch
from torch.optim import Optimizer

from . import utils
from ._misc import ClipGradNorm, ClipGradValue
from .data import DataLoader
from .metrics import Evaluator
from .models import Criterion
from .schedulers import LRScheduler
from .time import PhaseTimestamp
from .typing import Configurable, Moveable, Stateful

if TYPE_CHECKING:
    from ._state import State


# --------------------------------------------------------------------------- #
# Phase
# --------------------------------------------------------------------------- #

type Phase[S, O, A, P] = TrainingPhase[S, O, A, P] | EvaluationPhase[S, O, A, P]

# --------------------------------------------------------------------------- #
# Training Phase
# --------------------------------------------------------------------------- #


class TrainingPhase[S, O, A, P](Stateful, Configurable):
    """A training-like phase."""

    def __init__(
        self,
        dataloader: DataLoader[S, A, P],
        criterion: Criterion[O, A],
        optimizers: Optimizer | Iterable[Optimizer],
        schedulers: LRScheduler | Iterable[LRScheduler] | None,
        accumulation_steps: int = 1,
        clip_gradient: ClipGradNorm | ClipGradValue | None = None,
        evaluator: Evaluator[P] | None = None,
        run_interval: int | Callable[[State[S, O, A, P]], bool] = 1,
        label: str = "train",
    ) -> None:
        """Initializes a new training-like phase."""
        optimizers = utils.to_tuple(optimizers)
        if len(optimizers) == 0:
            msg = "At least one optimizer is required."
            raise ValueError(msg)

        if accumulation_steps < 1:
            msg = "The number of accumulation steps must be greater than 0."
            raise ValueError(msg)

        schedulers = utils.to_tuple(schedulers) if schedulers is not None else ()

        self._label = label
        self._running = False
        self._timestamp = PhaseTimestamp(dataloader.num_batches, dataloader.num_samples)
        self._dataloader = dataloader
        self._criterion = criterion
        self._optimizers = optimizers
        self._schedulers = schedulers
        self._accumulation_steps = accumulation_steps
        self._clip_gradient = clip_gradient
        self._evaluator = evaluator
        self._run_interval = run_interval

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def label(self) -> str:
        """The label of the phase."""
        return self._label

    @property
    def timestamp(self) -> PhaseTimestamp:
        """The timestamp of the phase."""
        return self._timestamp

    @property
    def dataloader(self) -> DataLoader[S, A, P]:
        """The dataloader of the phase."""
        return self._dataloader

    @property
    def criterion(self) -> Criterion[O, A]:
        """The criterion of the phase."""
        return self._criterion

    @property
    def optimizers(self) -> tuple[Optimizer, ...]:
        """The optimizers of the phase."""
        return self._optimizers

    @property
    def schedulers(self) -> tuple[LRScheduler, ...] | tuple[()]:
        """The schedulers of the phase (if any)."""
        return self._schedulers

    @property
    def accumulation_steps(self) -> int:
        """The number of accumulation steps."""
        return self._accumulation_steps

    @property
    def clip_gradient(self) -> ClipGradNorm | ClipGradValue | None:
        """The gradient clipping technique (if any)."""
        return self._clip_gradient

    @property
    def evaluator(self) -> Evaluator[P] | None:
        """The evaluator of the phase (if any)."""
        return self._evaluator

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def run(self) -> None:
        """Runs the phase."""
        self._running = True

    def stop(self) -> None:
        """Stops the phase."""
        if not self._running:
            msg = "Cannot stop a phase that is not running."
            raise RuntimeError(msg)

        self._timestamp.terminate_epoch()
        self._running = False

    def is_running(self) -> bool:
        """Returns whether the phase is currently running."""
        return self._running

    def should_run(self, state: State[S, O, A, P]) -> bool:
        """Returns whether the phase should run."""
        if isinstance(self._run_interval, int):
            return (state.num_epochs + 1) % self._run_interval == 0
        else:
            return self._run_interval(state)

    def state_dict(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "timestamp": self._timestamp.state_dict(),
            "dataloader": self._dataloader.state_dict(),
            "criterion": self._criterion.state_dict()
            if isinstance(self._criterion, Stateful)
            else None,
            "optimizers": tuple(
                optimizer.state_dict() for optimizer in self._optimizers
            ),
            "schedulers": tuple(
                scheduler.state_dict() if isinstance(scheduler, Stateful) else None
                for scheduler in self._schedulers
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        self._running = state_dict["running"]
        self._timestamp.load_state_dict(state_dict["timestamp"])
        self._dataloader.load_state_dict(state_dict["dataloader"])

        if isinstance(self._criterion, Stateful):
            self._criterion.load_state_dict(state_dict["criterion"])

        for o, o_state in zip(self._optimizers, state_dict["optimizers"], strict=True):
            o.load_state_dict(o_state)

        for s, s_state in zip(self._schedulers, state_dict["schedulers"], strict=True):
            if isinstance(s, Stateful):
                s.load_state_dict(s_state)

    def to(self, device: str | torch.device, *, non_blocking: bool = False) -> Self:
        """Move in-place the phase to the specified device."""
        if isinstance(self._criterion, Moveable):
            self._criterion = self._criterion.to(device, non_blocking=non_blocking)

        self._optimizers = tuple(
            optimizer.to(device, non_blocking=non_blocking)
            if isinstance(optimizer, Moveable)
            else optimizer
            for optimizer in self._optimizers
        )
        self._schedulers = tuple(
            scheduler.to(device, non_blocking=non_blocking)
            if isinstance(scheduler, Moveable)
            else scheduler
            for scheduler in self._schedulers
        )

        if isinstance(self._evaluator, Moveable):
            self._evaluator = self._evaluator.to(device, non_blocking=non_blocking)

        return self

    def get_config(self, recursive: bool) -> dict[str, Any]:
        config: dict[str, Any] = {
            "label": self._label,
            "accumulation_steps": self._accumulation_steps,
        }

        if self._clip_gradient is not None:
            config["clip_gradient"] = utils.get_config(self._clip_gradient, recursive)

        if isinstance(self._run_interval, int):
            config["run_interval"] = self._run_interval

        if recursive:
            config["dataloader"] = self._dataloader.get_config(recursive)
            config["criterion"] = utils.get_config(self._criterion, recursive)
            config["optimizers"] = [
                utils.get_config(optimizer, recursive) for optimizer in self._optimizers
            ]
            config["schedulers"] = [
                utils.get_config(scheduler, recursive) for scheduler in self._schedulers
            ]

            if self._evaluator is not None:
                config["evaluator"] = utils.get_config(self._evaluator, recursive)

        return config


# --------------------------------------------------------------------------- #
# Evaluation Phase
# --------------------------------------------------------------------------- #


class EvaluationPhase[S, O, A, P](Stateful, Configurable):
    """An evaluation-like phase."""

    def __init__(
        self,
        dataloader: DataLoader[S, A, P],
        evaluator: Evaluator[P],
        criterion: Criterion[O, A] | None = None,
        run_interval: int | Callable[[State[S, O, A, P]], bool] = 1,
        label: str = "eval",
    ) -> None:
        self._label = label
        self._running = False
        self._timestamp = PhaseTimestamp(dataloader.num_batches, dataloader.num_samples)
        self._dataloader = dataloader
        self._evaluator = evaluator
        self._criterion = criterion
        self._run_interval = run_interval

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def label(self) -> str:
        """The label of the phase."""
        return self._label

    @property
    def timestamp(self) -> PhaseTimestamp:
        """The timestamp of the phase."""
        return self._timestamp

    @property
    def dataloader(self) -> DataLoader[S, A, P]:
        """The dataloader of the phase."""
        return self._dataloader

    @property
    def evaluator(self) -> Evaluator[P]:
        """The evaluator of the phase."""
        return self._evaluator

    @property
    def criterion(self) -> Criterion[O, A] | None:
        """The criterion of the phase (if any)."""
        return self._criterion

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def run(self) -> None:
        """Runs the phase."""
        self._running = True

    def stop(self) -> None:
        """Stops the phase."""
        if not self._running:
            msg = "Cannot stop a phase that is not running."
            raise RuntimeError(msg)

        self._timestamp.terminate_epoch()
        self._running = False

    def is_running(self) -> bool:
        """Returns whether the phase is currently running."""
        return self._running

    def should_run(self, state: State[S, O, A, P]) -> bool:
        """Returns whether the phase should run."""
        if isinstance(self._run_interval, int):
            return (state.num_epochs + 1) % self._run_interval == 0
        else:
            return self._run_interval(state)

    def state_dict(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "timestamp": self._timestamp.state_dict(),
            "dataloader": self._dataloader.state_dict(),
            "evaluator": self._evaluator.state_dict()
            if isinstance(self._evaluator, Stateful)
            else None,
            "criterion": self._criterion.state_dict()
            if isinstance(self._criterion, Stateful)
            else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        self._running = state_dict["running"]
        self._timestamp.load_state_dict(state_dict["timestamp"])
        self._dataloader.load_state_dict(state_dict["dataloader"])

        if isinstance(self._evaluator, Stateful):
            self._evaluator.load_state_dict(state_dict["evaluator"])

        if isinstance(self._criterion, Stateful):
            self._criterion.load_state_dict(state_dict["criterion"])

    def to(self, device: str | torch.device, *, non_blocking: bool = False) -> Self:
        """Moves in-place the phase to the specified device."""
        if isinstance(self._evaluator, Moveable):
            self._evaluator = self._evaluator.to(device, non_blocking=non_blocking)

        if isinstance(self._criterion, Moveable):
            self._criterion = self._criterion.to(device, non_blocking=non_blocking)

        return self

    def get_config(self, recursive: bool) -> dict[str, Any]:
        config: dict[str, Any] = {"label": self._label}

        if isinstance(self._run_interval, int):
            config["run_interval"] = self._run_interval

        if recursive:
            config["dataloader"] = self._dataloader.get_config(recursive)
            config["evaluator"] = utils.get_config(self._evaluator, recursive)
            if self._criterion is not None:
                config["criterion"] = utils.get_config(self._criterion, recursive)

        return config
