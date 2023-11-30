##
##
##

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Self

import torch
from torch.optim import Optimizer

from deepsight import utils
from deepsight.core import Criterion, Evaluator
from deepsight.training import DataLoader
from deepsight.typing import Configs, Configurable, Moveable, StateDict, Stateful

from ._misc import ClipGradNorm, ClipGradValue

if TYPE_CHECKING:
    from deepsight.training.schedulers import LRScheduler

    from ._state import State

# --------------------------------------------------------------------------- #
# Training Phase
# --------------------------------------------------------------------------- #


class TrainingPhase[S, O, A, P](Stateful, Configurable):
    """A training-like epoch phase."""

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
        optimizers = utils.to_tuple(optimizers)
        if len(optimizers) == 0:
            raise ValueError("At least one optimizer is required.")

        if schedulers is not None:
            schedulers = utils.to_tuple(schedulers)
            if len(schedulers) == 0:
                raise ValueError("At least one scheduler is required.")

        self._label = label
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
    def schedulers(self) -> tuple[LRScheduler, ...] | None:
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

    def should_run(self, state: State[S, O, A, P]) -> bool:
        """Return whether the phase should run."""
        if isinstance(self._run_interval, int):
            return (state.timestamp.num_epochs + 1) % self._run_interval == 0
        else:
            return self._run_interval(state)

    def state_dict(self) -> StateDict:
        return {
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
            )
            if self._schedulers is not None
            else None,
        }

    def load_state_dict(self, state_dict: StateDict) -> Any:
        self._dataloader.load_state_dict(state_dict["dataloader"])

        if isinstance(self._criterion, Stateful):
            self._criterion.load_state_dict(state_dict["criterion"])

        for optimizer, optimizer_state in zip(
            self._optimizers, state_dict["optimizers"], strict=True
        ):
            optimizer.load_state_dict(optimizer_state)

        if self._schedulers is not None:
            for scheduler, scheduler_state in zip(
                self._schedulers, state_dict["schedulers"], strict=True
            ):
                if isinstance(scheduler, Stateful):
                    scheduler.load_state_dict(scheduler_state)

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

        if self._schedulers is not None:
            self._schedulers = tuple(
                scheduler.to(device, non_blocking=non_blocking)
                if isinstance(scheduler, Moveable)
                else scheduler
                for scheduler in self._schedulers
            )

        if isinstance(self._evaluator, Moveable):
            self._evaluator = self._evaluator.to(device, non_blocking=non_blocking)

        return self

    def get_configs(self, recursive: bool) -> Configs:
        configs: Configs = {
            "label": self._label,
            "accumulation_steps": self._accumulation_steps,
        }

        if self._clip_gradient is not None:
            configs["clip_gradient"] = utils.get_configs(self._clip_gradient, recursive)

        if isinstance(self._run_interval, int):
            configs["run_interval"] = self._run_interval

        if recursive:
            configs["dataloader"] = self._dataloader.get_configs(recursive)
            configs["criterion"] = utils.get_configs(self._criterion, recursive)
            configs["optimizers"] = [
                utils.get_configs(optimizer, recursive)
                for optimizer in self._optimizers
            ]
            if self._schedulers is not None:
                configs["schedulers"] = [
                    utils.get_configs(scheduler, recursive)
                    for scheduler in self._schedulers
                ]

            if self._evaluator is not None:
                configs["evaluator"] = utils.get_configs(self._evaluator, recursive)

        return configs


# --------------------------------------------------------------------------- #
# Evaluation Phase
# --------------------------------------------------------------------------- #


class EvaluationPhase[S, O, A, P](Stateful, Configurable):
    """An evaluation-like epoch phase."""

    def __init__(
        self,
        dataloader: DataLoader[S, A, P],
        evaluator: Evaluator[P],
        criterion: Criterion[O, A] | None = None,
        run_interval: int | Callable[[State[S, O, A, P]], bool] = 1,
        label: str = "eval",
    ) -> None:
        self._label = label
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

    def should_run(self, state: State[S, O, A, P]) -> bool:
        """Return whether the phase should run."""
        if isinstance(self._run_interval, int):
            return (state.timestamp.num_epochs + 1) % self._run_interval == 0
        else:
            return self._run_interval(state)

    def state_dict(self) -> StateDict:
        return {
            "dataloader": self._dataloader.state_dict(),
            "evaluator": self._evaluator.state_dict()
            if isinstance(self._evaluator, Stateful)
            else None,
            "criterion": self._criterion.state_dict()
            if isinstance(self._criterion, Stateful)
            else None,
        }

    def load_state_dict(self, state_dict: StateDict) -> Any:
        self._dataloader.load_state_dict(state_dict["dataloader"])

        if isinstance(self._evaluator, Stateful):
            self._evaluator.load_state_dict(state_dict["evaluator"])

        if isinstance(self._criterion, Stateful):
            self._criterion.load_state_dict(state_dict["criterion"])

    def to(self, device: str | torch.device, *, non_blocking: bool = False) -> Self:
        """Move in-place the phase to the specified device."""
        if isinstance(self._evaluator, Moveable):
            self._evaluator = self._evaluator.to(device, non_blocking=non_blocking)

        if isinstance(self._criterion, Moveable):
            self._criterion = self._criterion.to(device, non_blocking=non_blocking)

        return self

    def get_configs(self, recursive: bool) -> Configs:
        configs: Configs = {"label": self._label}

        if isinstance(self._run_interval, int):
            configs["run_interval"] = self._run_interval

        if recursive:
            configs["dataloader"] = self._dataloader.get_configs(recursive)
            configs["evaluator"] = utils.get_configs(self._evaluator, recursive)
            if self._criterion is not None:
                configs["criterion"] = utils.get_configs(self._criterion, recursive)

        return configs


# --------------------------------------------------------------------------- #
# Epoch Phase
# --------------------------------------------------------------------------- #

type EpochPhase[S, O, A, P] = TrainingPhase[S, O, A, P] | EvaluationPhase[S, O, A, P]
