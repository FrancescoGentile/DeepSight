##
##
##

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from torch.optim import Optimizer

from deepsight import utils
from deepsight.core import Criterion, Evaluator
from deepsight.training import DataLoader
from deepsight.training.schedulers import LRScheduler
from deepsight.typing import StateDict, Stateful

from ._misc import ClipGradNorm, ClipGradValue
from ._state import State

# --------------------------------------------------------------------------- #
# Training Phase
# --------------------------------------------------------------------------- #


class TrainingPhase[S, O, A, P](Stateful):
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
        raise NotImplementedError

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


# --------------------------------------------------------------------------- #
# Evaluation Phase
# --------------------------------------------------------------------------- #


class EvaluationPhase[S, O, A, P](Stateful):
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
        raise NotImplementedError

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


# --------------------------------------------------------------------------- #
# Epoch Phase
# --------------------------------------------------------------------------- #

type EpochPhase[S, O, A, P] = TrainingPhase[S, O, A, P] | EvaluationPhase[S, O, A, P]
