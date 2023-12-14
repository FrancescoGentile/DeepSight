##
##
##

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.cuda.amp import GradScaler

from deepsight import utils
from deepsight.training import Model
from deepsight.typing import Moveable, StateDict, Stateful

from ._misc import Precision
from ._timestamp import EpochPhaseTimestamp, Timestamp

if TYPE_CHECKING:
    from ._phase import EpochPhase
    from .callbacks import Callback


class State[S, O, A, P](Stateful):
    """The state of the training engine."""

    def __init__(
        self,
        run_name: str,
        model: Model[S, O, A, P],
        phases: tuple[EpochPhase[S, O, A, P], ...],
        timestamp: Timestamp,
        device: torch.device,
        precision: Precision,
        scaler: GradScaler,
        callbacks: tuple[Callback[S, O, A, P], ...],
    ) -> None:
        self._run_name = run_name
        self._resumed = False

        self._model = model.to(device, non_blocking=True)
        self._phases = tuple(phase.to(device, non_blocking=True) for phase in phases)
        self._current_phase_idx = 0
        self._timestamp = timestamp

        self._device = device
        self._precision = precision
        self._scaler = scaler

        self._callbacks = tuple(
            callback.to(device, non_blocking=True)
            if isinstance(callback, Moveable)
            else callback
            for callback in callbacks
        )

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def run_name(self) -> str:
        """The name of the run."""
        return self._run_name

    @property
    def resumed(self) -> bool:
        """Whether the training is resumed."""
        return self._resumed

    @resumed.setter
    def resumed(self, resumed: bool) -> None:
        self._resumed = resumed

    @property
    def model(self) -> Model[S, O, A, P]:
        """The model."""
        return self._model

    @property
    def phases(self) -> tuple[EpochPhase[S, O, A, P], ...]:
        """The phases."""
        return self._phases

    @property
    def current_phase_idx(self) -> int:
        """The index of the current phase."""
        return self._current_phase_idx

    @property
    def current_phase(self) -> EpochPhase[S, O, A, P]:
        """The current phase."""
        return self._phases[self._current_phase_idx]

    @property
    def current_phase_timestamp(self) -> EpochPhaseTimestamp:
        """The current phase timestamp."""
        return self.timestamp.phases[self.current_phase_idx]

    @property
    def timestamp(self) -> Timestamp:
        """The timestamp."""
        return self._timestamp

    @property
    def device(self) -> torch.device:
        """The device."""
        return self._device

    @property
    def precision(self) -> Precision:
        """The precision."""
        return self._precision

    @property
    def scaler(self) -> GradScaler:
        """The scaler."""
        return self._scaler

    @property
    def callbacks(self) -> tuple[Callback[S, O, A, P], ...]:
        """The callbacks."""
        return self._callbacks

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def next_epoch(self) -> None:
        """Moves to the next epoch."""
        self._current_phase_idx = 0
        self._timestamp.next_epoch()

    def next_phase(self) -> None:
        """Moves to the next phase."""
        if not self.current_phase_timestamp.has_ended():
            raise RuntimeError(
                "Cannot move to the next phase before the current one has ended."
            )

        self._current_phase_idx += 1

    def state_dict(self) -> StateDict:
        state = {
            "model": self._model.state_dict(),
            "phases": [phase.state_dict() for phase in self._phases],
            "current_phase_idx": self._current_phase_idx,
            "timestamp": self._timestamp.state_dict(),
            "scaler": self._scaler.state_dict(),
            "callbacks": [
                callback.state_dict() if isinstance(callback, Stateful) else None
                for callback in self._callbacks
            ],
        }

        return {
            "rng": utils.get_rng_state(),
            "state": state,
        }

    def load_state_dict(self, state_dict: StateDict) -> Any:
        utils.set_rng_state(state_dict["rng"])

        state_dict = state_dict["state"]

        self._model.load_state_dict(state_dict["model"])
        for phase, phase_state in zip(self._phases, state_dict["phases"], strict=True):
            phase.load_state_dict(phase_state)

        self._current_phase_idx = state_dict["current_phase_idx"]
        self._timestamp.load_state_dict(state_dict["timestamp"])
        self._scaler.load_state_dict(state_dict["scaler"])

        for callback, callback_state in zip(
            self._callbacks, state_dict["callbacks"], strict=True
        ):
            if isinstance(callback, Stateful):
                callback.load_state_dict(callback_state)
