# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/mosaicml/composer/blob/dev/composer/core/state.py
# --------------------------------------------------------------------------- #

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.cuda.amp import GradScaler

from deepsight import utils
from deepsight.models import Model
from deepsight.typing import Moveable, Stateful

from ._misc import Precision

if TYPE_CHECKING:
    from ._phase import Phase
    from .callbacks import Callback


class State[S, O, A, P](Stateful):
    """The state of the training engine."""

    def __init__(
        self,
        run_name: str,
        model: Model[S, O, A, P],
        phases: tuple[Phase[S, O, A, P], ...],
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
        self._num_epochs = 0

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
    def phases(self) -> tuple[Phase[S, O, A, P], ...]:
        """The phases."""
        return self._phases

    @property
    def current_phase_idx(self) -> int:
        """The index of the current phase."""
        return self._current_phase_idx

    @property
    def current_phase(self) -> Phase[S, O, A, P]:
        """The current phase."""
        return self._phases[self._current_phase_idx]

    @property
    def num_epochs(self) -> int:
        """The number of epochs that have been run.

        This can also be interpreted as the index of the epoch currently being run.
        Remember that epochs are 0-indexed. For example, if `num_epochs` is `5`, then
        this means that 5 epochs have been fully run, and the 6th epoch (i.e. the epoch
        with index 5) is currently being run.
        """
        return self._num_epochs

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
        if any(phase.is_running() for phase in self._phases):
            msg = "Cannot move to the next epoch while any phase is running."
            raise RuntimeError(msg)

        self._current_phase_idx = 0

    def next_phase(self) -> None:
        """Moves to the next phase."""
        if self.current_phase.is_running():
            msg = "Cannot move to the next phase while the current phase is running."
            raise RuntimeError(msg)

        self._current_phase_idx += 1
        self._phases[self._current_phase_idx].run()

    def state_dict(self) -> dict[str, Any]:
        state = {
            "model": self._model.state_dict(),
            "phases": [phase.state_dict() for phase in self._phases],
            "current_phase_idx": self._current_phase_idx,
            "num_epochs": self._num_epochs,
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

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        utils.set_rng_state(state_dict["rng"])

        state_dict = state_dict["state"]

        self._model.load_state_dict(state_dict["model"])
        for phase, phase_state in zip(self._phases, state_dict["phases"], strict=True):
            phase.load_state_dict(phase_state)

        self._current_phase_idx = state_dict["current_phase_idx"]
        self._num_epochs = state_dict["num_epochs"]
        self._scaler.load_state_dict(state_dict["scaler"])

        for c, c_state in zip(self._callbacks, state_dict["callbacks"], strict=True):
            if isinstance(c, Stateful):
                c.load_state_dict(c_state)
