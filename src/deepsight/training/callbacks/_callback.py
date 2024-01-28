# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
# --------------------------------------------------------------------------- #
# Modified from:
# https://github.com/mosaicml/composer/blob/dev/composer/core/callback.py
# --------------------------------------------------------------------------- #

from typing import Protocol

from deepsight.data import Batch
from deepsight.training import BatchLosses, State


class Callback[S, O, A, P](Protocol):
    def on_init(self, state: State[S, O, A, P]) -> None:
        pass

    def on_fit_start(self, state: State[S, O, A, P]) -> None:
        pass

    def on_epoch_start(self, state: State[S, O, A, P]) -> None:
        pass

    def on_phase_start(self, state: State[S, O, A, P]) -> None:
        pass

    def on_step_start(self, state: State[S, O, A, P]) -> None:
        pass

    def on_forward_start(
        self,
        state: State[S, O, A, P],
        samples: Batch[S],
        annotations: Batch[A] | None,
    ) -> None:
        pass

    def on_forward_end(self, state: State[S, O, A, P], output: O) -> None:
        pass

    def on_criterion_start(
        self,
        state: State[S, O, A, P],
        output: O,
        annotations: Batch[A],
    ) -> None:
        pass

    def on_criterion_end(
        self,
        state: State[S, O, A, P],
        losses: BatchLosses,
    ) -> None:
        pass

    def on_step_loss(self, state: State[S, O, A, P], losses: BatchLosses) -> None:
        pass

    def on_optimization_start(self, state: State[S, O, A, P]) -> None:
        pass

    def on_optimization_end(self, state: State[S, O, A, P]) -> None:
        pass

    def on_evaluation_start(self, state: State[S, O, A, P]) -> None:
        pass

    def on_evaluation_end(self, state: State[S, O, A, P]) -> None:
        pass

    def on_step_end(self, state: State[S, O, A, P]) -> None:
        pass

    def on_phase_end(self, state: State[S, O, A, P]) -> None:
        pass

    def on_epoch_end(self, state: State[S, O, A, P]) -> None:
        pass

    def on_fit_end(
        self,
        state: State[S, O, A, P],
        error: Exception | KeyboardInterrupt | None,
    ) -> None:
        pass
