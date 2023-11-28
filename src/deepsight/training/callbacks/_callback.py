##
##
##

from typing import Protocol

from deepsight.core import Batch
from deepsight.training.structs import BatchLosses, State


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
