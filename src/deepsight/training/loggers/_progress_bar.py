##
##
##

from tqdm import tqdm

from deepsight.core import Batch
from deepsight.training.callbacks import Callback
from deepsight.training.structs import State


class ProgressBarLogger[S, O, A, P](Callback[S, O, A, P]):
    def on_phase_start(self, state: State[S, O, A, P]) -> None:
        phase = state.current_phase
        assert phase is not None
        self._pbar = tqdm(
            total=phase.dataloader.num_samples(),
            desc=f"{phase.label}",
        )

    def on_step_start(self, state: State[S, O, A, P]) -> None:
        self._num_samples = 0

    def on_forward_start(
        self,
        state: State[S, O, A, P],
        samples: Batch[S],
        annotations: Batch[A] | None,
    ) -> None:
        self._num_samples += len(samples)

    def on_step_end(self, state: State[S, O, A, P]) -> None:
        self._pbar.update(self._num_samples)

    def on_phase_end(self, state: State[S, O, A, P]) -> None:
        self._pbar.close()
