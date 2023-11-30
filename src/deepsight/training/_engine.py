##
##
##

import hashlib
import pickle
import random
from collections import Counter
from collections.abc import Iterable
from typing import Callable

import coolname
import torch
from torch.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler

from deepsight import utils
from deepsight.core import Batch, Model
from deepsight.typing import Detachable

from .callbacks import Callback
from .structs import (
    BatchLosses,
    ClipGradNorm,
    ClipGradValue,
    EpochPhase,
    EvaluationPhase,
    Precision,
    State,
    Timestamp,
    TrainingPhase,
)


class Engine[S, O: Detachable, A, P]:
    def __init__(
        self,
        model: Model[S, O, A, P],
        phases: EpochPhase[S, O, A, P] | Iterable[EpochPhase[S, O, A, P]],
        callbacks: Callback[S, O, A, P] | Iterable[Callback[S, O, A, P]] | None = None,
        device: torch.device | str | None = None,
        precision: Precision = Precision.FP32,
        run_name: str | None = None,
        max_duration: int | Callable[[State[S, O, A, P]], bool] | None = None,
    ) -> None:
        phases = utils.to_tuple(phases)
        if len(phases) == 0:
            raise ValueError("At least one phase must be specified.")
        labels_counter = Counter(phase.label for phase in phases)
        if any(count > 1 for count in labels_counter.values()):
            raise ValueError("Duplicate phase labels are not allowed.")

        callbacks = () if callbacks is None else utils.to_tuple(callbacks)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        if device.type == "cuda" and precision.is_mixed_precision():
            scaler = GradScaler()
        else:
            scaler = GradScaler(enabled=False)

        if run_name is None:
            # If the run name is not specified, we need to generate a random one.
            # A simple approach woul be to call `coolname.generate_slug()`, but
            # if the user sets the same seed, the same name will be generated.
            # We want run with the same engine configurations to have the same name,
            # while different configurations should have different names.
            # So, we hash the engine configurations to generate the new seed to be used
            # to generate the run name.
            configs = {
                "model": utils.get_configs(model, recursive=True),
                "phases": [
                    utils.get_configs(phase, recursive=True) for phase in phases
                ],
            }
            digest = hashlib.sha256(pickle.dumps(configs)).hexdigest()

            random_rng_state = random.getstate()
            random.seed(digest)
            run_name = coolname.generate_slug(2)
            random.setstate(random_rng_state)

        elif len(run_name) == 0:
            raise ValueError("The run name cannot be empty.")

        self._state = State(
            run_name=run_name,
            model=model,
            phases=phases,
            timestamp=Timestamp.start(phases),
            device=device,
            precision=precision,
            scaler=scaler,
            callbacks=callbacks,
        )

        self._max_duration = max_duration

        for callback in self._state.callbacks:
            callback.on_init(self._state)

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def fit(self) -> None:
        for callback in self._state.callbacks:
            callback.on_fit_start(self._state)

        try:
            while not self._should_stop():
                self._execute_epoch()
                self._state.next_epoch()

            for callback in reversed(self._state.callbacks):
                callback.on_fit_end(self._state, None)
        except KeyboardInterrupt as e:
            for callback in reversed(self._state.callbacks):
                callback.on_fit_end(self._state, error=e)
            raise e
        except Exception as e:
            for callback in reversed(self._state.callbacks):
                callback.on_fit_end(self._state, error=e)
            raise e

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _should_stop(self) -> bool:
        match self._max_duration:
            case None:
                return False
            case int():
                return self._state.timestamp.num_epochs >= self._max_duration
            case _:
                return self._max_duration(self._state)

    def _execute_epoch(self) -> None:
        for callback in self._state.callbacks:
            callback.on_epoch_start(self._state)

        while self._state.current_phase_idx < len(self._state.phases):
            if self._state.current_phase.should_run(self._state):
                self._execute_phase()
            self._state.next_phase()

        for callback in reversed(self._state.callbacks):
            callback.on_epoch_end(self._state)

    def _execute_phase(self) -> None:
        phase = self._state.current_phase
        self._state.current_phase_timestamp.start_epoch()

        for callback in self._state.callbacks:
            callback.on_phase_start(self._state)

        match phase:
            case TrainingPhase():
                self._execute_training_phase(phase)
            case EvaluationPhase():
                self._execute_evaluation_phase(phase)

        for callback in reversed(self._state.callbacks):
            callback.on_phase_end(self._state)

        self._state.current_phase_timestamp.end_epoch()
        if phase.evaluator is not None:
            phase.evaluator.reset()

    # ----------------------------------------------------------------------- #
    # Evaluation
    # ----------------------------------------------------------------------- #

    def _execute_evaluation_phase(self, phase: EvaluationPhase[S, O, A, P]) -> None:
        torch.set_grad_enabled(False)
        self._state.model.eval()

        for samples, annotations, golds in phase.dataloader:
            for callback in self._state.callbacks:
                callback.on_step_start(self._state)

            samples = samples.to(self._state.device, non_blocking=True)
            annotations = annotations.to(self._state.device, non_blocking=True)
            golds = golds.to(self._state.device, non_blocking=True)

            with autocast(
                self._state.device.type,
                self._state.precision.to_torch_dtype(),
                enabled=self._state.precision.is_mixed_precision(),
            ):
                for callback in self._state.callbacks:
                    callback.on_forward_start(self._state, samples, None)

                outputs = self._state.model(samples, None)

                for callback in reversed(self._state.callbacks):
                    callback.on_forward_end(self._state, outputs)

                if phase.criterion is not None:
                    for callback in self._state.callbacks:
                        callback.on_criterion_start(self._state, outputs, annotations)

                    losses = phase.criterion.compute(outputs, annotations)
                    losses = BatchLosses(losses, len(samples))

                    for callback in reversed(self._state.callbacks):
                        callback.on_criterion_end(self._state, losses)

                    for callback in self._state.callbacks:
                        callback.on_step_loss(self._state, losses)

            for callback in self._state.callbacks:
                callback.on_evaluation_start(self._state)

            predictions = self._state.model.postprocess(outputs)
            phase.evaluator.update(predictions, golds)

            for callback in reversed(self._state.callbacks):
                callback.on_evaluation_end(self._state)

            for callback in reversed(self._state.callbacks):
                callback.on_step_end(self._state)

            self._state.current_phase_timestamp.next_batch(len(samples))

    # ----------------------------------------------------------------------- #
    # Training
    # ----------------------------------------------------------------------- #

    def _execute_training_phase(self, phase: TrainingPhase[S, O, A, P]) -> None:
        torch.set_grad_enabled(True)
        self._state.model.train()
        for optimizer in phase.optimizers:
            optimizer.zero_grad()

        for samples, annotations, golds in phase.dataloader:
            for callback in self._state.callbacks:
                callback.on_step_start(self._state)

            outputs = self._execute_accumulation_steps(phase, samples, annotations)
            self._execute_optimization_step(phase)
            self._execute_evaluation_step(phase, outputs, golds)

            for callback in reversed(self._state.callbacks):
                callback.on_step_end(self._state)

            self._state.current_phase_timestamp.next_batch(len(samples))

    def _execute_accumulation_steps(
        self,
        phase: TrainingPhase[S, O, A, P],
        samples: Batch[S],
        annotations: Batch[A],
    ) -> list[O]:
        losses: list[BatchLosses] = []
        outputs: list[O] = []
        for acc_samples, acc_annotations in zip(
            samples.split(phase.accumulation_steps),
            annotations.split(phase.accumulation_steps),
            strict=True,
        ):
            acc_output, acc_losses = self._execute_forward_backward(
                phase, acc_samples, acc_annotations
            )
            outputs.append(acc_output)
            losses.append(acc_losses)

        for callback in self._state.callbacks:
            callback.on_step_loss(self._state, BatchLosses.accumulate(losses))

        return outputs

    def _execute_forward_backward(
        self,
        phase: TrainingPhase[S, O, A, P],
        samples: Batch[S],
        annotations: Batch[A],
    ) -> tuple[O, BatchLosses]:
        samples = samples.to(self._state.device, non_blocking=True)
        annotations = annotations.to(self._state.device, non_blocking=True)
        with autocast(
            self._state.device.type,
            self._state.precision.to_torch_dtype(),
            enabled=self._state.precision.is_mixed_precision(),
        ):
            for callback in self._state.callbacks:
                callback.on_forward_start(self._state, samples, annotations)

            output = self._state.model(samples, annotations)

            for callback in reversed(self._state.callbacks):
                callback.on_forward_end(self._state, output)

            for callback in self._state.callbacks:
                callback.on_criterion_start(self._state, output, annotations)

            losses = phase.criterion.compute(output, annotations)
            losses = BatchLosses(losses, len(samples))

            for callback in reversed(self._state.callbacks):
                callback.on_criterion_end(self._state, losses)

            total_acc_loss = torch.as_tensor(sum(losses.values()))
            total_acc_loss = total_acc_loss / phase.accumulation_steps
            self._state.scaler.scale(total_acc_loss).backward()

        # Detach the losses and the output to avoid keeping in memory the associated
        # computation graph across accumulation steps.
        for name, value in losses.items():
            losses[name] = value.detach()

        output = output.detach()

        return output, losses

    def _execute_optimization_step(self, phase: TrainingPhase[S, O, A, P]) -> None:
        for callback in self._state.callbacks:
            callback.on_optimization_start(self._state)

        match phase.clip_gradient:
            case ClipGradValue(value):
                for optimizer in phase.optimizers:
                    self._state.scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_value_(self._state.model.parameters(), value)
            case ClipGradNorm(norm):
                for optimizer in phase.optimizers:
                    self._state.scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(self._state.model.parameters(), norm)
            case None:
                pass

        if phase.schedulers is not None:
            for scheduler in phase.schedulers:
                scheduler.step(self._state.current_phase_timestamp)

        for optimizer in phase.optimizers:
            self._state.scaler.step(optimizer)

        scale = self._state.scaler.get_scale()
        self._state.scaler.update()

        if scale > self._state.scaler.get_scale():
            # The scale factor has been decreased, which means that
            # there was an overflow. Since no parameter update was
            # performed, we do not call `scheduler.step()`.
            utils.get_library_logger().warning("Gradient overflow detected.")

        for callback in reversed(self._state.callbacks):
            callback.on_optimization_end(self._state)

        for optimizer in phase.optimizers:
            optimizer.zero_grad()

    def _execute_evaluation_step(
        self,
        phase: TrainingPhase[S, O, A, P],
        outputs: list[O],
        golds: Batch[P],
    ) -> None:
        if phase.evaluator is not None:
            for callback in self._state.callbacks:
                callback.on_evaluation_start(self._state)

            golds = golds.to(self._state.device, non_blocking=True)

            with torch.no_grad():
                predictions = Batch.concat(
                    [self._state.model.postprocess(output) for output in outputs]
                )

                phase.evaluator.update(predictions, golds)

            for callback in reversed(self._state.callbacks):
                callback.on_evaluation_end(self._state)
