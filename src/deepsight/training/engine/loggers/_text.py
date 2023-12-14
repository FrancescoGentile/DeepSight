##
##
##

import logging
from pathlib import Path
from typing import Any

from deepsight import utils
from deepsight.training.engine import BatchLosses, EpochPhase, State, TrainingPhase
from deepsight.training.engine.callbacks import Callback
from deepsight.typing import StateDict, Stateful


class TextLogger[S, O, A, P](Callback[S, O, A, P], Stateful):
    def __init__(
        self,
        level: int = logging.INFO,
        output_file: str | None = "output/{run_name}/training.log",
        console: bool = True,
        capture_library_logs: bool = True,
    ) -> None:
        super().__init__()

        if not console and output_file is None:
            raise ValueError("At least one of console or output_file must be True.")

        self._level = level
        self._output_file = output_file
        self._capture_library_logs = capture_library_logs

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(level)

        if console:
            handler = logging.StreamHandler()
            handler.setFormatter(_get_formatter())
            self._logger.addHandler(handler)

            if capture_library_logs:
                utils.get_library_logger().addHandler(handler)

        self._losses: dict[str, float] | None = None

    def on_init(self, state: State[S, O, A, P]) -> None:
        log_start = self._logger.hasHandlers()
        if self._output_file is not None:
            self._output_file = self._output_file.format(run_name=state.run_name)
            if Path(self._output_file).exists():
                if not state.resumed:
                    raise FileExistsError(
                        f"The output file '{self._output_file}' already exists."
                    )
                handler = logging.FileHandler(self._output_file, mode="a")
            else:
                Path(self._output_file).parent.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(self._output_file, mode="w+")
                log_start = True

            handler.setLevel(self._level)
            handler.setFormatter(_get_formatter())
            self._logger.addHandler(handler)

            if self._capture_library_logs:
                utils.get_library_logger().addHandler(handler)

        if log_start:
            self._log_start(state)

    def on_fit_start(self, state: State[S, O, A, P]) -> None:
        self._logger.info("Starting training.")

    def on_epoch_start(self, state: State[S, O, A, P]) -> None:
        self._logger.info("Epoch %d started.", state.timestamp.num_epochs + 1)

    def on_phase_start(self, state: State[S, O, A, P]) -> None:
        self._logger.info("Phase '%s' started.", state.current_phase.label)
        if state.current_phase.criterion is not None:
            self._losses = {
                loss.name: 0.0
                for loss in state.current_phase.criterion.get_losses_info()
            }
        else:
            self._losses = None

    def on_step_loss(self, state: State[S, O, A, P], losses: BatchLosses) -> None:
        if self._losses is None:
            # this should never happen
            return

        for name, value in losses.items():
            self._losses[name] += value.item() * losses.batch_size

    def on_phase_end(self, state: State[S, O, A, P]) -> None:
        phase = state.current_phase
        self._logger.info("Phase '%s' finished.", phase.label)

        if self._losses is not None:
            self._logger.info("Losses:")
            total_loss = 0.0
            for name, value in self._losses.items():
                value /= phase.dataloader.num_samples
                self._logger.info("\t%s: %f", name, value)
                total_loss += value

            self._logger.info("\tTotal: %f", total_loss)
            self._losses = None

        if phase.evaluator is not None:
            self._logger.info("Metrics:")
            for name, value in phase.evaluator.compute_numeric_metrics().items():
                self._logger.info("\t%s: %f", name, value)

    def on_epoch_end(self, state: State[S, O, A, P]) -> None:
        self._logger.info("Epoch %d finished.", state.timestamp.num_epochs + 1)

    def on_fit_end(
        self,
        state: State[S, O, A, P],
        error: Exception | KeyboardInterrupt | None,
    ) -> None:
        if error is None:
            self._logger.info("Training finished successfully.")
        elif isinstance(error, KeyboardInterrupt):
            self._logger.info("Training interrupted.")
        else:
            self._logger.error("Training failed.")
            self._logger.exception(error)

        # Close the handlers
        if self._capture_library_logs:
            for handler in self._logger.handlers:
                utils.get_library_logger().removeHandler(handler)

        for handler in self._logger.handlers:
            handler.close()
            self._logger.removeHandler(handler)

    def state_dict(self) -> StateDict:
        return {"losses": self._losses}

    def load_state_dict(self, state_dict: StateDict) -> Any:
        self._losses = state_dict["losses"]

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _log_start(self, state: State[S, O, A, P]) -> None:
        # Log the state properties
        self._logger.info("Run name: %s", state.run_name)
        self._logger.info("Using device: %s", state.device)
        self._logger.info("Using precision: %s", state.precision)

        # Log the model
        self._logger.info("Model:")
        self._logger.info(state.model)
        num_params = sum(p.numel() for p in state.model.parameters() if p.requires_grad)
        self._logger.info("Number of trainable parameters: %d", num_params)

        # Log the phases
        self._logger.info("Phases:")
        for phase in state.phases:
            self._log_phase(phase)

    def _log_phase(self, phase: EpochPhase[S, O, A, P]) -> None:
        self._logger.info("\tName: %s", phase.label)

        # Log the dataloader
        self._logger.info("\tDataset: %s", phase.dataloader.dataset)
        self._logger.info("\tNumber of batches: %d", phase.dataloader.num_batches)
        self._logger.info("\tNumber of samples: %d", phase.dataloader.num_samples)

        if isinstance(phase, TrainingPhase):
            self._logger.info("\tOptimizers:")
            for optimizer in phase.optimizers:
                self._logger.info("\t\t%s", optimizer)

            if phase.schedulers is not None:
                self._logger.info("\tSchedulers:")
                for scheduler in phase.schedulers:
                    self._logger.info("\t\t%s", scheduler)

            self._logger.info("\tClip gradient: %s", phase.clip_gradient)


# --------------------------------------------------------------------------- #
# Private Functions
# --------------------------------------------------------------------------- #


def _get_formatter() -> logging.Formatter:
    return logging.Formatter(
        "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
