# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any

import wandb

from deepsight import utils
from deepsight.training import (
    BatchLosses,
    EvaluationPhase,
    State,
    TrainingPhase,
)
from deepsight.training.callbacks import Callback
from deepsight.typing import Stateful


class WandbLogger[S, O, A, P](Callback[S, O, A, P], Stateful):
    def __init__(
        self,
        name: str | None = "{run_name}",
        project: str | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        log_phases: str | Iterable[str] | None = None,
        log_every_n_steps: int = 50,
    ) -> None:
        super().__init__()
        if log_phases is not None:
            log_phases = utils.to_tuple(log_phases)
            if len(log_phases) == 0:
                raise ValueError("At least one phase must be specified.")
        else:
            log_phases = ()

        self._name = name
        self._project = project
        self._entity = entity
        self._tags = tags
        self._notes = notes
        self._log_phases = set(log_phases)
        self._log_every_n_steps = log_every_n_steps

        self._id: str | None = None

        self._total_batch_size: dict[str, int]
        self._losses: dict[str, dict[str, float]]

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def on_init(self, state: State[S, O, A, P]) -> None:
        if self._name is not None:
            self._name = self._name.format(run_name=state.run_name)
        if len(self._log_phases) == 0:
            self._log_phases = {phase.label for phase in state.phases}

        all_labels = {phase.label for phase in state.phases}
        for label in self._log_phases:
            if label not in all_labels:
                raise ValueError(f"Cannot log phase '{label}' as it does not exist.")

        self._total_batch_size = {phase.label: 0 for phase in state.phases}
        self._losses = {phase.label: {} for phase in state.phases}

        wandb.init(
            name=self._name,
            project=self._project,
            entity=self._entity,
            tags=self._tags,
            notes=self._notes,
            config={
                "model": utils.get_config(state.model, recursive=True),
                "phases": [
                    utils.get_config(phase, recursive=True) for phase in state.phases
                ],
            },
            resume=False if self._id is None else "must",
            id=self._id,
        )
        assert wandb.run is not None
        self._id = wandb.run.id

        self._define_plots(state)

    def on_epoch_start(self, state: State[S, O, A, P]) -> None:
        wandb.log({"epoch": state.timestamp.num_epochs + 1})

    def on_step_loss(self, state: State[S, O, A, P], losses: BatchLosses) -> None:
        label = state.current_phase.label
        if label not in self._log_phases:
            return

        self._total_batch_size[label] += losses.batch_size
        if len(self._losses[label]) == 0:
            for name, value in losses.items():
                self._losses[label][name] = value.item() * losses.batch_size
        else:
            for name, value in losses.items():
                if name not in self._losses[label]:
                    raise RuntimeError("Losses must be the same for all batches.")

                self._losses[label][name] += value.item() * losses.batch_size

    def on_step_end(self, state: State[S, O, A, P]) -> None:
        label = state.current_phase.label
        step = state.current_phase_timestamp.num_batches + 1
        if label not in self._log_phases:
            return
        if step % self._log_every_n_steps != 0:
            return
        if isinstance(state.current_phase, EvaluationPhase):
            return

        total_loss = 0.0
        commit_info = {}
        commit_info[f"{label}/step"] = step

        for name, value in self._losses[label].items():
            value /= self._total_batch_size[label]
            commit_info[f"{label}/losses/{name}"] = value
            total_loss += value

        for optim_idx, optimizer in enumerate(state.current_phase.optimizers):
            for group_idx, group in enumerate(optimizer.param_groups):
                key = f"{label}/learning_rates/optimizer_{optim_idx}.param_group_{group_idx}"  # noqa: E501
                commit_info[key] = group["lr"]

        commit_info[f"{label}/losses/total"] = total_loss
        wandb.log(commit_info)

        self._total_batch_size[label] = 0
        self._losses[label] = {}

    def on_phase_end(self, state: State[S, O, A, P]) -> None:
        label = state.current_phase.label
        if label not in self._log_phases:
            return

        commit_info = {}
        if isinstance(state.current_phase, TrainingPhase):
            if state.current_phase.evaluator is None:
                return

            evaluator = state.current_phase.evaluator
            for name, value in evaluator.compute_numeric_metrics().items():
                commit_info[f"{label}/metrics/{name}"] = value
        else:
            if len(self._losses[label]) > 0:
                total_loss = 0.0
                for name, value in self._losses[label].items():
                    value /= self._total_batch_size[label]
                    total_loss += value
                    commit_info[f"{label}/losses/{name}"] = value

                commit_info[f"{label}/losses/total"] = total_loss
                self._total_batch_size[label] = 0
                self._losses[label] = {}

            evaluator = state.current_phase.evaluator
            for name, value in evaluator.compute_numeric_metrics().items():
                commit_info[f"{label}/metrics/{name}"] = value

        wandb.log(commit_info)

    def on_run_end(
        self,
        state: State[S, O, A, P],
        error: Exception | KeyboardInterrupt | None,
    ) -> None:
        if isinstance(error, Exception):
            # the training was interrupted by an error
            wandb.alert(
                title="Training Error",
                text=str(error),
                level=wandb.AlertLevel.ERROR,
            )
            wandb.finish(exit_code=1)
        else:
            # The training was completed successfully or interrupted by the user.
            # Here we consider both cases as successful. The user interruption
            # can be seen as setting the max_duration of the training at runtime.
            wandb.finish(exit_code=0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "id": self._id,
            "total_batch_size": self._total_batch_size,
            "losses": self._losses,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        self._id = state_dict["id"]
        self._total_batch_size = state_dict["total_batch_size"]
        self._losses = state_dict["losses"]

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _define_plots(self, state: State[S, O, A, P]) -> None:
        wandb.define_metric("epoch", summary="max")

        for phase in state.phases:
            if phase.label not in self._log_phases:
                continue

            if isinstance(phase, TrainingPhase):
                wandb.define_metric(f"{phase.label}/step", summary="max")
                for loss in phase.criterion.get_losses_info():
                    wandb.define_metric(
                        f"{phase.label}/losses/{loss.name}",
                        summary="min",
                        step_metric=f"{phase.label}/step",
                    )
                wandb.define_metric(
                    f"{phase.label}/losses/total",
                    summary="min",
                    step_metric=f"{phase.label}/step",
                )

                # log learning rate
                for optim_idx, optimizer in enumerate(phase.optimizers):
                    for group_idx in range(len(optimizer.param_groups)):
                        wandb.define_metric(
                            f"{phase.label}/learning_rates/optimizer_{optim_idx}.param_group_{group_idx}",
                            step_metric=f"{phase.label}/step",
                        )

                if phase.evaluator is not None:
                    for metric in phase.evaluator.get_metrics_info():
                        wandb.define_metric(
                            f"{phase.label}/metrics/{metric.name}",
                            summary="max" if metric.higher_is_better else "min",
                            step_metric="epoch",
                        )
            else:
                if phase.criterion is not None:
                    for loss in phase.criterion.get_losses_info():
                        wandb.define_metric(
                            f"{phase.label}/losses/{loss.name}",
                            summary="min",
                            step_metric="epoch",
                        )
                    wandb.define_metric(
                        f"{phase.label}/losses/total",
                        summary="min",
                        step_metric="epoch",
                    )

                for metric in phase.evaluator.get_metrics_info():
                    wandb.define_metric(
                        f"{phase.label}/metrics/{metric.name}",
                        summary="max" if metric.higher_is_better else "min",
                        step_metric="epoch",
                    )
