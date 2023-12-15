##
##
##

"""Train GAHOI on Human-Object Interaction datasets."""

from pathlib import Path

import deepsight.transforms.vision as TV  # noqa: N812
import torch
from deepsight import utils
from deepsight.training import DataLoader
from deepsight.training.engine import (
    ClipGradNorm,
    Engine,
    EvaluationPhase,
    Precision,
    State,
    TrainingPhase,
)
from deepsight.training.engine.callbacks import Callback
from deepsight.training.engine.loggers import ProgressBarLogger, TextLogger, WandbLogger
from deepsight.training.hoic import Annotations, Evaluator, Predictions, Sample
from deepsight.training.hoic.datasets import H2ODataset, HICODETDataset
from deepsight.training.hoic.models import gahoi
from deepsight.training.schedulers import LinearLR, ReciprocalLR
from deepsight.typing import Stateful
from torch.optim import AdamW

# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #


class CooldownCallback[S, O, A, P](Callback[S, O, A, P]):
    """Callback used to handle cooldown pahses before evaluation.

    An approximate way to evaluate the model during training like if it were evaluated
    at the end of the whole training process is to use a cooldown phase. During this
    phase the model is trained like in a training phase but the learning rate is
    linearly decreased to zero. After the cooldown phase, the model is evaluated and
    then the training resumes like normal (like if the cooldown phase never happened).

    To ensure that the training resumes like normal, this callback is used to save the
    training state before the cooldown phase and then load it after the cooldown phase.

    !!! warning

        This callback assumes that the cooldown phase is immediately followed by an
        evaluation phase that should be the last phase of the epoch.
    """

    def __init__(
        self,
        path: str = "output/{run_name}/cooldown.pt",
        label: str = "cooldown",
    ) -> None:
        """Initialize the callback.

        Args:
            path: The path where to save the training state before the cooldown phase.
            label: The label of the cooldown phase.
        """
        super().__init__()

        self._label = label
        self._path = path

    def on_init(self, state: State[S, O, A, P]) -> None:  # noqa: D102
        self._path = self._path.format(run_name=state.run_name)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

    def on_phase_start(self, state: State[S, O, A, P]) -> None:  # noqa: D102
        if state.current_phase.label != self._label:
            return

        pre_state = {}
        pre_state["model"] = state.model.state_dict()
        pre_state["scaler"] = state.scaler.state_dict()
        pre_state["callbacks"] = [
            callback.state_dict() if isinstance(callback, Stateful) else None
            for callback in state.callbacks
        ]

        for phase in state.phases:
            if isinstance(phase, EvaluationPhase):
                continue
            if phase.label == self._label:
                continue

            if isinstance(phase.criterion, Stateful):
                pre_state["criterion"] = phase.criterion.state_dict()

            pre_state["optimizers"] = [
                optimizer.state_dict() for optimizer in phase.optimizers
            ]

            if phase.schedulers is not None:
                pre_state["schedulers"] = [
                    scheduler.state_dict() if isinstance(scheduler, Stateful) else None
                    for scheduler in phase.schedulers
                ]

            if isinstance(phase.evaluator, Stateful):
                pre_state["evaluator"] = phase.evaluator.state_dict()

        torch.save(pre_state, self._path)

    def on_epoch_end(self, state: State[S, O, A, P]) -> None:  # noqa: D102
        if not Path(self._path).exists():
            return

        pre_state = torch.load(self._path)

        state.model.load_state_dict(pre_state["model"])
        state.scaler.load_state_dict(pre_state["scaler"])
        for callback, pre_callback_state in zip(
            state.callbacks, pre_state["callbacks"], strict=True
        ):
            if isinstance(callback, Stateful):
                callback.load_state_dict(pre_callback_state)

        for phase in state.phases:
            if isinstance(phase, EvaluationPhase):
                continue
            if phase.label == self._label:
                continue

            if isinstance(phase.criterion, Stateful):
                phase.criterion.load_state_dict(pre_state["criterion"])

            for optimizer, pre_optimizer_state in zip(
                phase.optimizers, pre_state["optimizers"], strict=True
            ):
                optimizer.load_state_dict(pre_optimizer_state)

            if phase.schedulers is not None:
                for scheduler, pre_scheduler_state in zip(
                    phase.schedulers, pre_state["schedulers"], strict=True
                ):
                    if isinstance(scheduler, Stateful):
                        scheduler.load_state_dict(pre_scheduler_state)

            if isinstance(phase.evaluator, Stateful):
                phase.evaluator.load_state_dict(pre_state["evaluator"])

        Path(self._path).unlink()


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def create_training_phase(
    dataset: HICODETDataset | H2ODataset,
    model: gahoi.Model,
    criterion: gahoi.Criterion,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    lr: float,
    warmup_epochs: int,
) -> TrainingPhase[Sample, gahoi.Output, Annotations, Predictions]:
    """Create the training phase."""
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    phase = TrainingPhase(
        label="train",
        dataloader=dataloader,
        criterion=criterion,
        optimizers=[optimizer],
        schedulers=[
            ReciprocalLR(optimizer, lr, warmup_epochs * dataloader.num_batches)
        ],
        accumulation_steps=1,
        clip_gradient=ClipGradNorm(0.1),
        run_interval=1,
    )

    return phase


def create_cooldown_phase(
    dataset: HICODETDataset | H2ODataset,
    model: gahoi.Model,
    criterion: gahoi.Criterion,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    run_interval: int = 5,
) -> TrainingPhase[Sample, gahoi.Output, Annotations, Predictions]:
    """Create the cooldown phase."""
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    phase = TrainingPhase(
        label="cooldown",
        dataloader=dataloader,
        criterion=criterion,
        optimizers=[optimizer],
        schedulers=[LinearLR(optimizer, dataloader.num_batches)],
        clip_gradient=ClipGradNorm(0.1),
        accumulation_steps=1,
        run_interval=run_interval,
    )

    return phase


def create_test_phase(
    dataset: HICODETDataset | H2ODataset,
    criterion: gahoi.Criterion,
    batch_size: int = 64,
    run_interval: int = 5,
) -> EvaluationPhase[Sample, gahoi.Output, Annotations, Predictions]:
    """Create the test phase."""
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    evaluator = Evaluator(
        dataset.num_interaction_classes,
        average="macro",
        thresholds=11,
    )

    phase = EvaluationPhase(
        dataloader=dataloader,
        evaluator=evaluator,
        criterion=criterion,
        run_interval=run_interval,
        label="test",
    )

    return phase


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #


def main() -> None:
    """Train GAHOI on Human-Object Interaction datasets."""
    utils.seed_all(3407)

    t_transform = TV.SequentialOrder([
        TV.RandomApply(TV.HorizonalFlip(), 0.5),
        TV.ColorJitter(0.4, 0.4, 0.4),
        TV.Resize((384, 384)),
        TV.ToDtype(torch.float32, scale=True),
        TV.Standardize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    e_transform = TV.SequentialOrder([
        TV.Resize((384, 384)),
        TV.ToDtype(torch.float32, scale=True),
        TV.Standardize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    t_dataset = H2ODataset("datasets/h2o", "train", transform=t_transform)
    e_dataset = H2ODataset("datasets/h2o", "test", transform=e_transform)

    config = gahoi.Configs(
        human_class_id=t_dataset.human_class_id,
        num_entity_classes=t_dataset.num_entity_classes,
        num_interaction_classes=t_dataset.num_interaction_classes,
        allow_human_human=True,
        num_decoder_layers=1,
        patch_dropout=0.7,
        qkv_dropout=0.7,
        attn_dropout=0.7,
        proj_dropout=0.7,
        ffn_dropout=0.7,
        classifier_dropout=0.7,
    )
    model = gahoi.Model(
        config,
        t_dataset.get_object_valid_interactions(["train", "test"]),
    )
    model.encoder.load_state_dict(torch.load("weights/vit_base_patch32_384.pt"))
    model.encoder.requires_grad_(False)

    criterion = gahoi.Criterion(
        layer_indices=range(config.num_decoder_layers),
        suppression_alpha=0.7,
        suppression_gamma=2.0,
        classification_alpha=0.7,
        classification_gamma=2.0,
    )

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    train_phase = create_training_phase(
        t_dataset,
        model,
        criterion,
        batch_size=64,
        optimizer=optimizer,
        lr=1e-4,
        warmup_epochs=5,
    )

    # If the following code is uncommented, then uncomment the cooldown_phase
    # and CooldownCallback as well.

    # cooldown_phase = create_cooldown_phase(
    #     t_dataset,
    #     model,
    #     criterion,
    #     batch_size=64,
    #     optimizer=optimizer,
    # )

    test_phase = create_test_phase(
        e_dataset,
        criterion,
        batch_size=64,
    )

    engine = Engine(
        model=model,
        phases=[
            train_phase,
            # cooldown_phase,
            test_phase,
        ],
        callbacks=[
            TextLogger(),
            ProgressBarLogger(),
            WandbLogger(project="H2O", log_every_n_steps=10),
            # CooldownCallback("output/{run_name}/cooldown.pt"),
        ],
        precision=Precision.AMP_FP16,
        device="cuda",
    )

    engine.fit()


if __name__ == "__main__":
    main()
