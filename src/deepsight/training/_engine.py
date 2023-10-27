##
##
##

import json
import logging
import math
import random
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Generic, TypeVar

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from deepsight.models import Criterion, DeepSightModel
from deepsight.tasks import Dataset, Evaluator, MetricType
from deepsight.typing import Configurable, JSONPrimitive, Moveable, Stateful

from ._dataloader import DataLoader
from .schedulers import ReciprocalLR

S = TypeVar("S", bound=Moveable)
O = TypeVar("O")  # noqa
A = TypeVar("A", bound=Moveable)
P = TypeVar("P", bound=Moveable)


class Engine(Stateful, Generic[S, O, A, P]):
    def __init__(
        self,
        train_dataset: Dataset[S, A, P],
        eval_dataset: Dataset[S, A, P],
        model: DeepSightModel[S, O, A, P],
        criterion: Criterion[O, A],
        evaluator: Evaluator[P],
        optimizer: Optimizer,
        scheduler: LRScheduler,
        step_after_batch: bool,
        train_batch_size: int,
        accumulation_steps: int,
        eval_batch_size: int,
        metric_to_optimize: str,
        max_epochs: int = -1,
        patience: int = 3,
        check_val_every_n_epochs: int = 1,
        log_every_n_steps: int = 50,
        keep_checkpoints: int = 1,
        max_grad_norm: float | None = None,
        init_scale: float | None = None,
        device: torch.device | None = None,
        precision: torch.dtype = torch.float32,
        output_dir: Path | str = "./output",
        wandb_project: str | None = None,
    ) -> None:
        """Initialize the training engine."""
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path(output_dir) / now
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.logger = _get_logger(self.output_dir / "train.log")
        self.logger.info(f"Output directory: {self.output_dir}.")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.logger.info(f"Using device: {device}.")

        self.precision = precision
        self.logger.info(f"Using precision: {precision}.")

        self.metric_to_optimize = metric_to_optimize
        for metric_name, metric_type in evaluator.metrics:
            if metric_name == metric_to_optimize:
                if metric_type != MetricType.NUMERIC:
                    raise ValueError(
                        f"Metric '{metric_name}' is not numeric and cannot be used "
                        "for optimization."
                    )
                break
        else:
            raise ValueError(
                f"Metric '{metric_to_optimize}' is not defined in the evaluator."
            )

        self.max_epochs = max_epochs
        self.patience = patience
        self.check_val_every_n_epochs = check_val_every_n_epochs
        self.log_every_n_steps = log_every_n_steps
        self.max_grad_norm = max_grad_norm
        self.keep_checkpoints = keep_checkpoints

        # Setup datasets
        self._setup_loaders(
            train_dataset,
            eval_dataset,
            train_batch_size,
            eval_batch_size,
            accumulation_steps,
        )

        # Setup model
        self._setup_model(model)
        if isinstance(criterion, Moveable):
            criterion = criterion.move(self.device, non_blocking=True)
        self.criterion = criterion

        if isinstance(evaluator, Moveable):
            evaluator = evaluator.move(self.device, non_blocking=True)
        self.evaluator = evaluator

        self.optimizer = optimizer
        self.logger.info(f"Using optimizer: {optimizer}.")
        self.scheduler = scheduler
        self.logger.info(f"Using scheduler: {scheduler}.")
        self.step_after_batch = step_after_batch

        self._setup_scaler(init_scale)

        configs = self._get_configs()
        with open(self.output_dir / "configs.json", "w") as f:
            json.dump(configs, f, indent=2)
        self._setup_wandb(wandb_project, configs)

    # ----------------------------------------------------------------------- #
    # Public static methods
    # ----------------------------------------------------------------------- #

    @staticmethod
    def setup_libraries(seed: int, deterministic: bool = False) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_default_dtype(torch.float32)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(deterministic)

    # ----------------------------------------------------------------------- #
    # Public methods
    # ----------------------------------------------------------------------- #

    def fit(self) -> None:
        """Starts the training loop."""
        try:
            self._run()
        except KeyboardInterrupt:
            self.logger.info("Training interrupted.")
            wandb.finish()
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}.")
            raise e

    def get_state(self) -> dict[str, Any]:
        state = {
            "model": self.model.get_state(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        if isinstance(self.criterion, Stateful):
            state["criterion"] = self.criterion.get_state()
        if isinstance(self.evaluator, Stateful):
            state["evaluator"] = self.evaluator.get_state()

        return state

    def set_state(self, state: dict[str, Any]) -> None:
        self.model.set_state(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])

        if isinstance(self.criterion, Stateful):
            self.criterion.set_state(state["criterion"])
        if isinstance(self.evaluator, Stateful):
            self.evaluator.set_state(state["evaluator"])

    # ----------------------------------------------------------------------- #
    # Private methods
    # ----------------------------------------------------------------------- #

    def _setup_loaders(
        self,
        train_dataset: Dataset[S, A, P],
        eval_dataset: Dataset[S, A, P],
        train_batch_size: int,
        eval_batch_size: int,
        accumulation_steps: int,
    ) -> None:
        self.accumulation_steps = accumulation_steps
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )

        self.logger.info(f"Using train dataset: {train_dataset}.")
        self.logger.info(f"\tbatch size: {train_batch_size}.")
        self.logger.info(f"Using eval dataset: {eval_dataset}.")

    def _setup_model(self, model: DeepSightModel[S, O, A, P]) -> None:
        model = model.move(self.device, non_blocking=True)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.model = model
        self.logger.info("Using model:")
        self.logger.info(model)
        self.logger.info(f"\ttotal number of parameters: {num_params}.")

    def _setup_scaler(self, init_scale: float | None) -> None:
        enabled = self.device.type == "cuda"
        enabled &= self.precision == torch.float16 or self.precision == torch.bfloat16
        if init_scale is not None:
            self.scaler = GradScaler(init_scale=init_scale, enabled=enabled)
        else:
            self.scaler = GradScaler(enabled=enabled)

    def _get_configs(self) -> dict[str, JSONPrimitive]:
        configs = {}
        objects = [
            ("train_dataset", self.train_loader.dataset),
            ("eval_dataset", self.eval_loader.dataset),
            ("model", self.model),
            ("criterion", self.criterion),
            ("evaluator", self.evaluator),
            ("optimizer", self.optimizer),
            ("scheduler", self.scheduler),
        ]

        for name, obj in objects:
            if isinstance(obj, Configurable):
                configs[name] = {
                    "__class__": obj.__class__.__name__,
                    "config": obj.get_config(),
                }

        return configs

    def _setup_wandb(
        self, project: str | None, config: dict[str, JSONPrimitive]
    ) -> None:
        if project is None:
            wandb.init(mode="disabled")
            return

        wandb.init(
            job_type="train", dir=self.output_dir, config=config, project=project
        )

        wandb.define_metric("train/step", hidden=True)
        wandb.define_metric("train/lr", step_metric="train/step")
        for loss in self.criterion.losses:
            wandb.define_metric(
                f"train/{loss}", step_metric="train/step", summary="min"
            )
        wandb.define_metric("train/total_loss", step_metric="train/step", summary="min")

        wandb.define_metric("epoch", hidden=True)
        for loss in self.criterion.losses:
            wandb.define_metric(f"eval/{loss}", step_metric="epoch", summary="min")
        wandb.define_metric("eval/total_loss", step_metric="epoch", summary="min")

        for metric_name, metric_type in self.evaluator.metrics:
            if metric_type == MetricType.NUMERIC:
                wandb.define_metric(
                    f"eval/{metric_name}", step_metric="epoch", summary="max"
                )

    def _run(self) -> None:
        self.logger.info("Starting training.")

        max_epochs = self.max_epochs if self.max_epochs > 0 else math.inf
        current_epoch = 0
        self.train_step = 0
        current_patience = self.patience
        current_optimal_metric = -math.inf

        while current_epoch < max_epochs:
            self.logger.info(f"Starting epoch {current_epoch + 1}/{max_epochs}.")
            wandb.log({"epoch": current_epoch + 1})

            self._train_epoch(current_epoch, False)
            torch.cuda.empty_cache()

            if (current_epoch + 1) % self.check_val_every_n_epochs == 0:
                if isinstance(self.scheduler, ReciprocalLR):
                    state = self.get_state()
                    torch.save(state, self.output_dir / "tmp_state.pt")
                    del state

                    self.scheduler.start_cooldown()
                    self._train_epoch(current_epoch, True)
                    self.scheduler.stop_cooldown()

                    self._eval_epoch(current_epoch)

                    # restore state
                    state = torch.load(self.output_dir / "tmp_state.pt")
                    self.set_state(state)
                else:
                    self._eval_epoch(current_epoch)

                torch.cuda.empty_cache()

                metrics = self.evaluator.compute_numeric_metrics()
                metric_value = metrics[self.metric_to_optimize]
                if metric_value > current_optimal_metric:
                    current_optimal_metric = metric_value
                    current_patience = self.patience
                    model_state = self.model.get_state()
                    torch.save(model_state, self.output_dir / "model.pt")
                else:
                    current_patience -= 1

            if current_patience > 0:
                self.logger.info(f"Current patience: {current_patience}.")
            else:
                self.logger.info("No patience left.")
                break

            state = self.get_state()
            torch.save(state, self.output_dir / f"checkpoint_{current_epoch}.pt")
            del state

            if current_epoch >= self.keep_checkpoints:
                checkpoint_path = (
                    self.output_dir
                    / f"checkpoint_{current_epoch - self.keep_checkpoints}.pt"
                )
                checkpoint_path.unlink()

            self.evaluator.reset()
            current_epoch += 1

        self.logger.info("Training finished.")

    def _train_epoch(self, epoch: int, cooldown: bool) -> None:  # noqa
        if cooldown:
            self.logger.info("Cooldown epoch started.")
        else:
            self.logger.info(f"Training epoch {epoch + 1} started.")

        self.model.train()
        self.optimizer.zero_grad()

        last_n_steps_losses = {loss: 0.0 for loss in self.criterion.losses}
        last_n_steps_losses["total_loss"] = 0.0
        last_n_steps_num_samples = 0

        all_losses = {loss: 0.0 for loss in self.criterion.losses}
        all_losses["total_loss"] = 0.0
        all_num_samples = 0
        enabled = self.precision == torch.float16 or self.precision == torch.bfloat16

        start = timer()
        for samples, annotations, _ in tqdm(self.train_loader, desc="Training"):
            acc_batches = zip(
                samples.split(num_splits=self.accumulation_steps),
                annotations.split(num_splits=self.accumulation_steps),
                strict=True,
            )

            last_n_steps_num_samples += len(samples)
            all_num_samples += len(samples)

            for acc_samples, acc_annotations in acc_batches:
                acc_samples = acc_samples.move(self.device, True)
                acc_annotations = acc_annotations.move(self.device, True)

                with autocast(self.device.type, self.precision, enabled):
                    outputs = self.model(acc_samples, acc_annotations)
                    losses = self.criterion.compute(outputs, acc_annotations)

                    for loss, value in losses.items():
                        last_n_steps_losses[loss] += value.item() * len(acc_samples)
                        all_losses[loss] += value.item() * len(acc_samples)

                    total_loss: Tensor = sum(losses.values())  # type: ignore
                    last_n_steps_losses["total_loss"] += total_loss.item() * len(
                        acc_samples
                    )
                    all_losses["total_loss"] += total_loss.item() * len(acc_samples)

                    total_loss = total_loss / self.accumulation_steps

                self.scaler.scale(total_loss).backward()  # type: ignore

            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(  # type: ignore
                    self.model.parameters(), self.max_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.step_after_batch:
                self.scheduler.step()

            if cooldown:
                # if we are in cooldown mode, we do not want to log anything to wandb
                continue

            if (self.train_step + 1) % self.log_every_n_steps == 0:
                for loss, value in last_n_steps_losses.items():
                    wandb.log({f"train/{loss}": value / last_n_steps_num_samples})

                wandb.log({"train/step": self.train_step + 1})
                wandb.log({"train/lr": self.scheduler.get_last_lr()[0]})

                last_n_steps_losses = {loss: 0.0 for loss in self.criterion.losses}
                last_n_steps_losses["total_loss"] = 0.0
                last_n_steps_num_samples = 0

            self.train_step += 1

        if self.step_after_batch:
            self.scheduler.step()

        end = timer()
        if cooldown:
            self.logger.info("Cooldown epoch finished.")
        else:
            self.logger.info(f"Training epoch {epoch + 1} finished.")

        self.logger.info("Statistics:")
        elapsed_time = end - start
        self.logger.info(f"\telapsed time: {elapsed_time:.2f} s.")
        self.logger.info(
            f"\tthroughput: {all_num_samples / elapsed_time:.2f} samples/s."
        )

        self.logger.info("Losses:")
        for loss, value in all_losses.items():
            self.logger.info(f"\t{loss}: {value / all_num_samples:.4f}.")
            wandb.log({f"train/{loss}": value / all_num_samples})

    @torch.inference_mode()
    def _eval_epoch(self, epoch: int) -> None:
        self.logger.info(f"Eval epoch {epoch + 1} started.")

        self.model.eval()

        losses = {loss: 0.0 for loss in self.criterion.losses}
        losses["total_loss"] = 0.0
        num_samples = 0

        start = timer()
        for samples, annotations, targets in tqdm(self.eval_loader, desc="Eval"):
            num_samples += len(samples)
            samples = samples.move(self.device, non_blocking=True)
            annotations = annotations.move(self.device, non_blocking=True)
            targets = targets.move(self.device, non_blocking=True)

            with autocast(
                enabled=self.scaler.is_enabled(),
                device_type=self.device.type,
                dtype=self.precision,
            ):
                outputs = self.model(samples, None)
                batch_losses = self.criterion.compute(outputs, annotations)
                total_loss = 0.0

                for loss, value in batch_losses.items():
                    total_loss += value.item() * len(samples)
                    losses[loss] += value.item() * len(samples)
                losses["total_loss"] += total_loss

                predictions = self.model.postprocess(outputs)
                self.evaluator.update(predictions, targets)

        end = timer()
        self.logger.info(f"Eval epoch {epoch + 1} finished.")

        self.logger.info("Statistics:")
        elapsed_time = end - start
        self.logger.info(f"\telapsed time: {elapsed_time:.2f} s.")
        self.logger.info(f"\tthroughput: {num_samples / elapsed_time:.2f} samples/s.")

        self.logger.info("Losses:")
        for loss, value in losses.items():
            self.logger.info(f"\t{loss}: {value / num_samples:.4f}.")
            wandb.log({f"eval/{loss}": value / num_samples})

        self.logger.info("Metrics:")
        metrics = self.evaluator.compute_numeric_metrics()
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"\t{metric_name}: {metric_value:.4f}.")
            wandb.log({f"eval/{metric_name}": metric_value})


# --------------------------------------------------------------------------- #
# Private helper functions
# --------------------------------------------------------------------------- #


def _get_logger(path: Path | None) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if path is not None:
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
