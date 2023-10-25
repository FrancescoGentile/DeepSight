##
##
##

import warnings
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from deepsight.typing import Configurable, JSONPrimitive


class ReciprocalLR(LRScheduler, Configurable):
    r"""A learning rate scheduler that uses a reciprocal schedule.

    The learning rate is increased linearly from 0 to the maximum learning rate
    during the warmup period, if any.

    During the main training period, the learning rate is decreased proportionally
    to the inverse of the step number. That is, the learning rate at step t is
    equal to:

    $$
        lr_t = max_lr * \\sqrt{\\frac{w}{t}}
    $$

    where `max_lr` is the maximum learning rate, `w` is the number of warmup steps,
    and `t` is the current step number. If there are no warmup steps, `w` is set to 1
    so that the learning rate is equal to `max_lr` at the first step.

    Since this scheduler does not require the number of training steps, it can be
    used to train the model for an arbitrary number of steps. Since in the last training
    steps the learning rate should be close to 0, the scheduler supports a cooldown
    phase to make it possible to train the model for few more steps during which the
    learning rate is rapidly decreased linearly to 0.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float | list[float],
        warmup_steps: int,
        cooldown_steps: int,
    ) -> None:
        """Initializes a new instance of ``ReciprocalLR``.

        Args:
            optimizer : The wrapped optimizer.
            max_lr : The maximum learning rate. If a list is provided, it must have the
                same length as the number of parameter groups in the optimizer.
            warmup_steps : The number of warmup steps.
            cooldown_steps : The number of cooldown steps.
        """
        if isinstance(max_lr, float):
            max_lr = [max_lr] * len(optimizer.param_groups)
        elif isinstance(max_lr, list):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(
                    "Expected max_lr to be a float or a list of floats "
                    f"with length {len(optimizer.param_groups)}."
                )
        else:
            raise ValueError(
                "Expected max_lr to be a float or a list of floats "
                f"with length {len(optimizer.param_groups)}."
            )

        self.max_lr = max_lr
        self.warmup_steps = warmup_steps if warmup_steps > 0 else 1
        self.cooldown_steps = cooldown_steps
        self.last_step_pre_cooldown = -1
        self.pre_cooldown_step = -1
        self.pre_cooldown_lr: list[float] | None = None
        self.warmup_deltas = [lr / self.warmup_steps for lr in self.max_lr]

        super().__init__(optimizer, -1, False)

    def start_cooldown(self) -> None:
        """Starts the cooldown phase.

        By invoking this method, the scheduler will start the cooldown phase. During
        the cooldown phase, the learning rate will be decreased linearly from the
        learning rate at the last step of the main training period to 0.

        !!! warning

            This method should be invoked only if the cooldown phase has not been
            started yet. Otherwise, a warning will be issued.
        """
        if self.pre_cooldown_lr is not None:
            warnings.warn(
                "A cooldown phase has already been started. "
                "Ignoring the request to start a new cooldown phase.",
                UserWarning,
                stacklevel=1,
            )
        else:
            self.pre_cooldown_step = self.last_epoch
            self.pre_cooldown_lr = self.get_last_lr()

    def stop_cooldown(self) -> None:
        """Stops the cooldown phase.

        By invoking this method, the scheduler will resume the main training phase
        like if the cooldown phase had never been started.

        !!! warning

            This method should be invoked only if the cooldown phase has been started.
            Otherwise, a warning will be issued.
        """
        if self.pre_cooldown_lr is None:
            warnings.warn(
                "Stopping the cooldown phase before it has been started.",
                UserWarning,
                stacklevel=1,
            )

        self.last_epoch = self.pre_cooldown_step - 1
        self.pre_cooldown_step = -1
        self.pre_cooldown_lr = None
        self.step()

    def get_lr(self) -> list[float]:  # type: ignore
        if not self._get_lr_called_within_step:  # type: ignore
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=1,
            )

        if self.pre_cooldown_lr is not None:
            # we need to decrease the learning rate linearly from the learning rate
            # used before the cooldown to 0 over the cooldown period
            deltas = [lr / self.cooldown_steps for lr in self.pre_cooldown_lr]
            return [
                max(lr - delta * (self.last_epoch - self.pre_cooldown_step), 0.0)
                for lr, delta in zip(self.pre_cooldown_lr, deltas, strict=True)
            ]

        if self.last_epoch < self.warmup_steps:
            # we need to increase the learning rate linearly from 0 to the max learning
            # rate over the warmup period
            return [delta * (self.last_epoch + 1) for delta in self.warmup_deltas]

        # we are in the main training phase, so we set the learning rate equal to
        # max_lr * 1 / sqrt(step)
        return [
            lr * (self.warmup_steps / (self.last_epoch + 1)) ** 0.5
            for lr in self.max_lr
        ]

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["last_step_pre_cooldown"] = self.last_step_pre_cooldown
        state["pre_cooldown_step"] = self.pre_cooldown_step
        state["pre_cooldown_lr"] = self.pre_cooldown_lr

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.last_step_pre_cooldown = state_dict["last_step_pre_cooldown"]
        self.pre_cooldown_step = state_dict["pre_cooldown_step"]
        self.pre_cooldown_lr = state_dict["pre_cooldown_lr"]

    def get_config(self) -> JSONPrimitive:
        return {
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            "cooldown_steps": self.cooldown_steps,
        }  # type: ignore

    def __str__(self) -> str:
        return (
            f"ReciprocalLR(max_lr={self.max_lr}, warmup_steps={self.warmup_steps}, "
            f"cooldown_steps={self.cooldown_steps})"
        )
