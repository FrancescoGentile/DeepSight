##
##
##

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import torch

from deepsight.typing import Configs, Configurable, Loss, Losses, str_enum


@dataclass(frozen=True)
class ClipGradNorm(Configurable):
    """Clips the norm of the gradients."""

    max_norm: float

    def get_configs(self, recursive: bool) -> Configs:
        return {"max_norm": self.max_norm}


@dataclass(frozen=True)
class ClipGradValue(Configurable):
    """Clips the value of the gradients."""

    value: float

    def get_configs(self, recursive: bool) -> Configs:
        return {"value": self.value}


class BatchLosses:
    """The losses computed over a batch."""

    # ----------------------------------------------------------------------- #
    # Constructor and Factory Methods
    # ----------------------------------------------------------------------- #

    def __init__(self, losses: Losses, batch_size: int) -> None:
        self._losses = losses
        self._batch_size = batch_size

    @classmethod
    def accumulate(cls, batch_losses: Iterable[Self]) -> Self:
        total_batch_size = sum(loss.batch_size for loss in batch_losses)
        accumulated = {}
        for loss in batch_losses:
            for name, value in loss.items():
                accumulated[name] = accumulated.get(name, 0) + value * loss.batch_size

        for name, value in accumulated.items():
            accumulated[name] = value / total_batch_size

        return cls(losses=accumulated, batch_size=total_batch_size)

    # ----------------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------------- #

    @property
    def batch_size(self) -> int:
        """The size of the batch."""
        return self._batch_size

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def keys(self) -> Iterable[str]:
        """Return an iterable over the loss names."""
        return self._losses.keys()

    def values(self) -> Iterable[Loss]:
        """Return an iterable over the loss values."""
        return self._losses.values()

    def items(self) -> Iterable[tuple[str, Loss]]:
        """Return an iterable over the losses."""
        return self._losses.items()

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __getitem__(self, name: str) -> Loss:
        return self._losses[name]

    def __setitem__(self, name: str, value: Loss) -> None:
        self._losses[name] = value


@str_enum
class Precision(enum.Enum):
    FP32 = "fp32"
    AMP_FP16 = "amp_fp16"
    AMP_BF16 = "amp_bf16"

    def is_mixed_precision(self) -> bool:
        """Return whether the precision is mixed."""
        return self in [Precision.AMP_FP16, Precision.AMP_BF16]

    def to_torch_dtype(self) -> torch.dtype:
        """Return the corresponding torch dtype."""
        match self:
            case Precision.FP32:
                return torch.float32
            case Precision.AMP_FP16:
                return torch.float16
            case Precision.AMP_BF16:
                return torch.bfloat16

    def is_supported_by_device(self, device: torch.device | str) -> bool:
        """Return whether the precision is supported by the device."""
        if self.is_mixed_precision():
            return torch.device(device).type == "cuda"
        else:
            return True
