##
##
##

from typing import Any

from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()

        self.drop_prob = drop_prob

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("DropPath is not implemented.")
