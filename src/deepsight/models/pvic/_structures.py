##
##
##

from dataclasses import dataclass
from typing import Annotated

from torch import Tensor


@dataclass(frozen=True, slots=True)
class Output:
    ho_logits: list[Annotated[Tensor, "I V", float]]
    ho_indices: list[Annotated[Tensor, "I 2", int]]
    num_entities: list[int]
