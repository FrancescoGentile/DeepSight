# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import abc
from collections.abc import Callable
from typing import Any


class Module(abc.ABC):
    """Base class for all modules."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    process: Callable[..., Any] = abc.abstractmethod


class B(Module): ...


b = B()
