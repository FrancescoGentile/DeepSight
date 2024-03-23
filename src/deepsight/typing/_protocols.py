# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import typing
from typing import Any, Protocol, Self

import torch


@typing.runtime_checkable
class Moveable(Protocol):
    """An interface for objects that can be moved to a device."""

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self:
        """Move this object to the given device.

        !!! note

            This interface does not specify whether the move should be in-place
            or not. This is left to the implementer.

        Args:
            device: The device to move to.
            non_blocking: Whether to perform the move asynchronously.

        Returns:
            The object on the given device.
        """
        ...


@typing.runtime_checkable
class Configurable(Protocol):
    """An interface for objects that can be configured."""

    def get_config(self, recursive: bool) -> dict[str, Any]:
        """Get the configuration of this object.

        Args:
            recursive: Whether to recursively get the configuration of this object.
                For example, if this object has a child object that is configurable,
                then the configuration of the child object will be included in the
                returned configuration.
        """
        ...


@typing.runtime_checkable
class Stateful(Protocol):
    """An interface for objects that have a state."""

    def state_dict(self) -> dict[str, Any]:
        """Get the state of this object."""
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        """Load the state of this object."""
        ...


@typing.runtime_checkable
class Detachable(Protocol):
    """An interface for objects that can be detached from the computation graph."""

    def detach(self) -> Self:
        """Detach this object from the computation graph."""
        ...
