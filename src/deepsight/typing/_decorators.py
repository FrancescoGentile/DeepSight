# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import enum


def str_enum[T: enum.Enum](cls: type[T]) -> type[T]:
    """Create a string enum from an enum class.

    To be a string enum, the enum class must follow the following conventions:
    - All values must be strings;
    - All names must be uppercase;
    - All values must be lowercase;
    - All values must be the lowercase version of the name.

    If the enum class follows these conventions, then this decorator will
    add:
    - a `__str__` method that returns the value of the enum member with
        underscores replaced with dashes;
    - a `__repr__` method that returns `EnumName.VALUE`.

    Args:
        cls: The enum class to convert to a string enum.

    Returns:
        The string enum class.

    Raises:
        ValueError: If the enum class does not follow the conventions.
    """
    for member in cls:
        if not isinstance(member.value, str):
            raise ValueError(f"Enum member <{member}> has a non-string value.")
        if member.name != member.name.upper():
            raise ValueError(f"Enum member <{member}> has a non-uppercase name.")
        if member.value != member.value.lower():
            raise ValueError(f"Enum member <{member}> has a non-lowercase value.")
        if member.value != member.name.lower():
            raise ValueError(
                f"Enum member <{member}> has a value <{member.value}> "
                "that is not the lowercase version of its name."
            )

    cls.__str__ = lambda self: self.value.replace("_", "-")  # type: ignore
    cls.__repr__ = lambda self: f"{self.__class__.__name__}.{self.value.upper()}"  # type: ignore

    return cls
