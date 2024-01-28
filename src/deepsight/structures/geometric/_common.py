# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import enum


class BatchMode(enum.Enum):
    CONCAT = enum.auto()
    STACK = enum.auto()
    SEQUENCE = enum.auto()
