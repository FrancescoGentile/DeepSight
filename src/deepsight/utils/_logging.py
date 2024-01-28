# Copyright 2024 The DeepSight Team.
# SPDX-License-Identifier: Apache-2.0

import logging


def get_library_logger() -> logging.Logger:
    """Return the logger of the library."""
    return logging.getLogger("deepsight")
