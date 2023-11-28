##
##
##

import logging


def get_library_logger() -> logging.Logger:
    """Return the logger of the library."""
    return logging.getLogger(__package__)
