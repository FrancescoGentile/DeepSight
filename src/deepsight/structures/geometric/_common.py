##
##
##

import enum


class BatchMode(enum.Enum):
    CONCAT = enum.auto()
    STACK = enum.auto()
    SEQUENCE = enum.auto()
