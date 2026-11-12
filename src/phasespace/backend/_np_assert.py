__all__ = [
    "assert_equal",
    "assert_greater_equal",
]

import numpy as np


def assert_equal(x, y, message="", name=None):
    np.testing.assert_equal(np.asarray(x), np.asarray(y), err_msg=message)


def assert_greater_equal(x, y, message="", name=None):
    np.testing.assert_equal(np.asarray(x) >= np.asarray(y), True, err_msg=message)
