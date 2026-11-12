from importlib.util import find_spec

import pytest

from phasespace.backend import BACKEND, BackendType

tf_only = pytest.mark.skipif(
    BACKEND != BackendType.TENSORFLOW,
    reason="Test requires tensorflow",
)

AVAILABLE_BACKENDS = [
    name for name in ("numpy", "tensorflow", "jax") if find_spec(name) is not None
]
