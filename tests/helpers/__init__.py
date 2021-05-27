import pytest

from phasespace.backend import BACKEND, BackendType

tf_only = pytest.mark.skipif(
    BACKEND != BackendType.TENSORFLOW,
    reason="Test requires tensorflow",
)
