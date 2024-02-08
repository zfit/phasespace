import numpy as np
import pytest

import phasespace as phsp

from .helpers import tf_only


def create_from_seed_input():
    import tensorflow as tf

    return tf.random.Generator.from_seed(15)


@tf_only
@pytest.mark.parametrize(
    "seed",
    [
        lambda: 15,
        create_from_seed_input,
    ],
)
def test_get_rng(seed):
    rng1 = phsp.random.get_rng(seed())
    rng2 = phsp.random.get_rng(seed())
    rnd1_seeded = rng1.uniform(shape=(100,))
    rnd2_seeded = rng2.uniform(shape=(100,))

    rng3 = phsp.random.get_rng()
    rng4 = phsp.random.get_rng(seed())
    # advance rng4 by one step
    _ = rng4.split(1)

    rnd3 = rng3.uniform(shape=(100,))
    rnd4 = rng4.uniform(shape=(100,))

    np.testing.assert_array_equal(rnd1_seeded, rnd2_seeded)
    assert not np.array_equal(rnd1_seeded, rnd3)
    assert not np.array_equal(rnd4, rnd3)
