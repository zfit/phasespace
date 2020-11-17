import numpy as np
import pytest
import tensorflow as tf

import phasespace as phsp


@pytest.mark.parametrize('seed', [lambda: 15, lambda: tf.random.Generator.from_seed(15)])
def test_get_rng(seed):
    rng1 = phsp.random.get_rng(seed())
    rng2 = phsp.random.get_rng(seed())
    rnd1_seeded = rng1.uniform_full_int(shape=(100,))
    rnd2_seeded = rng2.uniform_full_int(shape=(100,))

    rng3 = phsp.random.get_rng()
    rng4 = phsp.random.get_rng(seed())
    # advance rng4 by one step
    _ = rng4.split(1)

    rnd3 = rng3.uniform_full_int(shape=(100,))
    rnd4 = rng4.uniform_full_int(shape=(100,))

    np.testing.assert_array_equal(rnd1_seeded, rnd2_seeded)
    assert not np.array_equal(rnd1_seeded, rnd3)
    assert not np.array_equal(rnd4, rnd3)
