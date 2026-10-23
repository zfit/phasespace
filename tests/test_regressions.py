"""Regression tests for generation and kinematic edge cases."""

import numpy as np
import pytest


@pytest.mark.parametrize("backend_name", ["numpy", "tensorflow"])
def test_resonance_uses_generation_seed(backend_name, backend_context):
    backend_context(backend_name)
    import phasespace

    def mass(min_mass, max_mass, n_events, seed):
        uniform = phasespace.random.generate_uniform(seed, shape=(n_events,))
        return min_mass + (max_mass - min_mass) * uniform

    resonance = phasespace.GenParticle("resonance", mass).set_children(
        phasespace.GenParticle("a", 1), phasespace.GenParticle("b", 1)
    )
    decay = phasespace.GenParticle("parent", 10).set_children(
        resonance, phasespace.GenParticle("c", 1)
    )
    first_weights, first_parts = decay.generate(20, seed=42)
    weights, parts = decay.generate(20, seed=42)
    np.testing.assert_allclose(weights, first_weights)
    for name in parts:
        np.testing.assert_allclose(parts[name], first_parts[name])


@pytest.mark.parametrize("backend_name", ["numpy", "tensorflow"])
def test_zero_and_nonzero_boosts(backend_name, backend_context):
    backend_context(backend_name)
    from phasespace import kinematics
    from phasespace.backend import tnp

    momenta = tnp.asarray([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=tnp.float64)
    boosts = tnp.asarray([[0, 0, 0], [0, 0, 0.5]], dtype=tnp.float64)
    with np.errstate(divide="raise", invalid="raise"):
        boosted = kinematics.lorentz_boost(momenta, boosts)
    gamma = 1 / np.sqrt(1 - 0.5**2)
    np.testing.assert_allclose(boosted, [[0, 0, 0, 1], [0, 0, gamma * 0.5, gamma]])


def test_numpy_greater_equal_accepts_equality(backend_context):
    backend_context("numpy")
    from phasespace.backend import assert_greater_equal

    assert_greater_equal(
        np.array([0, 1]), np.array([0, 0]), message="inclusive comparison"
    )
    with pytest.raises(AssertionError):
        assert_greater_equal(
            np.array([-1, 1]), np.array([0, 0]), message="negative value"
        )
