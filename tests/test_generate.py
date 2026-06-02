#!/usr/bin/env python3
# =============================================================================
# @file   test_generate.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   27.02.2019
# =============================================================================
"""Basic dimensionality tests."""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(__file__))


def _check_backend_available(backend_name):
    """Check if a backend is available."""
    if backend_name == "tensorflow":
        try:
            import tensorflow  # noqa: F401

            return True
        except ImportError:
            return False
    elif backend_name == "numpy":
        return True
    return False


AVAILABLE_BACKENDS = [b for b in ["numpy", "tensorflow"] if _check_backend_available(b)]

from .helpers import decays  # noqa: E402

B0_MASS = decays.B0_MASS
PION_MASS = decays.PION_MASS


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_one_event(backend_name, backend_context):
    """Test B->pi pi pi."""
    backend_context(backend_name)
    import phasespace

    decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS])
    norm_weights, particles = decay.generate(n_events=1)
    assert norm_weights.shape[0] == 1
    assert len(particles) == 3
    assert all(part.shape == (1, 4) for part in particles.values())


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("as_vectors", [True, False], ids=["as_vectors", "as_arrays"])
def test_one_event_tf(backend_name, backend_context, as_vectors):
    """Test B->pi pi pi."""
    backend_context(backend_name)
    import phasespace

    decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS])
    norm_weights, particles = decay.generate(n_events=1, as_vectors=as_vectors)
    if as_vectors:
        particles = {
            k: np.stack([p.px, p.py, p.pz, p.E], axis=-1) for k, p in particles.items()
        }

    assert norm_weights.shape[0] == 1
    assert len(particles) == 3
    assert all(part.shape == (1, 4) for part in particles.values())


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("n_events", argvalues=[5, 523])
@pytest.mark.parametrize("as_vectors", [True, False], ids=["as_vectors", "as_arrays"])
def test_n_events(backend_name, backend_context, n_events, as_vectors):
    """Test 5 B->pi pi pi."""
    backend_context(backend_name)
    import phasespace

    decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS])
    norm_weights, particles = decay.generate(n_events=n_events, as_vectors=as_vectors)
    if as_vectors:
        particles = {
            k: np.stack([p.px, p.py, p.pz, p.E], axis=-1) for k, p in particles.items()
        }
    assert norm_weights.shape[0] == n_events
    assert len(particles) == 3
    assert all(part.shape == (n_events, 4) for part in particles.values())


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_deterministic_events(backend_name, backend_context):
    backend_context(backend_name)
    import phasespace

    decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, PION_MASS, PION_MASS])
    common_seed = 36
    norm_weights_seeded1, particles_seeded1 = decay.generate(
        n_events=100, seed=common_seed
    )
    norm_weights_global, particles_global = decay.generate(n_events=100)
    norm_weights_rnd, particles_rnd = decay.generate(n_events=100, seed=152)
    norm_weights_seeded2, particles_seeded2 = decay.generate(
        n_events=100, seed=common_seed
    )

    np.testing.assert_allclose(norm_weights_seeded1, norm_weights_seeded2)
    for part1, part2 in zip(particles_seeded1.values(), particles_seeded2.values()):
        np.testing.assert_allclose(part1, part2)

    assert not np.allclose(norm_weights_seeded1, norm_weights_rnd)
    for part1, part2 in zip(particles_seeded1.values(), particles_rnd.values()):
        assert not np.allclose(part1, part2)

    assert not np.allclose(norm_weights_global, norm_weights_rnd)
    for part1, part2 in zip(particles_global.values(), particles_rnd.values()):
        assert not np.allclose(part1, part2)


if __name__ == "__main__":
    test_one_event()
    test_n_events(5)

    # EOF
