#!/usr/bin/env python3
"""Test core generation functionality across all backends."""

import numpy as np
import pytest


def _check_backend_available(backend_name):
    """Check if a backend is available."""
    if backend_name == "tensorflow":
        try:
            import tensorflow  # noqa: F401

            return True
        except ImportError:
            return False
    elif backend_name == "numpy":
        return True  # Always available
    return False


# Create list of available backends for parametrization
AVAILABLE_BACKENDS = [b for b in ["numpy", "tensorflow"] if _check_backend_available(b)]

# Test constants
B0_MASS = 5279.58
PION_MASS = 139.57018
KAON_MASS = 493.677
KSTARZ_MASS = 895.81


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_nbody_decay_simple(backend_name, backend_context):
    """Test simple n-body decay generation."""
    backend_context(backend_name)

    import phasespace

    n_events = 1000
    decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, KAON_MASS])
    weights, particles = decay.generate(n_events=n_events)

    # Convert to numpy for assertions
    weights_np = np.asarray(weights)
    assert weights_np.shape == (n_events,)
    assert np.all(weights_np >= 0)
    # Weights are phase space weights, not necessarily <= 1

    # Check particles
    assert len(particles) == 2
    for i, p in enumerate(particles.values()):
        p_np = np.asarray(p)
        assert p_np.shape == (n_events, 4)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_sequential_decay(backend_name, backend_context):
    """Test sequential decay chain generation."""
    backend_context(backend_name)

    from phasespace import GenParticle

    n_events = 500

    # Build decay: B0 -> K* gamma, K* -> K+ pi-
    kaon = GenParticle("K+", KAON_MASS)
    pion = GenParticle("pi-", PION_MASS)
    kstar = GenParticle("K*", KSTARZ_MASS).set_children(kaon, pion)
    gamma = GenParticle("gamma", 0)
    b0 = GenParticle("B0", B0_MASS).set_children(kstar, gamma)

    weights, particles = b0.generate(n_events=n_events)

    # Convert to numpy
    weights_np = np.asarray(weights)
    assert weights_np.shape == (n_events,)

    # Check all particles are present
    expected_particles = {"K*", "K+", "pi-", "gamma"}
    assert set(particles.keys()) == expected_particles

    for name, p in particles.items():
        p_np = np.asarray(p)
        assert p_np.shape == (n_events, 4), f"Wrong shape for {name}"


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_energy_momentum_conservation(backend_name, backend_context):
    """Test that energy-momentum is conserved."""
    backend_context(backend_name)

    import phasespace

    n_events = 100
    masses = [PION_MASS, KAON_MASS]
    decay = phasespace.nbody_decay(B0_MASS, masses)
    _, particles = decay.generate(n_events=n_events)

    # Sum momenta
    total_momentum = np.zeros((n_events, 4))
    for p in particles.values():
        total_momentum += np.asarray(p)

    # Check conservation (should be at rest in parent frame)
    # Spatial components should be ~0
    np.testing.assert_allclose(total_momentum[:, :3], 0, atol=1e-6)
    # Energy should equal parent mass
    np.testing.assert_allclose(total_momentum[:, 3], B0_MASS, rtol=1e-6)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_mass_calculation(backend_name, backend_context):
    """Test that generated particles have correct masses."""
    backend_context(backend_name)

    import phasespace

    n_events = 100
    masses = [PION_MASS, KAON_MASS]
    decay = phasespace.nbody_decay(B0_MASS, masses)
    _, particles = decay.generate(n_events=n_events)

    for i, (name, p) in enumerate(particles.items()):
        p_np = np.asarray(p)
        # Calculate invariant mass: m^2 = E^2 - p^2
        mass_sq = p_np[:, 3] ** 2 - np.sum(p_np[:, :3] ** 2, axis=1)
        mass = np.sqrt(mass_sq)
        np.testing.assert_allclose(mass, masses[i], rtol=1e-6)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_boost(backend_name, backend_context):
    """Test boosting generated events."""
    backend_context(backend_name)

    import phasespace
    from phasespace.backend import tnp

    n_events = 100
    decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, KAON_MASS])

    # Create boost momentum (moving in z direction) - one per event
    energy = np.sqrt(B0_MASS**2 + 1000**2)
    boost_momentum = tnp.asarray([[0, 0, 1000, energy]] * n_events)

    _, particles = decay.generate(n_events=n_events, boost_to=boost_momentum)

    # Check that particles have non-zero z-momentum on average
    for p in particles.values():
        p_np = np.asarray(p)
        mean_pz = np.mean(p_np[:, 2])
        assert mean_pz > 0, "Boost should give positive z-momentum on average"


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_reproducibility_with_seed(backend_name, backend_context):
    """Test that generation is reproducible with same seed."""
    backend_context(backend_name)

    import phasespace

    n_events = 100
    decay = phasespace.nbody_decay(B0_MASS, [PION_MASS, KAON_MASS])

    # Generate with seed
    weights1, particles1 = decay.generate(n_events=n_events, seed=42)
    weights2, particles2 = decay.generate(n_events=n_events, seed=42)

    np.testing.assert_allclose(np.asarray(weights1), np.asarray(weights2))
    for (_, p1), (_, p2) in zip(particles1.items(), particles2.items()):
        np.testing.assert_allclose(np.asarray(p1), np.asarray(p2))


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_different_n_particles(backend_name, backend_context):
    """Test generation with different numbers of decay products."""
    backend_context(backend_name)

    import phasespace

    n_events = 50

    for n_particles in [2, 3, 4, 5]:
        masses = [PION_MASS] * n_particles
        decay = phasespace.nbody_decay(B0_MASS, masses)
        weights, particles = decay.generate(n_events=n_events)

        weights_np = np.asarray(weights)
        assert weights_np.shape == (n_events,)
        assert len(particles) == n_particles


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_set_backend_function(backend_name, backend_context):
    """Test the set_backend function."""
    backend_context(backend_name)

    from phasespace.backend import BackendType, get_backend

    # Get current backend
    current = get_backend()
    assert current is not None

    # Verify it matches what we set
    expected = BackendType.from_string(backend_name)
    assert current == expected


class TestBackendSwitching:
    """Test backend switching functionality."""

    def test_switch_warning(self, backend_context):
        """Test that switching backends emits a warning."""
        if len(AVAILABLE_BACKENDS) < 2:
            pytest.skip("Need at least 2 backends to test switching")

        backend_context(AVAILABLE_BACKENDS[0])

        from phasespace.backend import set_backend

        with pytest.warns(UserWarning, match="Switching backend"):
            set_backend(AVAILABLE_BACKENDS[1])

    def test_same_backend_no_warning(self, backend_context):
        """Test that setting same backend doesn't warn."""
        import warnings

        backend_context(AVAILABLE_BACKENDS[0])

        from phasespace.backend import get_backend, set_backend

        current = get_backend()
        # Setting same backend should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            set_backend(current)  # Should not raise any warning
