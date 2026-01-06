#!/usr/bin/env python3
# =============================================================================
# @file   test_chain.py
# @author Albert Puig (albert.puig@cern.ch)
# @date   01.03.2019
# =============================================================================
"""Test decay chain tools."""

import os
import sys

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


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_name_clashes(backend_name, backend_context):
    """Test clashes in particle naming."""
    backend_context(backend_name)
    from phasespace import GenParticle

    # In children
    with pytest.raises(KeyError):
        GenParticle("Top", 0).set_children(
            GenParticle("Kstarz", mass=decays.KSTARZ_MASS),
            GenParticle("Kstarz", mass=decays.KSTARZ_MASS),
        )
    # With itself
    with pytest.raises(KeyError):
        GenParticle("Top", 0).set_children(
            GenParticle("Top", mass=decays.KSTARZ_MASS),
            GenParticle("Kstarz", mass=decays.KSTARZ_MASS),
        )
    # In grandchildren
    with pytest.raises(KeyError):
        GenParticle("Top", 0).set_children(
            GenParticle("Kstarz0", mass=decays.KSTARZ_MASS).set_children(
                GenParticle("K+", mass=decays.KAON_MASS),
                GenParticle("pi-", mass=decays.PION_MASS),
            ),
            GenParticle("Kstarz0", mass=decays.KSTARZ_MASS).set_children(
                GenParticle("K+", mass=decays.KAON_MASS),
                GenParticle("pi-_1", mass=decays.PION_MASS),
            ),
        )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_wrong_children(backend_name, backend_context):
    """Test wrong number of children."""
    backend_context(backend_name)
    from phasespace import GenParticle

    with pytest.raises(ValueError):
        GenParticle("Top", 0).set_children(
            GenParticle("Kstarz0", mass=decays.KSTARZ_MASS)
        )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_grandchildren(backend_name, backend_context):
    """Test that grandchildren detection is correct."""
    backend_context(backend_name)
    from phasespace import GenParticle

    top = GenParticle("Top", 0)
    assert not top.has_children
    assert not top.has_grandchildren
    assert not top.set_children(
        GenParticle("Child1", mass=decays.KSTARZ_MASS),
        GenParticle("Child2", mass=decays.KSTARZ_MASS),
    ).has_grandchildren


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_reset_children(backend_name, backend_context):
    """Test when children are set twice."""
    backend_context(backend_name)
    from phasespace import GenParticle

    top = GenParticle("Top", 0).set_children(
        GenParticle("Child1", mass=decays.KSTARZ_MASS),
        GenParticle("Child2", mass=decays.KSTARZ_MASS),
    )
    with pytest.raises(ValueError):
        top.set_children(
            GenParticle("Child3", mass=decays.KSTARZ_MASS),
            GenParticle("Child4", mass=decays.KSTARZ_MASS),
        )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_no_children(backend_name, backend_context):
    """Test when no children have been configured."""
    backend_context(backend_name)
    from phasespace import GenParticle

    top = GenParticle("Top", 0)
    with pytest.raises(ValueError):
        top.generate(n_events=1)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_resonance_top(backend_name, backend_context):
    """Test when a resonance is used as the top particle."""
    if backend_name != "tensorflow":
        pytest.skip("Test requires TensorFlow for resonance mass functions")
    backend_context(backend_name)

    kstar = decays.b0_to_kstar_gamma().children[0]
    with pytest.raises(ValueError):
        kstar.generate(n_events=1)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_kstargamma(backend_name, backend_context):
    """Test B0 -> K*gamma."""
    if backend_name != "tensorflow":
        pytest.skip("Test requires TensorFlow for resonance mass functions")
    backend_context(backend_name)

    decay = decays.b0_to_kstar_gamma()
    norm_weights, particles = decay.generate(n_events=1000)
    assert norm_weights.shape[0] == 1000
    assert len(particles) == 4
    assert set(particles.keys()) == {"K*0", "gamma", "K+", "pi-"}
    assert all(part.shape == (1000, 4) for part in particles.values())


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_k1gamma(backend_name, backend_context):
    """Test B+ -> K1 (K*pi) gamma."""
    if backend_name != "tensorflow":
        pytest.skip("Test requires TensorFlow for resonance mass functions")
    backend_context(backend_name)

    decay = decays.bp_to_k1_kstar_pi_gamma()
    norm_weights, particles = decay.generate(n_events=1000)
    assert norm_weights.shape[0] == 1000
    assert len(particles) == 6
    assert set(particles.keys()) == {"K1+", "K*0", "gamma", "K+", "pi-", "pi+"}
    assert all(part.shape == (1000, 4) for part in particles.values())


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_repr(backend_name, backend_context):
    """Test string representation."""
    if backend_name != "tensorflow":
        pytest.skip("Test requires TensorFlow for resonance mass functions")
    backend_context(backend_name)

    b0 = decays.b0_to_kstar_gamma()
    kst = b0.children[0]
    assert (
        str(b0)
        == "<phasespace.GenParticle: name='B0' mass=5279.58 children=[K*0, gamma]>"
    )
    assert (
        str(kst)
        == "<phasespace.GenParticle: name='K*0' mass=variable children=[K+, pi-]>"
    )


if __name__ == "__main__":
    test_name_clashes()
    test_kstargamma()
    test_k1gamma()

# EOF
