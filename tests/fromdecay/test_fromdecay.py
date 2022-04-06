from __future__ import annotations

from copy import deepcopy

import pytest
from decaylanguage import DecayChain, DecayMode
from numpy.testing import assert_almost_equal

from phasespace.fromdecay import GenMultiDecay
from phasespace.fromdecay.mass_functions import DEFAULT_CONVERTER

from . import example_decay_chains


def check_norm(full_decay: GenMultiDecay, **kwargs) -> list[tuple]:
    """Helper function that tests whether the normalize_weights argument works for GenMultiDecay.generate.

    Args:
        full_decay: full_decay.generate will be called.
        kwargs: Additional parameters passed to generate.

    Returns:
        All the values returned by generate, both times.
        The return arguments from normalize_weights=True is the first element in the returned list.
    """
    all_return_args = []
    for norm in (True, False):
        return_args = full_decay.generate(normalize_weights=norm, **kwargs)
        assert len(return_args) == 2 if norm else 3
        assert sum(len(w) for w in return_args[0]) == kwargs["n_events"]
        if not norm:
            assert all(
                len(w) == len(mw) for w, mw in zip(return_args[0], return_args[1])
            )

        all_return_args.append(return_args)

    return all_return_args


def test_invalid_chain():
    """Test that a ValueError is raised when a value in the fs key is not a str or dict."""
    dm1 = DecayMode(1, "K- pi+ pi+ pi0", model="PHSP", zfit="rel-BW")
    dm2 = DecayMode(1, "gamma gamma")
    dc = DecayChain("D+", {"D+": dm1, "pi0": dm2}).to_dict()
    dc["D+"][0]["fs"][0] = 1
    with pytest.raises(TypeError):
        GenMultiDecay.from_dict(dc)


def test_invalid_mass_func_name():
    """Test if an invalid mass function name as the zfit parameter raises a KeyError."""
    dm1 = DecayMode(1, "K- pi+ pi+ pi0", model="PHSP")
    dm2 = DecayMode(1, "gamma gamma", zfit="invalid name")
    dc = DecayChain("D+", {"D+": dm1, "pi0": dm2}).to_dict()
    with pytest.raises(KeyError):
        GenMultiDecay.from_dict(dc, tolerance=1e-10)


def test_single_chain():
    """Test converting a DecayLanguage dict with only one possible decay.

    Since dplus_single is constructed using DecayChain.to_dict, this also tests that the code works dicts
    created from DecayChains, not just .dec files.
    """
    container = GenMultiDecay.from_dict(
        example_decay_chains.dplus_single, tolerance=1e-10
    )
    output_decay = container.gen_particles
    assert len(output_decay) == 1
    prob, gen = output_decay[0]
    assert_almost_equal(prob, 1)
    assert gen.name == "D+"
    assert {p.name for p in gen.children} == {"K-", "pi+", "pi+ [0]", "pi0"}
    for p in gen.children:
        if "pi0" == p.name[:3]:
            assert not p.has_fixed_mass
            assert p._mass.__name__ == "relbw"
        else:
            assert p.has_fixed_mass

    check_norm(container, n_events=1)
    (_, decay_list), _ = check_norm(container, n_events=100)
    assert len(decay_list) == 1
    events = decay_list[0]
    assert set(events.keys()) == {"K-", "pi+", "pi+ [0]", "pi0", "gamma", "gamma [0]"}
    assert all(len(p) == 100 for p in events.values())


def test_branching_children():
    """Test converting a DecayLanguage dict where the mother particle can decay in many ways."""
    container = GenMultiDecay.from_dict(
        example_decay_chains.pi0_4branches, tolerance=1e-10
    )
    output_decays = container.gen_particles
    assert len(output_decays) == 4
    assert_almost_equal(sum(d[0] for d in output_decays), 1)
    check_norm(container, n_events=1)
    check_norm(container, n_events=100)


def test_branching_grandchilden():
    """Test converting a DecayLanguage dict where children to the mother particle can decay in many ways."""
    # Specify different mass functions for the different decays of pi0
    decay_dict = deepcopy(example_decay_chains.dplus_4grandbranches)

    # Add different zfit parameters to all pi0 decays. The fourth decay has no zfit parameter
    for mass_function, decay_mode in zip(
        ("relbw", "bw", "gauss"), decay_dict["D+"][0]["fs"][-1]["pi0"]
    ):
        decay_mode["zfit"] = mass_function

    container = GenMultiDecay.from_dict(decay_dict, tolerance=1e-10)

    output_decays = container.gen_particles
    assert len(output_decays) == 4
    assert_almost_equal(sum(d[0] for d in output_decays), 1)

    for p, mass_func in zip(
        output_decays, ("relbw", "bw", "gauss", GenMultiDecay.DEFAULT_MASS_FUNC)
    ):
        gen_particle = p[1]  # Ignore probability
        assert gen_particle.children[-1].name == "pi0"
        # Check that the zfit parameter assigns the correct mass function
        assert gen_particle.children[-1]._mass.__name__ == mass_func

    check_norm(container, n_events=1)
    check_norm(container, n_events=100)


def test_particle_model_map():
    """Test that the particle_model_map parameter works as intended."""
    container = GenMultiDecay.from_dict(
        example_decay_chains.dplus_4grandbranches,
        particle_model_map={"pi0": "bw"},
        tolerance=1e-10,
    )
    output_decays = container.gen_particles
    assert len(output_decays) == 4
    assert_almost_equal(sum(d[0] for d in output_decays), 1)
    for p in output_decays:
        gen_particle = p[1]  # Ignore probability
        assert gen_particle.children[-1].name[:3] == "pi0"
        # Check that particle_model_map has assigned the bw mass function to all pi0 decays.
        assert gen_particle.children[-1]._mass.__name__ == "bw"
    check_norm(container, n_events=1)
    check_norm(container, n_events=100)


def test_mass_converter():
    """Test that the mass_converter parameter works as intended."""
    dplus_4grandbranches_massfunc = deepcopy(example_decay_chains.dplus_4grandbranches)
    dplus_4grandbranches_massfunc["D+"][0]["fs"][-1]["pi0"][-1]["zfit"] = "rel-BW"
    container = GenMultiDecay.from_dict(
        dplus_4grandbranches_massfunc,
        tolerance=1e-10,
        mass_converter={"rel-BW": DEFAULT_CONVERTER["relbw"]},
    )
    output_decays = container.gen_particles
    assert len(output_decays) == 4
    assert_almost_equal(sum(d[0] for d in output_decays), 1)

    for decay in output_decays:
        for child in decay[1].children:
            if "pi0" in child.name:
                assert not child.has_fixed_mass

    check_norm(container, n_events=1)
    check_norm(container, n_events=100)


def test_big_decay():
    """Create a GenMultiDecay object from a large dict with many branches and subbranches."""
    container = GenMultiDecay.from_dict(example_decay_chains.dstarplus_big_decay)
    output_decays = container.gen_particles
    assert_almost_equal(sum(d[0] for d in output_decays), 1)
    check_norm(container, n_events=1)
    check_norm(container, n_events=100)
    # TODO add more asserts here


def test_mass_width_tolerance():
    """Test changing the MASS_WIDTH_TOLERANCE class variable."""
    GenMultiDecay.MASS_WIDTH_TOLERANCE = 1e-10
    output_decays = GenMultiDecay.from_dict(
        example_decay_chains.dplus_4grandbranches
    ).gen_particles
    for p in output_decays:
        gen_particle = p[1]  # Ignore probability
        assert gen_particle.children[-1].name[:3] == "pi0"
        # Check that particle_model_map has assigned the bw mass function to all pi0 decays.
        assert not gen_particle.children[-1].has_fixed_mass
    # Restore class variable to not affect other tests
    GenMultiDecay.MASS_WIDTH_TOLERANCE = 1e-10


def test_default_mass_func():
    """Test changing the DEFAULT_MASS_FUNC class variable."""
    GenMultiDecay.DEFAULT_MASS_FUNC = "bw"
    output_decays = GenMultiDecay.from_dict(
        example_decay_chains.dplus_4grandbranches, tolerance=1e-10
    ).gen_particles
    for p in output_decays:
        gen_particle = p[1]  # Ignore probability
        assert gen_particle.children[-1].name[:3] == "pi0"
        # Check that particle_model_map has assigned the bw mass function to all pi0 decays.
        assert gen_particle.children[-1]._mass.__name__ == "bw"

    # Restore class variable to not affect other tests
    GenMultiDecay.DEFAULT_MASS_FUNC = "bw"
