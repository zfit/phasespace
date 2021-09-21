from example_decay_chains import *  # TODO remove * since it is bad practice?
from numpy.testing import assert_almost_equal

from phasespace.fulldecay import FullDecay


def check_norm(full_decay: FullDecay, **kwargs) -> list[tuple]:
    """Checks whether the normalize_weights argument works for FullDecay.generate.

    Parameters
    ----------
    full_decay : FullDecay
        full_decay.generate will be called.
    kwargs
        Additional parameters passed to generate.

    Returns
    -------
    list[tuple]
        All the values returned by generate, both times.
        The return arguments from normalize_weights=True is the first element in the returned list.

    Notes
    -----
    The function is called check_norm instead of test_norm since it is used by other functions and is not a stand-alone test.
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


def test_single_chain():
    """Test converting a decaylanguage dict with only one possible decay."""
    container = FullDecay.from_dict(dplus_single, tolerance=1e-10)
    output_decay = container.gen_particles
    assert len(output_decay) == 1
    prob, gen = output_decay[0]
    assert_almost_equal(prob, 1)
    assert gen.name == "D+"
    assert {p.name for p in gen.children} == {"K-", "pi+", "pi+ [0]", "pi0"}
    for p in gen.children:
        if "pi0" == p.name[:3]:
            assert not p.has_fixed_mass
        else:
            assert p.has_fixed_mass

    check_norm(container, n_events=1)
    (normed_weights, decay_list), _ = check_norm(container, n_events=100)
    assert len(decay_list) == 1
    events = decay_list[0]
    assert set(events.keys()) == {"K-", "pi+", "pi+ [0]", "pi0", "gamma", "gamma [0]"}
    assert all(len(p) == 100 for p in events.values())


def test_branching_children():
    container = FullDecay.from_dict(pi0_4branches, tolerance=1e-10)
    output_decays = container.gen_particles
    assert len(output_decays) == 4
    assert_almost_equal(sum(d[0] for d in output_decays), 1)
    check_norm(container, n_events=1)
    (normed_weights, events), _ = check_norm(container, n_events=100)


def test_branching_grandchilden():
    container = FullDecay.from_dict(dplus_4grandbranches)
    output_decays = container.gen_particles
    assert_almost_equal(sum(d[0] for d in output_decays), 1)
    check_norm(container, n_events=1)
    (normed_weights, events), _ = check_norm(container, n_events=100)
    # TODO add more asserts here


def test_big_decay():
    container = FullDecay.from_dict(dstarplus_big_decay)
    output_decays = container.gen_particles
    assert_almost_equal(sum(d[0] for d in output_decays), 1)
    check_norm(container, n_events=1)
    (normed_weights, events), _ = check_norm(container, n_events=100)
    # TODO add more asserts here
