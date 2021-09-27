import itertools
from typing import Callable, Union

import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from particle import Particle

from phasespace import GenParticle

from .mass_functions import _DEFAULT_CONVERTER

_MASS_WIDTH_TOLERANCE = 0.01
_DEFAULT_MASS_FUNC = "rel-BW"


class FullDecay:
    def __init__(self, gen_particles: list[tuple[float, GenParticle]]):
        """
        A container that works like GenParticle that can handle multiple decays. Can be created from

        Parameters
        ----------
        gen_particles : list[tuple[float, GenParticle]]
            All the GenParticles and their corresponding probabilities.
            The list must be of the format [[probability, GenParticle instance], [probability, ...
        Notes
        -----
        Input format might change
        """
        self.gen_particles = gen_particles

    @classmethod
    def from_dict(
        cls,
        dec_dict: dict,
        mass_converter: dict[str, Callable] = None,
        tolerance: float = _MASS_WIDTH_TOLERANCE,
    ):
        """Create a FullDecay instance from a dict in the decaylanguage format.

        Parameters
        ----------
        dec_dict : dict
            The input dict from which the FullDecay object will be created from.
        mass_converter : dict[str, Callable]
            A dict with mass function names and their corresponding mass functions.
            These functions should take the average particle mass and the mass width as inputs
            and return a mass function that phasespace can understand.
            This dict will be combined with the predefined mass functions in this package.
        tolerance : float
            Minimum mass width of the particle to use a mass function instead of assuming the mass to be constant.

        Returns
        -------
        FullDecay
            The created FullDecay object.
        """
        if mass_converter is None:
            total_mass_converter = _DEFAULT_CONVERTER
        else:
            # Combine the mass functions specified by the package to the mass functions specified from the input.
            total_mass_converter = {**_DEFAULT_CONVERTER, **mass_converter}

        gen_particles = _recursively_traverse(
            dec_dict, total_mass_converter, tolerance=tolerance
        )
        return cls(gen_particles)

    def generate(
        self, n_events: int, normalize_weights: bool = False, **kwargs
    ) -> Union[
        tuple[list[tf.Tensor], list[tf.Tensor]],
        tuple[list[tf.Tensor], list[tf.Tensor], list[tf.Tensor]],
    ]:
        """Generate four-momentum vectors from the decay(s).

        Parameters
        ----------
        n_events : int
            Total number of events combined, for all the decays.
        normalize_weights : bool
            Normalize weights according to all events generated. This also changes the return values.
            See the phasespace documentation for more details.
        kwargs
            Additional parameters passed to all calls of GenParticle.generate

        Returns
        -------
        The arguments returned by GenParticle.generate are returned. See the phasespace documentation for details.
        However, instead of being 2 or 3 tensors, it is 2 or 3 lists of tensors, each entry in the lists corresponding
        to the return arguments from the corresponding GenParticle instances in self.gen_particles.
        Note that when normalize_weights is True, the weights are normalized to the maximum of all returned events.
        """
        # Input to tf.random.categorical must be 2D
        rand_i = tf.random.categorical(
            tnp.log([[dm[0] for dm in self.gen_particles]]), n_events
        )
        # Input to tf.unique_with_counts must be 1D
        dec_indices, _, counts = tf.unique_with_counts(rand_i[0])
        counts = tf.cast(counts, tf.int64)
        weights, max_weights, events = [], [], []
        for i, n in zip(dec_indices, counts):
            weight, max_weight, four_vectors = self.gen_particles[i][1].generate(
                n, normalize_weights=False, **kwargs
            )
            weights.append(weight)
            max_weights.append(max_weight)
            events.append(four_vectors)

        if normalize_weights:
            total_max = tnp.max([tnp.max(mw) for mw in max_weights])
            normed_weights = [w / total_max for w in weights]
            return normed_weights, events

        return weights, max_weights, events


def _unique_name(name: str, preexisting_particles: set[str]) -> str:
    """Create a string that does not exist in preexisting_particles based on name.

    Parameters
    ----------
    name : str
        Name that should be
    preexisting_particles : set[str]
        Preexisting names

    Returns
    -------
    name : str
        Will be `name` if `name` is not in preexisting_particles or of the format "name [i]" where i will begin at 0
        and increase until the name is not preexisting_particles.
    """
    if name not in preexisting_particles:
        preexisting_particles.add(name)
        return name

    name += " [0]"
    i = 1
    while name in preexisting_particles:
        name = name[: name.rfind("[")] + f"[{str(i)}]"
        i += 1
    preexisting_particles.add(name)
    return name


def _get_particle_mass(
    name: str,
    mass_converter: dict[str, Callable],
    mass_func: str,
    tolerance: float = _MASS_WIDTH_TOLERANCE,
) -> Union[Callable, float]:
    """
    Get mass or mass function of particle using the particle package.
    Parameters
    ----------
    name : str
        Name of the particle. Name must be recognizable by the particle package.
    tolerance : float
        See _recursively_traverse

    Returns
    -------
    Callable, float
        Returns a function if the mass has a width smaller than tolerance.
        Otherwise, return a constant mass.
    TODO try to cache results for this function in the future for speedup.
    """
    particle = Particle.from_evtgen_name(name)

    if particle.width <= tolerance:
        return tf.cast(particle.mass, tf.float64)
    # If name does not exist in the predefined mass distributions, use Breit-Wigner
    return mass_converter[mass_func](mass=particle.mass, width=particle.width)


def _recursively_traverse(
    decaychain: dict,
    mass_converter: dict[str, Callable],
    preexisting_particles: set[str] = None,
    tolerance: float = _MASS_WIDTH_TOLERANCE,
) -> list[tuple[float, GenParticle]]:
    """Create all possible GenParticles by recursively traversing a dict from decaylanguage.

    Parameters
    ----------
    decaychain: dict
        Decay chain with the format from decaylanguage
    preexisting_particles : set
        names of all particles that have already been created.
    tolerance : float
        Minimum mass width for a particle to set a non-constant mass to a particle.

    Returns
    -------
    list[tuple[float, GenParticle]]
        The generated particle
    """
    # Get the only key inside the decaychain dict
    original_mother_name, = decaychain.keys()

    if preexisting_particles is None:
        preexisting_particles = set()
        is_top_particle = True
    else:
        is_top_particle = False

    # This is in the form of dicts
    decay_modes = decaychain[original_mother_name]
    mother_name = _unique_name(original_mother_name, preexisting_particles)
    # This will contain GenParticle instances and their probabilities
    all_decays = []
    for dm in decay_modes:
        dm_probability = dm["bf"]
        daughter_particles = dm["fs"]
        daughter_gens = []

        for daughter_name in daughter_particles:
            if isinstance(daughter_name, str):
                # Always use constant mass for stable particles
                daughter = GenParticle(
                    _unique_name(daughter_name, preexisting_particles),
                    Particle.from_evtgen_name(daughter_name).mass,
                )
                daughter = [(1.0, daughter)]
            elif isinstance(daughter_name, dict):
                daughter = _recursively_traverse(
                    daughter_name,
                    mass_converter,
                    preexisting_particles,
                    tolerance=tolerance,
                )
            else:
                raise TypeError(
                    f'Expected elements in decaychain["fs"] to only be str or dict '
                    f"but found an instance of type {type(daughter_name)}"
                )
            daughter_gens.append(daughter)

        for daughter_combination in itertools.product(*daughter_gens):
            p = tnp.prod([decay[0] for decay in daughter_combination]) * dm_probability
            if is_top_particle:
                mother_mass = Particle.from_evtgen_name(original_mother_name).mass
            else:
                mother_mass = _get_particle_mass(
                    original_mother_name,
                    mass_converter=mass_converter,
                    mass_func=dm.get("zfit", _DEFAULT_MASS_FUNC),
                    tolerance=tolerance,
                )

            one_decay = GenParticle(mother_name, mother_mass).set_children(
                *(decay[1] for decay in daughter_combination)
            )
            all_decays.append((p, one_decay))

    return all_decays
