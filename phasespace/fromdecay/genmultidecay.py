from __future__ import annotations

import itertools
from collections.abc import Callable

import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from particle import Particle

from phasespace import GenParticle

from .mass_functions import DEFAULT_CONVERTER


class GenMultiDecay:
    MASS_WIDTH_TOLERANCE = 0.01
    DEFAULT_MASS_FUNC = "relbw"

    def __init__(self, gen_particles: list[tuple[float, GenParticle]]):
        """A `GenParticle`-type container that can handle multiple decays.

        Args:
            gen_particles: All the GenParticles and their corresponding probabilities.
                The list must be of the format [[probability, GenParticle instance], [probability, ...
        """
        self.gen_particles = gen_particles

    @classmethod
    def from_dict(
        cls,
        dec_dict: dict,
        mass_converter: dict[str, Callable] = None,
        tolerance: float = None,
        particle_model_map: dict[str, str] = None,
    ):
        """Create a `GenMultiDecay` instance from a dict in the `DecayLanguage` package format.

        Args:
            dec_dict:
                The input dict from which the GenMultiDecay object will be created from.
                A dec_dict has the same structure as the dicts used in DecayLanguage, see the examples below.
            mass_converter: A dict with mass function names and their corresponding mass functions.
                These functions should take the particle mass and the mass width as inputs
                and return a mass function that phasespace can understand.
                This dict will be combined with the predefined mass functions in this package.
                See the Example below or the tutorial for how to use this parameter.
            tolerance: Minimum mass width of the particle to use a mass function instead of
                assuming the mass to be constant. If None, the default value, defined by the class variable
                MASS_WIDTH_TOLERANCE, will be used. This value can be customized if desired.
            particle_model_map: A dict where the key is a particle name and the value is a mass function name.
                If a particle is specified in the particle_model_map, then all appearances of this particle
                in dec_dict will get the same mass function. This way, one does not have to manually add the
                zfit parameter to every place where this particle appears in dec_dict. If the zfit parameter
                is specified for a particle which is also included in particle_model_map, the zfit parameter
                mass function name will be prioritized.
        Returns:
            The created GenMultiDecay object.

        Examples:
            DecayLanguage is usually used to create a dict that can be understood by GenMultiDecay.
            >>> from decaylanguage import DecFileParser
            >>> from phasespace.fromdecay import GenMultiDecay

            Parse a .dec file to create a DecayLanguage dict describing a D*+ particle
            that can decay in 2 different ways: D*+ -> D0(->K- pi+) pi+ or D*+ -> D+ gamma.

            >>> parser = DecFileParser('example_decays.dec')
            >>> parser.parse()
            >>> dst_chain = parser.build_decay_chains("D*+")
            >>> dst_chain
            {'D*+': [{'bf': 0.984,
                'fs': [{'D0': [{'bf': 1.0,
                        'fs': ['K-', 'pi+']}]},
                    'pi+']},
                {'bf': 0.016,
                 'fs': ['D+', 'gamma']}]}

            If the D0 particle should have a mass distribution of a gaussian when it decays, one can pass the
            `particle_model_map` parameter to `from_dict`:
            >>> dst_gen = GenMultiDecay.from_dict(dst_chain, particle_model_map={"D0": "gauss"})
            This will then set the mass function of D0 to a gaussian for all its decays.

            If more custom control is required, e.g., if D0 can decay in multiple ways and one of the decays
            should have a specific mass function, a `zfit` parameter can be added to its decay dict:
            >>> dst_chain["D*+"][0]["fs"][0]["D0"][0]["zfit"] = "gauss"
            >>> dst_chain
            {'D*+': [{'bf': 0.984,
                'fs': [{'D0': [{'bf': 1.0,
                        'fs': ['K-', 'pi+'],
                        'zfit': 'gauss'}]},
                    'pi+']},
                {'bf': 0.016,
                'fs': ['D+', 'gamma']}]}
            This dict can then be passed to `GenMultiDecay.from_dict`:
            >>> dst_gen = GenMultiDecay.from_dict(dst_chain)
            This will now convert make the D0 particle have a gaussian mass function, only when it decays into
            K- and pi+. In this case, there are no other ways that D0 can decay, so using `particle_model_map`
            is a cleaner and easier option.

            If the decay of the D0 particle should be modelled by a mass distribution that does not
            come with the package, a custom distribution can be created:
            >>> def custom_gauss(mass, width):
            >>>     particle_mass = tf.cast(mass, tf.float64)
            >>>     particle_width = tf.cast(width, tf.float64)
            >>>     def mass_func(min_mass, max_mass, n_events):
            >>>         min_mass = tf.cast(min_mass, tf.float64)
            >>>         max_mass = tf.cast(max_mass, tf.float64)
            >>>         # Use a zfit PDF
            >>>         pdf = zfit.pdf.Gauss(mu=particle_mass, sigma=particle_width, obs="")
            >>>         iterator = tf.stack([min_mass, max_mass], axis=-1)
            >>>         return tf.vectorized_map(
            >>>             lambda lim: pdf.sample(1, limits=(lim[0], lim[1])), iterator
            >>>         )
            >>>     return mass_func

            Once again change the distribution in the `dst_chain` dict. Here, it is changed to "custom_gauss"
            but any name can be used.
            >>> dst_chain["D*+"][0]["fs"][0]["D0"][0]["zfit"] = "custom_gauss"

            One can then pass the `custom_gauss` function and its name (in this case "custom_gauss") as a
            `dict`to `from_dict` as the mass_converter parameter:
            >>> dst_gen = GenMultiDecay.from_dict(dst_chain, mass_converter={"custom_gauss": custom_gauss})

        Notes:
            For a more in-depth tutorial, see the tutorial on GenMultiDecay in the
            [documentation](https://phasespace.readthedocs.io/en/stable/GenMultiDecay_Tutorial.html).
        """
        if tolerance is None:
            tolerance = cls.MASS_WIDTH_TOLERANCE
        if particle_model_map is None:
            particle_model_map = {}
        if mass_converter is None:
            total_mass_converter = DEFAULT_CONVERTER
        else:
            # Combine the default mass functions specified with the mass functions from input.
            total_mass_converter = {**DEFAULT_CONVERTER, **mass_converter}

        gen_particles = _recursively_traverse(
            dec_dict,
            total_mass_converter,
            particle_model_map=particle_model_map,
            tolerance=tolerance,
        )
        return cls(gen_particles)

    def generate(
        self, n_events: int, normalize_weights: bool = True, **kwargs
    ) -> tuple[list[tf.Tensor], list[tf.Tensor]] | tuple[
        list[tf.Tensor], list[tf.Tensor], list[tf.Tensor]
    ]:
        """Generate four-momentum vectors from the decay(s).

        Args:
            n_events: Total number of events combined, for all the decays.
            normalize_weights: Normalize weights according to all events generated.
                This also changes the return values. See the phasespace documentation for more details.
            kwargs: Additional parameters passed to all calls of GenParticle.generate

        Returns:
            The arguments returned by GenParticle.generate are returned. See the phasespace documentation for
            details. However, instead of being 2 or 3 tensors, it is 2 or 3 lists of tensors,
            each entry in the lists corresponding to the return arguments from the corresponding GenParticle
            instances in self.gen_particles. Note that when normalize_weights is True,
            the weights are normalized to the maximum of all returned events.
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

    Args:
        name: Original name
        preexisting_particles: Names that the particle cannot have as name.

    Returns:
        name: Will be `name` if `name` is not in preexisting_particles or of the format "name [i]" where i
            begins at 0 and increases until the name is not preexisting_particles.
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
    tolerance: float,
) -> Callable | float:
    """Get mass or mass function of particle using the particle package.

    Args:
        name: Name of the particle. Name must be recognizable by the particle package.
        mass_converter: See _recursively_traverse
        mass_func: See the name of the mass function, e.g., 'rel-bw'. Must be a valid key for mass_converter.
        tolerance: See _recursively_traverse

    Returns:
        A function if the mass has a width smaller than tolerance. Otherwise, return a constant mass.
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
    particle_model_map: dict[str, str],
    tolerance: float,
    preexisting_particles: set[str] = None,
) -> list[tuple[float, GenParticle]]:
    """Create all possible GenParticles by recursively traversing a dict from DecayLanguage, see Examples.

    Args:
        decaychain: Decay chain with the format from DecayLanguage
        mass_converter: Maps from mass function names to the actual callable mass function.
        particle_model_map: See GenMultiDecay.from_dict.
        preexisting_particles: Names of all particles that have already been created.
        tolerance: Minimum mass width for a particle to set a non-constant mass to a particle.
        If None, set to default value, given by GenMultiDecay.MASS_WIDTH_TOLERANCE

    Returns:
        The generated GenParticle instances, one for each possible way of the decay.
    """
    if tolerance is None:
        tolerance = GenMultiDecay.MASS_WIDTH_TOLERANCE
    # Get the only key inside the decaychain dict
    (original_mother_name,) = decaychain.keys()

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
                    particle_model_map,
                    tolerance,
                    preexisting_particles,
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
                if "zfit" in dm:
                    mass_func = dm["zfit"]
                elif original_mother_name in particle_model_map:
                    mass_func = particle_model_map[original_mother_name]
                else:
                    mass_func = GenMultiDecay.DEFAULT_MASS_FUNC

                mother_mass = _get_particle_mass(
                    original_mother_name,
                    mass_converter=mass_converter,
                    mass_func=mass_func,
                    tolerance=tolerance,
                )

            one_decay = GenParticle(mother_name, mother_mass).set_children(
                *(decay[1] for decay in daughter_combination)
            )
            all_decays.append((p, one_decay))

    return all_decays
