from decaylanguage import DecayMode, DecayChain, DecFileParser
import os.path

script_dir = os.path.dirname(os.path.abspath(__file__))

dfp = DecFileParser(f"{script_dir}/example_decays.dec")
dfp.parse()

# D+ particle with only one way of decaying
dplus_decay = DecayMode(1, 'K- pi+ pi+ pi0', model='PHSP', zfit="relbw")
pi0_decay = DecayMode(1, 'gamma gamma')
dplus_single = DecayChain('D+', {'D+': dplus_decay, 'pi0': pi0_decay}).to_dict()

# pi0 particle that can decay in 4 possible ways
pi0_4branches = dfp.build_decay_chains('pi0')
# Specify different mass functions for the different decays of pi0
mass_functions = ["relbw", "bw", "gauss"]
for mass_function, decay_mode in zip(mass_functions, pi0_4branches["pi0"]):
    decay_mode["zfit"] = mass_function

# D+ particle that decays into 4 particles, out of which one particle in turn decays in 4 different ways.
dplus_4grandbranches = dfp.build_decay_chains("D+")

# D*+ particle that has multiple child particles, grandchild particles, many of which can decay in multiple ways.
dstarplus_big_decay = dfp.build_decay_chains("D*+")
