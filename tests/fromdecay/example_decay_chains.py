import os.path

from decaylanguage import DecayChain, DecayMode, DecFileParser

script_dir = os.path.dirname(os.path.abspath(__file__))

dfp = DecFileParser(f"{script_dir}/example_decays.dec")
dfp.parse()

# D+ particle with only one way of decaying
dplus_decay = DecayMode(1, "K- pi+ pi+ pi0", model="PHSP", zfit="relbw")
pi0_decay = DecayMode(1, "gamma gamma", zfit="relbw")
dplus_single = DecayChain("D+", {"D+": dplus_decay, "pi0": pi0_decay}).to_dict()

# pi0 particle that can decay in 4 possible ways
pi0_4branches = dfp.build_decay_chains("pi0")

# D+ particle that decays into 4 particles, out of which one particle in turn decays in 4 different ways.
dplus_4grandbranches = dfp.build_decay_chains("D+")

# D*+ particle that has multiple children, grandchild particles, many of which can decay in multiple ways.
dstarplus_big_decay = dfp.build_decay_chains("D*+")
