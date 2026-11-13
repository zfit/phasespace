"""Skip the fromdecay tests when the ``fromdecay`` extra is not installed.

The extra depends on TensorFlow, which is not available on every Python version that phasespace supports.
"""

from importlib.util import find_spec

collect_ignore_glob = []
if any(find_spec(name) is None for name in ("decaylanguage", "tensorflow", "zfit")):
    collect_ignore_glob.append("test_*.py")
