# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = phasespace
SOURCEDIR     = .
BUILDDIR      = _build

# Filter out TensorFlow/absl stderr warnings
FILTER_WARNINGS = 2>&1 | grep -v "All log messages before absl" | grep -v "Could not find cuda" | grep -v "cuFFT\|cuDNN\|cuBLAS\|computation placer" || true

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) $(FILTER_WARNINGS)
