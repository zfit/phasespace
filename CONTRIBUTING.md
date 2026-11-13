# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/zfit/phasespace/issues>.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

PhaseSpace could always use more documentation, whether as part of the official PhaseSpace docs, in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at <https://github.com/zfit/phasespace/issues>.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `phasespace` for local development.

1. Fork the `phasespace` repo on GitHub.

2. Clone your fork locally:

   ```shell
   git clone git@github.com:your_name_here/phasespace.git
   ```

3. Install your local copy into a virtual environment. With [uv](https://docs.astral.sh/uv), this is:

   ```shell
   cd phasespace/
   uv sync --extra all
   ```

   Alternatively, use `pip install -e .[all] --group dev` (pip 25.1 or newer) in a virtual environment of your choice.

4. Create a branch for local development:

   ```shell
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass pre-commit and the tests:

   ```shell
   pytest
   pre-commit run -a
   ```

6. Commit your changes and push your branch to GitHub:

   ```shell
   git add .
   git commit -m "Your detailed description of your changes."
   git push origin name-of-your-bugfix-or-feature
   ```

7. Submit a pull request through the GitHub website.

## Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org) from [MyST Markdown](https://myst-parser.readthedocs.io) files in the `docs` folder. Notebooks are committed without output (the [nbstripout](https://github.com/kynan/nbstripout) pre-commit hook takes care of that) and are executed by [MyST-NB](https://myst-nb.readthedocs.io) during the build. To build the documentation locally, run:

```shell
make -C docs html
```

and open `docs/_build/html/index.html`. Read the Docs publishes the documentation to <https://phasespace.readthedocs.io> and builds a preview for every pull request.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in `README.md`.
3. The pull request should work for all supported Python versions. Check <https://github.com/zfit/phasespace/actions/workflows/ci.yml> and make sure that the tests pass for all supported Python versions.

## Tips

To run a subset of tests (for example those in `tests/test_generate.py`):

```shell
pytest -k test_generate
```

## Deploying

A reminder for the maintainers on how to deploy. Make sure all your changes are committed (including an entry in `CHANGELOG.md`). Then run:

```shell
bumpversion patch # possible: major / minor / patch
git push
git push --tags
```

GitHub Actions will then deploy to PyPI if tests pass.
