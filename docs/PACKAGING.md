# PyMultiDIC packaging

PyMultiDIC is packaged as platform-specific wheels plus a source distribution.

For end users, the intended install command is:

```bash
pip install pymultidic
```

`pip` selects the wheel matching the user's operating system, CPU architecture,
and Python version. Because PyMultiDIC ships native C++ components, Linux,
Windows, and macOS each need their own wheel files.

## Release channels

Recommended release flow:

1. Publish wheels and the source distribution to PyPI so `pip install
   pymultidic` works normally.
2. Attach the same files to a GitHub Release for users who want to download
   artifacts manually.

GitHub Releases are useful for visibility and archival, but PyPI is the normal
index used by `pip install pymultidic`.

Do not store PyPI passwords in the repository or GitHub Secrets. Use PyPI
Trusted Publishing for GitHub Actions.

Trusted Publishing settings on PyPI:

- Project name: `pymultidic`
- Owner: `lbd-hfut`
- Repository name: `Multi-DIC`
- Workflow name: `wheels.yml`
- Environment name: `pypi`

After that is configured, pushing a tag such as `v0.1.0` builds wheels, creates
a GitHub Release, and publishes the same distributions to PyPI.

## Local build

```bash
python -m pip install --upgrade build scikit-build-core
python -m build
```

The wheel build compiles and installs:

- `native_recon3d` as a Python extension module.
- `ncorr_cli` under `pymultidic/bin/`.

## CI build

`.github/workflows/wheels.yml` uses `cibuildwheel` to build wheels on:

- Linux
- Windows
- macOS

The workflow uploads artifacts for pull requests and manual runs. For version
tags such as `v0.1.0`, it also creates a GitHub Release and publishes to PyPI
through Trusted Publishing.
