# Building and Publishing DocUVerse

This document describes how to build and publish the DocUVerse wheel package.

## Prerequisites

Install the required build tools:

```bash
pip install build twine
```

## Building the Wheel

### Quick Build

Simply run:

```bash
./scripts/build_wheel.sh
```

This will:
1. Clean previous build artifacts
2. Build the wheel
3. Verify the wheel contents
4. Display the built wheel location

### Build and Verify

To build and perform comprehensive verification:

```bash
./scripts/build_wheel.sh --verify
```

This performs additional checks to ensure all required packages are included.

### Clean Only

To clean build artifacts without building:

```bash
./scripts/build_wheel.sh --clean
```

## Publishing to PyPI

### Test on TestPyPI First (Recommended)

Before publishing to production PyPI, test your package on TestPyPI:

```bash
./scripts/build_wheel.sh --test
```

This will:
1. Build the wheel
2. Verify contents
3. Upload to TestPyPI (you'll need credentials)

**Setup TestPyPI credentials:**

Option 1 - Interactive (will prompt):
```bash
./scripts/build_wheel.sh --test
```

Option 2 - Using environment variables:
```bash
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-xxxxxxxxxxxxxxxxxxxxx"
./scripts/build_wheel.sh --test
```

Option 3 - Using `.pypirc` file:
```ini
# ~/.pypirc
[testpypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxx
```

**Test installation from TestPyPI:**
```bash
pip install -i https://test.pypi.org/simple/ docuverse==0.0.13
```

### Publish to Production PyPI

Once you've verified the package works on TestPyPI:

```bash
./scripts/build_wheel.sh --upload
```

**IMPORTANT:** This uploads to production PyPI. You'll be asked to confirm before proceeding.

**Setup PyPI credentials:**

Same options as TestPyPI:

```bash
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-xxxxxxxxxxxxxxxxxxxxx"
./scripts/build_wheel.sh --upload
```

Or configure `~/.pypirc`:
```ini
# ~/.pypirc
[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxx
```

## Getting API Tokens

### TestPyPI Token
1. Register/login at https://test.pypi.org
2. Go to https://test.pypi.org/manage/account/token/
3. Create a new API token
4. Copy the token (starts with `pypi-`)

### PyPI Token
1. Register/login at https://pypi.org
2. Go to https://pypi.org/manage/account/token/
3. Create a new API token
4. Copy the token (starts with `pypi-`)

## Installation

### Basic Installation

```bash
pip install docuverse
```

### With Optional Dependencies

```bash
# Install with FAISS support
pip install docuverse[faiss]

# Install with Milvus support
pip install docuverse[milvus]

# Install with ChromaDB support
pip install docuverse[chromadb]

# Install with Elasticsearch support
pip install docuverse[elastic]

# Install with extra tools (scikit-learn, matplotlib, etc.)
pip install docuverse[extra]

# Install everything
pip install docuverse[all]
```

## Versioning

The package version is read from the `VERSION` file. To release a new version:

1. Update the `VERSION` file:
   ```bash
   echo "0.0.14" > VERSION
   ```

2. Build and publish:
   ```bash
   ./scripts/build_wheel.sh --test   # Test first
   ./scripts/build_wheel.sh --upload # Then publish
   ```

## Troubleshooting

### Authentication Errors

If you get authentication errors:
- Verify your token is correct and not expired
- Ensure username is `__token__` (not your PyPI username)
- Check that you have upload permissions for the package

### Package Already Exists

PyPI doesn't allow re-uploading the same version. If you need to fix something:
1. Increment the version in `VERSION`
2. Rebuild and upload

### Build Failures

If the build fails:
- Check that `pyproject.toml` is valid
- Ensure all required files exist (VERSION, README.md, LICENSE)
- Run `./scripts/build_wheel.sh --clean` and try again

## Manual Build Process

If you prefer manual control:

```bash
# Clean
rm -rf dist/ build/ *.egg-info

# Build
python -m build --wheel

# Verify
python -m zipfile -l dist/*.whl

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Files in the Package

The wheel includes:
- Core package: `docuverse/`
- Retrieval engines: Elastic, Milvus, ChromaDB, FAISS
- Reranking engines: Dense, SPLADE, Cross-encoder
- Utilities: Evaluation, embeddings, calibration
- All required dependencies

## Dependencies

### Core Dependencies (always installed)
- torch, transformers, sentence-transformers
- numpy, scipy, pandas
- pyyaml, jinja2, grpcio
- tqdm, orjson, urllib3
- And more (see pyproject.toml)

### Optional Dependencies
- **faiss**: FAISS vector search
- **milvus**: Milvus vector database
- **chromadb**: ChromaDB vector database
- **elastic**: Elasticsearch integration
- **extra**: Additional tools (scikit-learn, matplotlib, etc.)
