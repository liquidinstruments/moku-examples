# Contributing to Moku Examples

Thank you for your interest in contributing to the Moku Examples repository! This guide will help you set up your development environment and understand our contribution workflow.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Code Quality Standards](#code-quality-standards)
- [Contribution Workflow](#contribution-workflow)
- [Example Guidelines](#example-guidelines)
- [Testing Your Changes](#testing-your-changes)

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A Moku device for testing
- Basic familiarity with the Moku API

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/moku-examples.git
   cd moku-examples
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/liquidinstruments/moku-examples.git
   ```

## Development Environment Setup

### 1. Install Python Dependencies

Install the base Moku package and development dependencies:

```bash
# Install the Moku API
pip install moku

# Install optional dependencies for specific examples
pip install 'moku[neuralnetwork]'  # For neural network examples

# Install development dependencies (type stubs, etc.)
pip install -e ".[dev]"
```

### 2. Install uv/uvx (Recommended)

We use `uvx` (part of the `uv` toolchain) to run linting and type-checking tools in the pre-commit hooks. This ensures consistent tool versions without polluting your global Python environment.

#### What is uv?

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver written in Rust. `uvx` is its command runner that can execute tools in isolated environments.

#### Install uv

Choose your platform:

**macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip (cross-platform):**
```bash
pip install uv
```

**Using Homebrew (macOS):**
```bash
brew install uv
```

#### Verify Installation

```bash
# Check that uv is installed
uv --version

# Check that uvx is available
uvx --version
```

Both commands should show version information (e.g., `uv 0.5.0`).

#### Why use uv/uvx?

- **Isolation**: Tools run in isolated environments without conflicting with your project dependencies
- **Speed**: Much faster than traditional pip-based workflows
- **Consistency**: Everyone uses the same tool versions specified in the hooks
- **Convenience**: No need to manually install ruff, mypy, etc.

### 3. Install Pre-commit Hooks

We use pre-commit hooks to maintain code quality. **This step is required for all contributors.**

#### Install pre-commit

```bash
pip install pre-commit
```

#### Set up the hooks

```bash
# Install the pre-commit hooks into your git repository
pre-commit install
```

This will automatically run code quality checks before each commit using `uvx` to execute the tools.

#### Manual Hook Execution

You can manually run the hooks on all files at any time:

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files only
pre-commit run --files python-api/oscilloscope_basic.py
```

### 4. Install Additional Tools (Optional)

#### For Linting and Type Checking

The pre-commit hooks use `ruff` and `mypy`, which are installed automatically via `uvx` when the hooks run. However, you can install them locally for IDE integration:

```bash
# Install ruff for linting and formatting
pip install ruff

# Install mypy for type checking
pip install mypy

# Install type stubs for OpenCV (if working with neural network examples)
pip install opencv-stubs
```

#### For Jupyter Notebooks

If you're working with notebook examples:

```bash
pip install jupyter notebook ipykernel
```

### 5. Verify Your Setup

Test that everything is working:

```bash
# Run the pre-commit hooks
pre-commit run --all-files

# Should see output like:
# ruff-check...............................................................Passed
# ruff-format..............................................................Passed
# mypy.....................................................................Passed
```

## Code Quality Standards

We enforce code quality using automated tools. All contributions must pass these checks.

### Pre-commit Hooks

The repository uses three pre-commit hooks:

1. **ruff-check**: Linting (checks for code issues)
   - Checks: pycodestyle errors (E), warnings (W), pyflakes (F), import sorting (I)
   - Auto-fixes: Import sorting, some code issues

2. **ruff-format**: Code formatting
   - Enforces consistent code style
   - Auto-formats: Line length (110 chars), quotes, spacing

3. **mypy**: Type checking
   - Checks: Type annotations and type consistency
   - Python version: 3.9+

### Configuration

Code quality settings are defined in `pyproject.toml`. Key settings:

```toml
[tool.ruff]
line-length = 110
target-version = "py38"

[tool.mypy]
python_version = "3.9"
ignore_missing_imports = true  # Lenient for examples
```

### Common Issues and Fixes

#### Ambiguous variable names
```python
# ❌ Bad - single letter that looks like zero/one
O = np.pi / np.array([o for o in range(8, 64)])

# ✅ Good - descriptive name
omega = np.pi / np.array([o for o in range(8, 64)])
```

#### Bare except clauses
```python
# ❌ Bad - catches everything
try:
    data = device.get_data()
except:
    data = None

# ✅ Good - specific exception types
try:
    data = device.get_data()
except (KeyError, IndexError):
    data = None
```

#### Boolean comparisons
```python
# ❌ Bad - explicit comparison to False
while status == False:
    do_something()

# ✅ Good - pythonic boolean check
while not status:
    do_something()
```

#### Type hints for matplotlib
```python
# ❌ Bad - list instead of tuple
ax.set_xlim([0, 10])

# ✅ Good - tuple for axis limits
ax.set_xlim((0, 10))
```

## Contribution Workflow

### 1. Create a Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-oscilloscope-fft-example`
- `fix/phasemeter-streaming-bug`
- `docs/improve-neural-network-readme`

### 2. Make Your Changes

- Write clear, well-commented code
- Follow existing code style in the repository
- Test your code with actual Moku hardware if possible
- Update documentation as needed

### 3. Commit Your Changes

The pre-commit hooks will run automatically:

```bash
git add .
git commit -m "Add oscilloscope FFT example"
```

If the hooks fail:
1. Review the error messages
2. Fix the issues (some are auto-fixed)
3. Stage the changes again: `git add .`
4. Commit again

### 4. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description of what the PR does
- Any testing performed
- Screenshots/plots if applicable

## Example Guidelines

### File Naming

Use descriptive, lowercase names with underscores:
- `oscilloscope_fft_analysis.py`
- `waveformgenerator_swept_sine.py`
- `neural_network_pid_control.ipynb`

### Code Structure

#### Python Scripts (.py)
```python
# Brief description of what the example does
#
# (c) 2024 Liquid Instruments Pty. Ltd.

from moku.instruments import Oscilloscope

# Configuration constants
MOKU_IP = '192.168.1.100'  # Update with your device IP
SAMPLE_RATE = '1MSa/s'

try:
    # Connect to device
    osc = Oscilloscope(MOKU_IP, force_connect=True)

    # Configure instrument
    osc.set_timebase(0, 1e-3)

    # Your example code here

finally:
    # Always close the connection
    osc.relinquish_ownership()
```

#### Jupyter Notebooks (.ipynb)
- Start with a markdown cell explaining the purpose
- Use markdown headers to organize sections
- Include visualization outputs
- Add explanatory text between code cells
- Clear all outputs before committing (optional, but keeps diffs clean)

### Documentation in Examples

Each example should include:
1. **Header comment**: Brief description and date
2. **Inline comments**: Explain non-obvious code
3. **Configuration section**: Clearly marked parameters users need to change
4. **Error handling**: Use try/finally for cleanup
5. **Output**: Print relevant information or generate plots

### IP Address Handling

Never commit your actual device IP. Use placeholder or example IPs:
```python
# ✅ Good
MOKU_IP = '192.168.1.100'  # Update with your device IP

# ✅ Also good
MOKU_IP = '192.168.###.###'  # Replace with your device IP

# ❌ Bad (reveals your network)
MOKU_IP = '192.168.50.247'
```

## Testing Your Changes

### Manual Testing

1. **Test with actual hardware** if possible
2. **Verify the example runs** without errors
3. **Check outputs** (plots, data files, console output)
4. **Test error conditions** (e.g., wrong IP, disconnected device)

### Pre-commit Testing

Always run before pushing:
```bash
# Test all files
pre-commit run --all-files

# Test specific files you changed
pre-commit run --files python-api/your_new_example.py
```

### Notebook Testing

For Jupyter notebooks:
```bash
# Run the notebook and check for errors
jupyter nbconvert --to notebook --execute your_notebook.ipynb
```

## Questions?

- **Technical questions**: support@liquidinstruments.com
- **Contribution questions**: Open an issue on GitHub
- **General discussion**: [Liquid Instruments Forum](https://forum.liquidinstruments.com/)

## Code of Conduct

- Be respectful and constructive
- Help others learn
- Focus on what is best for the community
- Show empathy towards other contributors

---

Thank you for contributing to Moku Examples! Your contributions help others learn and build with Moku devices.
