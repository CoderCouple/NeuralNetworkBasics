# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational project for building neural networks from scratch using pure Python and NumPy. No deep learning frameworks (TensorFlow, PyTorch) are used — everything is implemented manually for learning purposes.

## Setup and Running

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
cd src && pip install -e .

# Run a chapter example (e.g., chapter 1)
python src/ch1_neuron/01_single_neuron.py
```

No test framework or linter is currently configured.

## Architecture

- **src/**: Main package (`neuralnetworkbasics`), installable via `setup.py`
- **Chapter-based organization**: Each chapter lives in its own subpackage (e.g., `ch1_neuron/`), meant to progressively build up neural network concepts
- **NumPy-only**: All numerical operations use NumPy — the sole dependency
- **Python 3.9+** required
