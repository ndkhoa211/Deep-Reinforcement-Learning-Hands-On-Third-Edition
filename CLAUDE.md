# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the code repository for "Deep Reinforcement Learning Hands-On, Third Edition" by Maxim Lapan (Packt Publishing). The book covers practical implementations of reinforcement learning algorithms from basic Q-learning and DQN to advanced methods like PPO and RLHF.

## Environment Setup

### Python Version
- Requires Python 3.11+ (tested with 3.11, project configured for >=3.12)
- Uses `uv` package manager (uv.lock present)

### Installation

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

Or using the pyproject.toml:
```bash
pip install -e .
```

Key dependencies:
- `gymnasium[atari,classic-control,accept-rom-license]` - RL environments
- `torch` and `torchvision` - Deep learning framework
- `ptan` - PyTorch Agent Net library for RL building blocks
- `stable-baselines3` - High-level RL library
- `torchrl` - TorchRL library
- `ray[tune]` - Hyperparameter tuning
- `tensorboard` - Training visualization
- `opencv-python` - Image processing
- `pytest` - Testing framework

### GPU Acceleration
The code is designed to leverage GPU acceleration. Training times can be 10-100x slower on CPU. Consider using:
- A CUDA-compatible GPU
- Cloud instances (AWS, GCP)
- Google Colab for free GPU access

## Repository Structure

The repository is organized by book chapters (Chapter02 through Chapter22), each containing:

- **Numbered scripts** (e.g., `01_cartpole.py`, `02_dqn_pong.py`) - Main examples demonstrating specific concepts
- **lib/** subdirectories - Reusable modules containing:
  - Model architectures (`dqn_model.py`, `models.py`)
  - Environment wrappers (`wrappers.py`, `environ.py`)
  - Common utilities (`common.py`)
  - Data processing (`data.py`)
- **tests/** subdirectories - Unit tests using pytest
- **adhoc/** or **attic/** subdirectories - Experimental or deprecated code
- **bench/** subdirectories - Benchmarking scripts
- **conftest.py** - pytest configuration (where present)

### Key Architectural Patterns

1. **Environment Wrappers**: Custom Gymnasium environment wrappers for preprocessing (e.g., frame stacking, reward scaling)

2. **Agent Abstractions**: Uses `ptan` library for agent implementations with separation of concerns:
   - Agent selection logic
   - Experience sources
   - Replay buffers
   - Target network synchronization

3. **Model Architectures**:
   - Convolutional networks for Atari games (DQN architecture)
   - Fully connected networks for simple environments
   - Actor-Critic architectures (A2C, A3C, PPO)
   - Specialized architectures per chapter topic

4. **Training Scripts**: Self-contained scripts that:
   - Initialize environments
   - Create models and optimizers
   - Implement training loops with TensorBoard logging
   - Often include tuning versions (e.g., `05_pong_pg_tune.py`)

## Running Examples

### Basic Execution
Most scripts are executable and can be run directly:
```bash
python Chapter06/02_dqn_pong.py
```

### Training with TensorBoard
Training scripts typically log to TensorBoard:
```bash
tensorboard --logdir=runs
```

### Running Tests
For chapters with tests (e.g., Chapter10, Chapter13):
```bash
cd Chapter10
pytest
```

Or run specific tests:
```bash
pytest Chapter10/tests/test_environ.py
```

### Type Checking
The code uses Python type annotations. Check types with:
```bash
mypy Chapter10/
```

## Common Development Workflows

### Understanding a Specific Algorithm
1. Locate the chapter corresponding to the algorithm (see README.md for chapter list)
2. Start with the simplest numbered script (e.g., `01_*.py`)
3. Review the lib/ modules for implementation details
4. Check tests/ for usage examples

### Modifying or Extending Code
1. Each chapter is largely self-contained with its own lib/ modules
2. Shared concepts (like DQN models) are reimplemented per chapter to show evolution
3. Some chapters have requirements.txt for additional dependencies (e.g., Chapter13 for TextWorld)

### Debugging Training Issues
1. Check TensorBoard logs for training metrics
2. Verify environment is returning expected observations
3. Use smaller networks or simpler environments first
4. GPU memory issues: reduce batch size or replay buffer size

## Chapter-Specific Notes

- **Chapter10 (Stock Trading)**: Uses custom stock market environment with price data in `data/` directory
- **Chapter13 (TextWorld)**: Requires TextWorld library, uses language models for text-based games
- **Chapter14 (Web Navigation)**: Uses Selenium for browser automation
- **Chapters with tuning**: Some chapters include `*_tune.py` scripts using Ray Tune for hyperparameter optimization

## Code Style

- Uses type annotations (`typing` module)
- PyTorch-style module definitions
- Follows Gymnasium API conventions
- TensorBoard integration for logging
- pytest for unit tests where applicable
