# Chapter 03 - Fixes Summary

## Overview
This document summarizes the fixes applied to make the Chapter 03 code compatible with Gymnasium (the successor to OpenAI Gym).

## Files Fixed
1. `03_atari_gan.py` - Atari GAN training script
2. `dl_pytorch.ipynb` - Deep Learning with PyTorch notebook

## Issues and Solutions

### Issue 1: Logger API Incompatibility

**Problem:**
The old OpenAI Gym library had `gym.logger.set_level()` method, but Gymnasium removed this function. The original code used:
```python
log = gym.logger
log.set_level(gym.logger.INFO)
```

This caused an `AttributeError: module 'gymnasium.logger' has no attribute 'set_level'`.

**Solution:**
Directly set the `min_level` attribute on the gymnasium logger:
```python
# Configure gymnasium logger - set minimum level directly
gym.logger.min_level = gym.logger.WARN  # Use gym.logger.ERROR for less verbose output
```

**Available logging levels:**
- `gym.logger.WARN = 30` (default, shows warnings and errors)
- `gym.logger.ERROR = 40` (only shows errors)

**Additional change:**
Replaced `log.info()` calls with standard `print()` statements:
```python
# Before
log.info("Iter %d in %.2fs: gen_loss=%.3e, dis_loss=%.3e", ...)

# After
print(f"Iter {iter_no} in {dt:.2f}s: gen_loss={np.mean(gen_losses):.3e}, dis_loss={np.mean(dis_losses):.3e}")
```

### Issue 2: Atari Environments Not Registered

**Problem:**
When trying to create Atari environments (Breakout-v4, AirRaid-v4, Pong-v4), the script failed with:
```
gymnasium.error.NameNotFound: Environment `Breakout` doesn't exist.
```

This is because Gymnasium requires explicit registration of ALE (Arcade Learning Environment) environments.

**Solution:**
Added imports and registration before using the environments:
```python
# Register Atari environments
import ale_py
gym.register_envs(ale_py)
```

**Note:** This requires `ale-py` to be installed (which is included in `gymnasium[atari]`).

## Complete Fixed Code Section

### In 03_atari_gan.py (lines 15-25):
```python
import gymnasium as gym
from gymnasium import spaces

import numpy as np

# Register Atari environments
import ale_py
gym.register_envs(ale_py)

# Configure gymnasium logger - set minimum level directly
gym.logger.min_level = gym.logger.WARN  # Use gym.logger.ERROR for less verbose output
```

## Testing

The script was tested and confirmed working:
```bash
cd Chapter03
python 03_atari_gan.py
```

Output shows successful initialization:
```
A.L.E: Arcade Learning Environment (version 0.11.2+ecc1138)
[Powered by Stella]
```

The training loop successfully:
- Creates the three Atari environments (Breakout, AirRaid, Pong)
- Collects gameplay observations
- Trains the GAN discriminator and generator networks
- Logs progress every 100 iterations
- Saves generated images every 1000 iterations to TensorBoard

## Dependencies

Ensure these packages are installed:
- `gymnasium` (version 1.2.1 or compatible)
- `ale-py` (version 0.11.2 or compatible)
- `torch` and `torchvision`
- `opencv-python` (cv2)
- `tensorboard`

Install with:
```bash
pip install gymnasium[atari,accept-rom-license] torch torchvision opencv-python tensorboard
```

## Notes

- The Gymnasium library is a maintained fork of the original OpenAI Gym
- The logger API was simplified in Gymnasium, removing some methods like `set_level()`
- Atari environments require explicit registration via `gym.register_envs(ale_py)`
- Training the GAN is computationally intensive and benefits from GPU acceleration
