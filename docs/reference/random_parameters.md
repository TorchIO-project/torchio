# Random parameters

Many transforms accept a value, a range, or a distribution for their
randomizable parameters (for example `degrees`, `scales`, or `std`). You pass
the specification directly and the transform samples from it at apply time:

- a scalar is deterministic, e.g. `degrees=10`;
- a 2-tuple `(a, b)` samples uniformly, e.g. `degrees=(-10, 10)`;
- a 3-tuple sets per-axis values and a 6-tuple per-axis ranges (for spatial
  parameters);
- a [`Choice`](#choice) samples from a discrete set;
- any `torch.distributions.Distribution` samples from that distribution.

## Choice

::: torchio.Choice

## Distribution

For continuous randomness, pass any `torch.distributions.Distribution`.
The transform draws a fresh sample from it each time it is applied, so
you can use any distribution supported by PyTorch:

<!-- pytest-codeblocks:skip -->
```python
import torch
import torchio as tio

# Sample the rotation angle from a normal distribution (mean 0°, std 5°)
transform = tio.Affine(degrees=torch.distributions.Normal(0, 5))
```

A distribution can also be combined per axis with fixed values and
ranges:

<!-- pytest-codeblocks:skip -->
```python
# Fixed along I, uniform range along J, normal distribution along K
transform = tio.Affine(degrees=(0, (-10, 10), torch.distributions.Normal(0, 5)))
```
