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
