"""Random seed handling.

Two sources of randomness matter in this project:

1. NumPy, used to sample token sequences from the hidden Markov models.
2. PyTorch, used to initialise and train the transformer.

For NumPy we avoid the global random state. Instead we create an explicit
`numpy.random.Generator` and pass it into every function that needs randomness.
That way a function signature tells you whether the function is random, and two
pieces of code cannot silently share (and disturb) one hidden global stream.

PyTorch's global generator is harder to avoid, because layer initialisation uses
it, so we seed it directly.

Reproducibility caveat: the same seed gives identical results on the same
machine with the same library versions. Across machines or versions, results can
differ in the last few decimal places.
"""

import numpy as np
import torch


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed PyTorch's global generator and return a fresh NumPy Generator.

    Args:
        seed: A non-negative integer.

    Returns:
        A NumPy Generator seeded with `seed`. Pass it explicitly to any function
        that samples with NumPy.
    """
    torch.manual_seed(seed)
    numpy_generator = np.random.default_rng(seed)
    return numpy_generator
