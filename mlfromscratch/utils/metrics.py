import jax
import jax.numpy as jnp

def mean_squared_error(features, targets):
    if features.size == 0 or targets.size == 0:
        raise ValueError("Features and targets cannot be empty.")
    if features.shape != targets.shape:
        raise ValueError("Features and targets must have the same shape.")
    return jnp.mean((features - targets)**2)

