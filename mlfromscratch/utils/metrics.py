import jax
import jax.numpy as jnp

def mean_squared_error(features, targets):
    return jnp.mean((features - targets)**2)

