import jax
import jax.numpy as jnp

def mean_squared_error(x, y):
    return jnp.mean((x - y)**2)

