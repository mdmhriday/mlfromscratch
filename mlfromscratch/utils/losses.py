import jax
import jax.numpy as jnp

def mean_squared_error(features, targets):
    if features.size == 0 or targets.size == 0:
        raise ValueError("Features and targets cannot be empty.")
    if features.shape != targets.shape:
        raise ValueError("Features and targets must have the same shape.")
    return jnp.mean((features - targets)**2)


def r2_score(y_true, y_pred):
    mean_y = jnp.mean(y_true)
    ss_tot = jnp.sum((y_true - mean_y) ** 2)
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot