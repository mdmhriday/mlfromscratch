import jax
import jax.numpy as jnp

def r2_score(y_true, y_pred):
    mean_y = jnp.mean(y_true)
    ss_tot = jnp.sum((y_true - mean_y) ** 2)
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot