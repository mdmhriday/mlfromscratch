import jax
import jax.numpy as jnp
from jax import jit
from ...utils.losses import mean_squared_error


class LinearRegression:
    def __init__(self, learning_rate=0.003, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weight = None
        self.bias = 0.0

    def fit(self, X, y):
        X = jnp.array(X)
        y = jnp.array(y)

        n_features = X.shape[1]

        self.weight = jax.random.normal(jax.random.PRNGKey(0), (n_features,))

        @jit
        def update(weight, bias, X_batch, y_true, lr):
            # forward pass
            y_pred = jnp.dot(X_batch, weight) + bias
            loss = mean_squared_error(y_true, y_pred)

            # compute gradients
            error = y_pred - y_true
            dw = jnp.dot(X_batch.T, error) / len(y_true)
            db = jnp.mean(error)

            # update
            new_weight = weight - lr*dw
            new_bias = bias - lr*db

            return new_weight, new_bias, loss
        
        # training loop
        for _ in range(self.n_iters):
            self.weight, self.bias, loss = update(self.weight,
            self.bias, X, y, self.learning_rate
            )

    def predict(self, X):
        return jnp.dot(jnp.array(X), self.weight) + self.bias