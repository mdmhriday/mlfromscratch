import jax
import jax.numpy as jnp
import unittest
from ...utils.metrics import mean_squared_error

class TestMeanSquaredError(unittest.TestCase):
    def test_basic_mse(self):
        features = jnp.array([1.0, 2.5, 4.0])
        targets = jnp.array([2.0, 3.0, 4.0])

        actual_mse = .4167
        expected_mse = mean_squared_error(features, targets)
        self.assertAlmostEqual(actual_mse, expected_mse, places=4)
    def test_empty_arrays(self):
        features = jnp.array([])
        targets = jnp.array([])
        with self.assertRaises(ValueError):
            mean_squared_error(features, targets)
    def test_different_shape(self):
        features = jnp.array([1.0, 2.5, 4.0])
        targets = jnp.array([2.0, 3.0, 4.0, 5.0])
        with self.assertRaises(ValueError):
            mean_squared_error(features, targets)
    def test_identical_arrays(self):
        features = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        expected_mse = 0.0
        actual_mse = mean_squared_error(features, targets)
        self.assertEqual(actual_mse, expected_mse)



if __name__ == "__main__":
    unittest.main()