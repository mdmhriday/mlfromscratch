import matplotlib.pyplot as plt
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn import datasets
# from sklearn.metrics import r2_score
from .linear_regression import LinearRegression
from ...utils.metrics import r2_score

def main():
    X, y = datasets.make_regression(
        n_samples=1000, n_features=1, noise=20, random_state=4
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(learning_rate=0.03, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = jnp.mean((y_test - predictions) ** 2)
    accu = r2_score(y_test, predictions)
    print(f"MSE: {mse:.4f}, R² Score: {accu:.4f}")

    y_pred_line = regressor.predict(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color="blue", s=10, label="Train data")
    plt.scatter(X_test, y_test, color="red", s=10, label="Test data")
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()