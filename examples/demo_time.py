"""
demo_time.py

Demo of Gaussian Process Regression (time series of temperatures for one city).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from gpr.gpr_time import gpr_predict, mse


def load_city_temperatures(csv_path, city_name):
    """
    Load daily temperatures for a given city.
    """
    df = pd.read_csv(csv_path)
    city_data = df[df["City"] == city_name]
    days = city_data["Day"].values
    temps = city_data["Temperature"].values
    return days, temps


def loss_function(params, X_train, y_train, X_val, y_val, nu=1.5):
    """
    Loss = MSE between predicted and validation temperatures.
    """
    sigma, l = params
    mu, _ = gpr_predict(X_train, y_train, X_val, l, nu, sigma)
    return mse(y_val, mu)


if __name__ == "__main__":
    # Load Marseille temperatures
    days, temps = load_city_temperatures("../data/temperatures.csv", "Marseille")

    # Training on first 30 days
    X_train = np.arange(1, 31)
    y_train = temps

    # Validation on days 31–40 (synthetic example: reuse last 10 days of dataset if available)
    X_val = np.arange(31, 41)
    # For demo we simulate validation by reusing the last values cyclically
    y_val = temps[:10]

    # Optimize hyperparameters (sigma, length scale)
    result = minimize(loss_function, x0=[1.0, 1.0],
                      args=(X_train, y_train, X_val, y_val),
                      bounds=[(1e-3, None), (1e-3, None)])
    sigma_opt, l_opt = result.x
    print("Optimized sigma:", sigma_opt)
    print("Optimized length scale:", l_opt)

    # Predict for 40 days
    X_test = np.arange(1, 41)
    mu, cov = gpr_predict(X_train, y_train, X_test, l_opt, nu=1.5, sigma=sigma_opt)
    std = np.sqrt(np.diag(cov))

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(X_train, y_train, "o", label="Training data")
    plt.plot(X_test, mu, "r-", label="Prediction")
    plt.fill_between(X_test, mu - 1.96 * std, mu + 1.96 * std,
                     alpha=0.3, color="red", label="95% CI")
    plt.xlabel("Day")
    plt.ylabel("Temperature (°C)")
    plt.title("Gaussian Process Prediction for Marseille")
    plt.legend()
    plt.show()
