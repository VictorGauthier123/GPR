"""
demo_2d.py

Demo of 2D Gaussian Process Regression (latitude, longitude)
for temperature prediction in France.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from gpr.gpr_2d import gpr_predict


def load_training_data(csv_path="../data/temperatures.csv"):
    """
    Load one snapshot of temperatures for all cities (use Day=1 by default).
    """
    df = pd.read_csv(csv_path)
    snapshot = df[df["Day"] == 1]  # <-- choisir le jour que tu veux
    X = snapshot[["Latitude", "Longitude"]].values
    y = snapshot["Temperature"].values
    city_names = snapshot["City"].values
    return X, y, city_names


def plot_on_map(X_train, y_train, city_names, X_test, mu, n_lat=100, n_lon=100):
    """
    Plot predicted temperatures on a map of France.
    """
    mu_plot = mu.reshape((n_lat, n_lon))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-5, 10, 41, 51], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)

    im = ax.imshow(mu_plot, extent=(-7, 10, 42, 52), origin="lower",
                   transform=ccrs.PlateCarree(), cmap="coolwarm", vmin=5, vmax=25)

    scatter = ax.scatter(X_train[:, 1], X_train[:, 0], c=y_train, cmap="coolwarm",
                         s=100, edgecolor="k", linewidth=1,
                         transform=ccrs.Geodetic(), vmin=5, vmax=25)

    for i, name in enumerate(city_names):
        ax.text(X_train[i, 1], X_train[i, 0] + 0.1, name, fontsize=9,
                ha="center", va="bottom", transform=ccrs.Geodetic())

    plt.colorbar(im, orientation="vertical", label="Temperature (°C)")
    plt.title("2D Gaussian Process Prediction of Temperatures in France")
    plt.show()


if __name__ == "__main__":
    # Load training data
    X_train, y_train, city_names = load_training_data("../data/temperatures.csv")

    # Build test grid
    n_lat, n_lon = 100, 100
    lats = np.linspace(42, 52, n_lat)
    lons = np.linspace(-7, 10, n_lon)
    X_test = np.array([[lat, lon] for lat in lats for lon in lons])

    # Run GPR
    mu = gpr_predict(X_train, y_train, X_test, length_scale=1.0)

    # Plot results
    plot_on_map(X_train, y_train, city_names, X_test, mu, n_lat, n_lon)
