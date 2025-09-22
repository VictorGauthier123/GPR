"""
demo_3d.py

Demo: Hybrid 2D (spatial) + 1D (temporal) GPR.
Displays temperature map at day 30 (observed)
and day 33 (predicted).
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from gpr.gpr_3d import predict_temperatures_for_day, predict_spatial_map


def plot_map(lats, lons, mu_grid, X_train, y_train, city_names, day):
    """
    Plot a temperature map for a given day.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-5, 10, 41, 51], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)

    im = ax.imshow(mu_grid, extent=(-7, 10, 42, 52), origin="lower",
                   transform=ccrs.PlateCarree(), cmap="coolwarm", vmin=5, vmax=25)

    ax.scatter(X_train[:, 1], X_train[:, 0], c=y_train, cmap="coolwarm",
               s=100, edgecolor="k", linewidth=1,
               transform=ccrs.Geodetic(), vmin=5, vmax=25)

    for i, name in enumerate(city_names):
        ax.text(X_train[i, 1], X_train[i, 0] + 0.1, name,
                fontsize=9, ha="center", va="bottom", transform=ccrs.Geodetic())

    plt.colorbar(im, orientation="vertical", label="Temperature (°C)")
    plt.title(f"Temperature Map - Day {day}")
    plt.show()


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../data/temperatures.csv")

    for day in [30, 31]:
        # Predict city temperatures (observed if day<=30, predicted otherwise)
        X_train, y_train, city_names = predict_temperatures_for_day(df, day)

        # Predict spatial map
        lats, lons, mu_grid = predict_spatial_map(X_train, y_train, n_lat=100, n_lon=100)

        # Plot map
        plot_map(lats, lons, mu_grid, X_train, y_train, city_names, day)
