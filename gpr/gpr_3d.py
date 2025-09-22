"""
gpr_3d.py

Hybrid Gaussian Process Regression (2D spatial + 1D temporal).
"""

import numpy as np
import pandas as pd
from gpr.gpr_2d import gpr_predict as gpr_spatial
from gpr.gpr_time import gpr_predict as gpr_temporal


def get_city_series(df, city):
    """
    Extract (days, temperatures) for a given city.
    """
    sub = df[df["City"] == city]
    return sub["Day"].values, sub["Temperature"].values, sub["Latitude"].iloc[0], sub["Longitude"].iloc[0]


def predict_temperatures_for_day(df, day, l_time=2.0, nu=1.5, sigma=1.0):
    """
    Predict temperatures for all cities at a given day.
    If day <= 30: take observed data.
    If day > 30: extrapolate using temporal GPR.
    """
    temps = []
    lats = []
    lons = []
    cities = []

    for city in df["City"].unique():
        days, series, lat, lon = get_city_series(df, city)
        cities.append(city)
        lats.append(lat)
        lons.append(lon)

        if day <= series.shape[0]:  # observed
            temps.append(series[day - 1])  # index décalé car Day commence à 1
        else:  # predicted
            mu, _ = gpr_temporal(days.astype(float), series, np.array([float(day)]),
                                 l=1.0, nu=1.5, sigma=5.0)

            temps.append(mu[0])

    X = np.array(list(zip(lats, lons)))
    y = np.array(temps)
    return X, y, cities


def predict_spatial_map(X_train, y_train, n_lat=100, n_lon=100, length_scale=1.0):
    """
    Predict spatial map at one day using GPR 2D.
    """
    lats = np.linspace(42, 52, n_lat)
    lons = np.linspace(-7, 10, n_lon)
    X_grid = np.array([[lat, lon] for lat in lats for lon in lons])
    mu = gpr_spatial(X_train, y_train, X_grid, length_scale=length_scale)
    return lats, lons, mu.reshape((n_lat, n_lon))
