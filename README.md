# Gaussian Process Regression (GPR)

This repository contains a small Gaussian Process Regression (GPR) library implemented from scratch in **Python**.  
It demonstrates how Gaussian Processes can be applied to **spatial (2D)**, **temporal (1D)**, and **spatio-temporal (3D)** datasets.

## Project structure

```
GPR/
│── data/
│   └── temperatures.csv # Temperature dataset (13 French cities, 30 days)
│
│── gpr/ # Core implementation
│   ├── gpr_2d.py # GPR for spatial data (latitude, longitude)
│   ├── gpr_3d.py # GPR combining spatial + temporal (2D + 1D)
│   ├── gpr_time.py # GPR for time series (per city)
│   └── kernels.py # Kernel functions (RBF, Matérn, covariance utils)
│
│── examples/ # Usage examples
│   ├── demo_2d.py # Plot temperature map for France (day t)
│   ├── demo_3d.py # Plot temperature maps (day 30 vs day 33)
│   └── demo_time.py # Forecast temperature evolution for a city
│
└── requirements.txt # Dependencies
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/GPR.git
   cd GPR
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run examples:
   ```bash
   python examples/demo_2d.py
   python examples/demo_time.py
   python examples/demo_3d.py
   ```

## Features

- **2D GPR**: Predict temperature map for France at a given day.  
- **1D GPR**: Forecast temperature evolution for a city over time.  
- **3D GPR**: Combine spatial + temporal GPR to predict temperatures beyond observed days.  

## Dependencies

- numpy  
- matplotlib  
- cartopy  
- scipy  
- pandas  

## Notes

This project was for educational purposes and demonstrates how Gaussian Processes can be implemented from scratch without external ML libraries (like scikit-learn).
