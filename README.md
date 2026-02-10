# S&P500-Merton-Jump-Diffusion-Pricing-Engine
**Result example**
<img width="1856" height="1018" alt="Image" src="https://github.com/user-attachments/assets/ef7c3ff7-2179-46e3-a712-d1351fe5f3b1" />

# S&P 500 Merton Call Option Pricing Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This is a quantitative finance tool that prices **S&P 500 (SPX)** options using the **Merton Jump Diffusion Model (1976)**.

Unlike the standard Black-Scholes model, which assumes stock prices move smoothly, the Merton model accounts for sudden **"jumps"** in asset prices (e.g., due to market crashes or earnings shocks). This allows for a more accurate pricing of "out-of-the-money" options and captures the **volatility smile** observed in real markets.

## Key Features

* **Real-Time Calibration:** Fetches live SPX option chains and Treasury yields via `yfinance`.
* **Fast Optimization:** Uses `scipy.optimize.minimize` (SLSQP) to quickly calibrate model parameters to current market prices.
* **3D Visualization:** Plots the implied volatility/pricing surface to visualize market structure. 
* **Custom Pricing:** After calibration, users can price any arbitrary option (custom Strike & Maturity).

---

## Under the Hood: How It Works

This engine follows a standard quantitative workflow: **Fetch $\rightarrow$ Process $\rightarrow$ Calibrate $\rightarrow$ Price.**

### 1. Data Ingestion
* **Spot Price ($S_0$):** Fetched live from `^SPX`.
* **Risk-Free Rate ($r$):** Constructed dynamically using US Treasury yields using Fredapito build a yield curve.
* **Dividend Yield ($q$):** Web-scraped via https://www.gurufocus.com/economic_indicators/150/sp-500-dividend-yield
* **Market Options:** We download the full option chain for SPX, filtering for liquid contracts (Volume > 0, Bid > 0.5,...) to ensure data quality.

### 2. The Merton Jump Diffusion Model

The mathematical formulation is:

<img width="807" height="316" alt="Image" src="https://github.com/user-attachments/assets/d20dd193-4946-4e15-b329-43ccd77b44a2" />

Where BS stands for Black-Scholes call option pricing formula:

<img width="552" height="266" alt="Image" src="https://github.com/user-attachments/assets/6fcb35b3-3c3e-45b9-aee0-6922e13bb398" />

### 3. Calibration (The Core Logic)
This is the most computationally intensive part. We do not "guess" the parameters; we solve for them.

* **Objective:** Find the set of parameters $\theta = \{ \sigma, \lambda, \mu_J, \delta_J \}$ that minimizes the difference between the **Model Price** and the **Market Price** across all traded options.
* **Loss Function:** We use **Mean of Squared Errors (MSE)**.
    $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\text{MarketPrice}_i - \text{ModelPrice}_i)^2$$
* **Optimizer:** We use **SLSQP (Sequential Least Squares Programming)** via `scipy.optimize.minimize`. This gradient-based method is fast and efficient for finding the optimal parameters when starting from a reasonable guess.

### 4. Pricing
Once the model is calibrated, we use the analytical solution (Merton's formula, which is essentially a weighted sum of Black-Scholes prices) to price any call option requested by the user.

---


To run this application locally on your machine:

**1. Clone the repository**

**2. Install requirements.txt**

**3. Run live_app.py (gradio will handle the web interface)**
