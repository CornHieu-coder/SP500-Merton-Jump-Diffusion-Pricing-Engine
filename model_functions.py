
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import math
import os

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad

from nelson_siegel_svensson.calibrate import calibrate_nss_ols

from fredapi import Fred

import plotly.graph_objects as go
from sklearn.metrics import r2_score

from seleniumbase import Driver
import re
import time

# Get options data from yfinance:

Params = {}
def getting_data():
    
    """I. Get yfinance data

    """
    ticker = "^SPX"
    Ticker = yf.Ticker(ticker)
    # Getting stock price:
    hist = Ticker.history(period="1d")
    S = hist['Close'].iloc[-1]
    Params['Stock_price'] = S
    # Get a list of all available options expiration dates
    expiration_dates = Ticker.options

    # List containing the full dataframe:
    all_options = []

    for date in expiration_dates:
        option_chain = Ticker.option_chain(date)
        calls = option_chain.calls

        # Calculate maturity:
        exp_date = dt.datetime.strptime(date, '%Y-%m-%d')
        today = dt.datetime.today()
        days_to_expire = (exp_date - today).days

        # Add new column:
        calls['daysToExpiration'] = days_to_expire
        all_options.append(calls)
    df = pd.concat(all_options)

    df['moneyness'] = S/df['strike']
    df['yearsToExpiration'] = df['daysToExpiration']/365
    df['midPrice'] = (df['bid'] + df['ask']) / 2
    df['spread'] = df['ask'] - df['bid']
    df_clean = df[
        (df['volume'] > 0) &
        (df['moneyness'] > 0.8) &
        (df['moneyness'] < 1.1) &
        (df['bid'] > 0.50) &
        (df['impliedVolatility'] > 0.05) &
        (df['impliedVolatility'] < 1.0 ) &
        (df['yearsToExpiration'] > 0.5) &
        (df['yearsToExpiration'] < 2.0) &
        (df['spread'] < df['midPrice'])
    ]
    cleaning_status = f"Data cleaned: {df_clean.shape[0]} options selected from {df.shape[0]} total options."
    return df_clean,S,cleaning_status
def div_yield(df_clean,S):
    """II. Get risk-free zero coupon rate:"""

    # Using fredapi to fetch yield curve data from the environment.
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        raise ValueError(
            "FRED_API_KEY is not set."
            " Set it before running live_app.py, for example:"
            " $env:FRED_API_KEY='your_fred_api_key_here' on Windows PowerShell"
            " or export FRED_API_KEY=your_fred_api_key_here on macOS/Linux."
        )

    fred = Fred(api_key=fred_api_key)

    series_ids = {
        '1 Mo': 'DGS1MO',
        '3 Mo': 'DGS3MO',
        '6 Mo': 'DGS6MO',
        '1 Yr': 'DGS1',
        '2 Yr': 'DGS2',
        '3 Yr': 'DGS3',
        '5 Yr': 'DGS5',
        '7 Yr': 'DGS7',
        '10 Yr': 'DGS10',
        '20 Yr': 'DGS20',
        '30 Yr': 'DGS30'
    }

    data_frames = {}
    for label, fred_id in series_ids.items():
        series = fred.get_series(fred_id)
        data_frames[label] = series

    yield_curve_table = pd.DataFrame(data_frames)
    yield_curve_table = yield_curve_table.dropna()
    latest_curve = yield_curve_table.iloc[-1]

    # Using the Nelson-Siegel-Svensson (NSS) model to calibrate the yield curve:
    yield_maturities = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    yields = latest_curve.values.astype(float) / 100
    curve_fit, status = calibrate_nss_ols(yield_maturities,yields)
    Params['riskFreeRate'] = curve_fit

    # Add the risk free rate to the dataframe
    df_clean['riskFreeRate'] = df_clean['yearsToExpiration'].apply(curve_fit)
    yield_status = "Risk-free rates added to dataframe."
    return df_clean,yield_status

def BS(S, K, T, r, sigma, q):
        d1 = (np.log(S/K) + (r - q + (sigma**2) / 2 ) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        C = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return C

def merton_jump_call(S, K, T, r, sigma, m, v, lam, q):
    lam_prime = lam * m
    p = 0
    n = 0
    max = 100
    while n < max:
        weight = (np.exp(-lam_prime*T)*(lam_prime*T)**n)/ math.factorial(n)
        sigma_n = np.sqrt(sigma**2 + n * v**2 / T)
        r_n = r - lam * (m - 1) + n * np.log(m) / T

        BS_price = BS(S, K, T, r_n, sigma_n,q)
        new_val = weight * BS_price
        p += new_val
        n += 1

    return p

def get_yield_robust():
        # Initialize the browser
        driver = Driver(uc=True, headless=False)

        try:
            url = "https://www.gurufocus.com/economic_indicators/150/sp-500-dividend-yield"
            print(f"Navigating to {url}...")
            driver.get(url)

            # Wait for the page to fully load
            print("Waiting for data...")
            time.sleep(10)

            # 1. Grab ALL text from the page (Visible text only)
            full_text = driver.get_text("body")
            
            # 2. Use Regex to find the specific pattern
            match = re.search(r"current dividend yield is\s+(\d+\.\d+%)", full_text, re.IGNORECASE)
            
            if match:
                return match.group(1)
            
            # Pattern B: "S&P 500 Dividend Yield: X.XX%"
            match = re.search(r"S&P 500 Dividend Yield\s*:\s*(\d+\.\d+%)", full_text, re.IGNORECASE)
            
            if match:
                return match.group(1)

            # Pattern C: Just find ANY percentage near the word "Yield" (Fallback)
            fallback_match = re.search(r"Yield.{0,50}?(\d+\.\d+%)", full_text, re.IGNORECASE)
            
            if fallback_match:
                return fallback_match.group(1)

            return "Could not find percentage in page text."

        finally:
            driver.quit()

def get_live_yield():
    q = 0.015  # Default value in case scraping fails
    result = get_yield_robust()
    q = result.strip('%')  
    q = float(q) / 100  
    Params['Dividend_yield'] = q
    return q
def get_model_prices(df_clean,S,q):
    """IV. Calibration:

    """

# Mean squared error function to calibrate model parameters:
    def SqrDiff(x, market_data, S):
        sigma, m, v, lam = x
        
        K_array = market_data['strike'].values
        T_array = market_data['yearsToExpiration'].values
        r_array = market_data['riskFreeRate'].values
        market_prices = market_data['midPrice'].values
        
        q = Params['Dividend_yield']

        model_prices = merton_jump_call(S, K_array, T_array, r_array, sigma, m, v, lam, q)
        
        diff = model_prices - market_prices
        mean_err = np.mean(diff**2)
        
        return mean_err

    initial_bounds = [0.15, 1.0, 0.1, 0.5]
    bnds = ((0.01, 1.0), (0.1, 2.0), (0.01, 1.0), (0.0, 5.0))

    result = minimize(
        fun=SqrDiff,
        x0=initial_bounds,
        args=(df_clean,S),
        method='SLSQP',
        bounds=bnds,
        tol=1e-6
    )

    sigma, m , v , lam = [param for param in result.x]
    Params['sigma'] = sigma
    Params['m'] = m
    Params['v'] = v
    Params['lam'] = lam

    df_clean['model_prices'] = merton_jump_call(
    S, 
    df_clean['strike'].values,             
    df_clean['yearsToExpiration'].values,  
    df_clean['riskFreeRate'].values,       
    sigma, m, v, lam, q                    
)
    calibration_status = "Merton Jump Diffusion model calibrated and prices computed."
    return df_clean,calibration_status
def compare_prices(df_clean):
    
    """V. Compare market prices with model prices"""

    fig = go.Figure(data=[go.Mesh3d(x=df_clean.yearsToExpiration, y=df_clean.strike, z=df_clean.midPrice, color='mediumblue', opacity=0.55)])

    fig.add_scatter3d(x=df_clean.yearsToExpiration, y=df_clean.strike, z=df_clean.model_prices, mode='markers')

    fig.update_layout(
        title_text='Market Prices (Mesh) vs Calibrated Merton-Jump Diffusion Prices (Markers)',
        scene = dict(xaxis_title='TIME (Years)',
                        yaxis_title='STRIKES (Pts)',
                        zaxis_title='INDEX OPTION PRICE (Pts)'),
        height=800,
        width=800
    )

    return fig
def evaluate_model(df_clean):
    """VI. Evaluate model performance"""
    df_clean['error'] = df_clean['model_prices'] - df_clean['midPrice']
    df_clean['pct_error'] = (df_clean['error'] / df_clean['midPrice']).abs()
    mape = df_clean['pct_error'].mean() * 100

    y_true = df_clean['midPrice']
    y_pred = df_clean['model_prices']
    r2 = r2_score(y_true, y_pred)
    return f"Model MAPE: {mape}%, R²: {r2}"

def get_full_df():
    print("--- Running get_full_df ---") 

    df_clean, S, cleaning_status = getting_data()
    
    print(f"Cleaning Status: [{cleaning_status}]") 

    df_clean, yield_status = div_yield(df_clean, S)
    print(f"Yield Status: [{yield_status}]")

    q = Params.get('Dividend_yield', 0.015)  # Use the scraped dividend yield if available
    df_clean, calibration_status = get_model_prices(df_clean, S, q)
    print(f"Calibration Status: [{calibration_status}]")
    return df_clean,cleaning_status,yield_status,calibration_status
def main():
    df_clean,cleaning_status,yield_status,calibration_status = get_full_df()
    fig = compare_prices(df_clean)
    metrics = evaluate_model(df_clean)
    return cleaning_status,yield_status,calibration_status,metrics,fig

def price_arbitrary_option(strike, maturity):
    S = Params['Stock_price']
    q = Params['Dividend_yield']
    sigma = Params['sigma']
    m = Params['m']
    v = Params['v']
    lam = Params['lam']
    r = Params['riskFreeRate'](maturity)

    price = merton_jump_call(S, strike, maturity, r, sigma, m, v, lam, q)
    contract_value = price * 100 
    return f"Price: {price:.2f} pts (Value: ${contract_value:,.2f})"

