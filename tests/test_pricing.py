import math

import pytest

from model_functions import BS, merton_jump_call


S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
q = 0.01
m = 1.0
v = 0.1


def test_black_scholes_call_price_is_positive_and_finite():
    price = BS(S, K, T, r, sigma, q)

    assert math.isfinite(price)
    assert price > 0


def test_black_scholes_call_price_increases_when_stock_price_increases():
    lower_stock_price = BS(95, K, T, r, sigma, q)
    higher_stock_price = BS(105, K, T, r, sigma, q)

    assert higher_stock_price > lower_stock_price


def test_black_scholes_call_price_decreases_when_strike_increases():
    lower_strike_price = BS(S, 95, T, r, sigma, q)
    higher_strike_price = BS(S, 105, T, r, sigma, q)

    assert higher_strike_price < lower_strike_price


def test_merton_call_price_is_positive_and_finite():
    price = merton_jump_call(S, K, T, r, sigma, m, v, 0.5, q)

    assert math.isfinite(price)
    assert price > 0


def test_merton_with_zero_lambda_matches_black_scholes():
    black_scholes_price = BS(S, K, T, r, sigma, q)
    merton_price = merton_jump_call(S, K, T, r, sigma, m, v, 0.0, q)

    assert merton_price == pytest.approx(black_scholes_price)
