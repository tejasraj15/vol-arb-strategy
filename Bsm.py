'''

import yfinance as yf
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from datetime import datetime
import seaborn as sns
import pandas as pd
from scipy.optimize import brentq

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S        # Underlying asset price
        self.K = K        # Option strike price
        self.T = T        # Time to expiration in years
        self.r = r        # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying asset

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_option_price(self):
        return (self.S * si.norm.cdf(self.d1(), 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0))
    
    def put_option_price(self):
        return (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0) - self.S * si.norm.cdf(-self.d1(), 0.0, 1.0))



def implied_volatility_call(self, market_price):
    """Find the volatility that makes the model price equal to market price"""
    def objective(sigma):
        self.sigma = sigma
        return self.call_option_price() - market_price
    
    # Solve for sigma between 0.01 and 5.0
    return brentq(objective, 0.01, 5.0)

def implied_volatility_put(self, market_price):
    """Find the volatility that makes the model price equal to market price"""
    def objective(sigma):
        self.sigma = sigma
        return self.put_option_price() - market_price
    
    # Solve for sigma between 0.01 and 5.0
    return brentq(objective, 0.01, 5.0)

bsm = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2)

'''