from typing import Any
import numpy as np
from scipy.stats import norm


class monte_carlo:
    def __init__(self, S0, K, r, T, sigma, dlta, N, numiter):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.N = N
        self.dlta = dlta
        self.numiter = numiter

    def asset_price(S0, r, T, sigma, dlta, N):
        S = np.zeros(N)
        z = np.random.normal(loc=0,scale=1,size=N)
        S[0] = S0
        dt = T/N
        nudt = (r-dlta-0.5*sigma**2)*dt
        sigmadt = sigma*np.sqrt(dt)
        for i in range(1,N):
            S[i] = S[i-1] * np.exp(nudt + sigmadt * z[i])
        return S
    
    def call(S, K):
        return np.max([0, S-K])
    
    def S_sim(S0, r, T, sigma, dlta, N, numiter):
        S_mc = np.zeros((N,numiter))
        for i in range(numiter):
            S_mc[:,i] = monte_carlo.asset_price(S0, r, T, sigma, dlta, N)
        return S_mc
    
    def mc_call(S_mc, r, T, N, numiter, K):
        call_mc = np.zeros((N,numiter))
        C0expected = np.zeros(numiter)
        for i in range(numiter):
            call_mc[:,i] = [monte_carlo.call(S_mc[j,i],K) for j in range(N)]
        C0expected = np.exp(-r*T)*(1/numiter)*np.sum(call_mc, axis=1)
        return C0expected







     

        

