# Monte Carlo evaluation of hedging error in discrete vs continuous portfolio updating



# ------ Dépendances ------

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt



# ------ Paramètres ------

SIGMA = 0.2
K = 100
T = 1 
R = 0.05
S0 = 100
M = 10000 # nombre de trajectoires MC
N_values = [4, 12, 52, 252]
mu = 0.05



# ------ Fonctions ------

def init_call_price(S_0=S0, K=K, r=R, sigma=SIGMA, T=T):
    """Calcule la valeur initiale du portefeuille F(0,S_0)"""
    d1 = (np.log(S_0/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S_0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)


def delta(tau, S, K=K, sigma=SIGMA, T=T, r=R):
    """
    Calcule la valeur du delta pour un time to maturity tau 
    et une valeur du sous jacent S donnnés
    """
    if tau <=0: # Delta à maturité 
        return 1.0 if S > K else 0.0
    
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    return norm.cdf(d1)


def sim_S(N, mu=mu, S_0=S0, sigma=SIGMA, T=T): 
    """Simule une trajectoire aléatoire de l'actif sous-jacent"""
    dt = T/N
    S = np.zeros(N+1)
    S[0] = S_0
    for k in range(N):
        Z = np.random.normal()
        S[k+1] = S[k]*np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return S


def sim_portefeuille(S, K=K, r=R, sigma=SIGMA, T=T):
    """Simule un portefeuille de couverture à partir d'une trajectoire de sous-jacent S"""
    N = len(S) - 1
    dt = T/N
    V = np.zeros(N+1)
    V[0] = init_call_price(S_0=S[0])
    for k in range(N):
        t = k * dt
        tau = T - t
        d = delta(tau, S[k], K, sigma, T, r)
        V[k+1] = d*S[k+1] + (V[k]-d*S[k])*np.exp(r*dt)
    return V


def erreur_couverture(N, mu=mu, S_0=S0, K=K, r=R, sigma=SIGMA, T=T):
    """Calcule l'erreur de couverture pour une trajectoire de simulation"""
    S = sim_S(N, mu, S_0, sigma, T)
    V = sim_portefeuille(S, K, r, sigma, T)
    payoff = max(S[-1] - K, 0)
    error = V[-1] - payoff
    return error, S[-1], V[-1], payoff



# ------ Boucle Monte-Carlo ------

def mc_erreur_couverture(N, mu=mu, S_0=S0, r=R, K=K, T=T, sigma=SIGMA, M=M):
    errors = np.zeros(M)
    V_sims = np.zeros(M)
    S_sims = np.zeros(M)
    payoffs = np.zeros(M)

    for m in range(M): 
        error, S, V, payoff = erreur_couverture(N, mu, S_0, K, r, sigma, T)

        errors[m] = error
        V_sims[m] = V
        S_sims[m] = S
        payoffs[m] = payoff

    mse = np.mean(errors**2)
    return mse, errors, V_sims, S_sims, payoffs



# ------ SIMULATION -------

if __name__ == "__main__": 

    simulations = {}

    for N in N_values: 
        sim_stats = mc_erreur_couverture(N, mu=mu, S_0=S0, r=R, K=K, T=T, sigma=SIGMA, M=M)
        simulations[N] = {'N': N,
                          'mse': sim_stats[0],
                          'errors': sim_stats[1],
                          'VT': sim_stats[2], 
                          'ST': sim_stats[3], 
                          'payoffs': sim_stats[4]}
        print(f"N={N}, MSE={sim_stats[0]:.6f}")

    # ----- Graphiques 

    # Erreur de couverture en fonction du nombre de rebalancements

    N_list = []
    mse_list = []

    for N in N_values:
        N_list.append(N)
        mse_list.append(simulations[N]["mse"])

    plt.figure(figsize=(8,5))
    plt.plot(N_list, mse_list, marker="o")

    plt.xlabel("Number of rebalancings N")
    plt.ylabel("Mean squared hedging error")
    plt.title("Hedging error vs rebalancing frequency")

    plt.grid(True)
    plt.savefig("plots/hedging_error_vs_N.png", dpi=300, bbox_inches="tight")


    # Scatterplot (S, V)

    N_plot = 52

    ST = simulations[N_plot]["ST"]
    VT = simulations[N_plot]["VT"]

    x = np.linspace(min(ST), max(ST), 500)
    payoff_curve = np.maximum(x - K, 0)

    plt.figure(figsize=(8,5))

    plt.scatter(ST, VT, s=8, alpha=0.5, label="Portfolio terminal value")
    plt.plot(x, payoff_curve, color="red", linewidth=2, label="Call payoff")

    plt.xlabel(r"$S_T$")
    plt.ylabel(r"$V_T^N$")
    plt.title(f"Terminal portfolio value vs payoff (N={N_plot})")

    plt.legend()
    plt.grid(True)

    plt.savefig("plots/ST_vs_VT.png", dpi=300, bbox_inches="tight")


    # Histogramme des erreurs 

    N_plot = 52

    errors = simulations[N_plot]["errors"]

    plt.figure(figsize=(8,5))

    plt.hist(errors, bins=60)

    plt.xlabel(r"$V_T^N - f(S_T)$")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of hedging errors (N={N_plot})")

    plt.grid(True)

    plt.savefig("plots/error_histogram.png", dpi=300, bbox_inches="tight")








 