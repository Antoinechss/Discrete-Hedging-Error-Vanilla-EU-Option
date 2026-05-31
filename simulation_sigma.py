# Monte Carlo evaluation of hedging error in discrete vs continuous portfolio updating
# Impact of volatility parameter sigma



# ------ Dépendances ------

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt



# ------ Paramètres ------

mu = 0.05
K = 100
T = 1
R = 0.05
S0 = 100
M = 10000 # nombre de trajectoires MC
N_values = np.linspace(4, 256, 20).astype(int)
SIGMA_values = [0.05, 0.10, 0.20]
N_plot = 52



# ------ Fonctions ------

def init_call_price(S_0=S0, K=K, r=R, sigma=0.2, T=T):
    """Calcule la valeur initiale du portefeuille F(0,S_0)"""
    d1 = (np.log(S_0/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S_0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)


def delta(tau, S, K=K, sigma=0.2, T=T, r=R):
    """
    Calcule la valeur du delta pour un time to maturity tau
    et une valeur du sous jacent S donnnés
    """
    if tau <= 0: # Delta à maturité
        return 1.0 if S > K else 0.0

    d1 = (np.log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    return norm.cdf(d1)


def sim_S(N, sigma, mu=mu, S_0=S0, T=T):
    """Simule une trajectoire aléatoire de l'actif sous-jacent"""
    dt = T/N
    S = np.zeros(N+1)
    S[0] = S_0
    for k in range(N):
        Z = np.random.normal()
        S[k+1] = S[k]*np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return S


def sim_portefeuille(S, sigma, K=K, r=R, T=T):
    """Simule un portefeuille de couverture à partir d'une trajectoire de sous-jacent S"""
    N = len(S) - 1
    dt = T/N
    V = np.zeros(N+1)
    V[0] = init_call_price(S_0=S[0], sigma=sigma)
    for k in range(N):
        t = k * dt
        tau = T - t
        d = delta(tau, S[k], K, sigma, T, r)
        V[k+1] = d*S[k+1] + (V[k]-d*S[k])*np.exp(r*dt)
    return V


def erreur_couverture(N, sigma, mu=mu, S_0=S0, K=K, r=R, T=T):
    """Calcule l'erreur de couverture pour une trajectoire de simulation"""
    S = sim_S(N, sigma, mu, S_0, T)
    V = sim_portefeuille(S, sigma, K, r, T)
    payoff = max(S[-1] - K, 0)
    error = V[-1] - payoff
    return error, S[-1], V[-1], payoff



# ------ Boucle Monte-Carlo ------

def mc_erreur_couverture(N, sigma, mu=mu, S_0=S0, r=R, K=K, T=T, M=M):
    errors = np.zeros(M)
    V_sims = np.zeros(M)
    S_sims = np.zeros(M)
    payoffs = np.zeros(M)

    for m in range(M):
        error, S, V, payoff = erreur_couverture(N, sigma, mu, S_0, K, r, T)

        errors[m] = error
        V_sims[m] = V
        S_sims[m] = S
        payoffs[m] = payoff

    mse = np.mean(errors**2)
    return mse, errors, V_sims, S_sims, payoffs



# ------ SIMULATION -------

if __name__ == "__main__":

    simulations = {}

    for sigma in SIGMA_values:
        simulations[sigma] = {}
        for N in N_values:
            sim_stats = mc_erreur_couverture(N, sigma=sigma, mu=mu, S_0=S0, r=R, K=K, T=T, M=M)
            simulations[sigma][N] = {'N': N,
                                     'mse': sim_stats[0],
                                     'errors': sim_stats[1],
                                     'VT': sim_stats[2],
                                     'ST': sim_stats[3],
                                     'payoffs': sim_stats[4]}
            print(f"sigma={sigma}, N={N}, MSE={sim_stats[0]:.6f}")

    # ----- Graphiques

    # Erreur de couverture en fonction du nombre de rebalancements

    plt.figure(figsize=(8,5))

    for sigma in SIGMA_values:
        N_list = list(N_values)
        mse_list = [simulations[sigma][N]["mse"] for N in N_values]
        plt.plot(N_list, mse_list, marker="o", label=rf"$\sigma={sigma}$")

    plt.xlabel("Number of rebalancings N")
    plt.ylabel("Mean squared hedging error")
    plt.title(r"Hedging error vs rebalancing frequency — varying $\sigma$")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/sigma_dependence/hedging_error_vs_N_sigma.png", dpi=300, bbox_inches="tight")


    # Scatterplot (S, V)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, sigma in enumerate(SIGMA_values):
        ST = simulations[sigma][N_plot]["ST"]
        VT = simulations[sigma][N_plot]["VT"]

        x = np.linspace(min(ST), max(ST), 500)
        payoff_curve = np.maximum(x - K, 0)

        axes[i].scatter(ST, VT, s=8, alpha=0.5, label="Portfolio terminal value")
        axes[i].plot(x, payoff_curve, color="red", linewidth=2, label="Call payoff")
        axes[i].set_xlabel(r"$S_T$")
        axes[i].set_ylabel(r"$V_T^N$")
        axes[i].set_title(rf"$\sigma={sigma}$, N={N_plot}")
        axes[i].legend()
        axes[i].grid(True)

    fig.suptitle(r"Terminal portfolio value vs payoff — varying $\sigma$", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/sigma_dependence/ST_vs_VT_sigma.png", dpi=300, bbox_inches="tight")
