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
#N_values = np.arange(4, 252, 10)
mu = 0.05
c = 0.001 # taux de frais de transaction 



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

def sim_portefeuille_couts(S, K=K, r=R, sigma=SIGMA, T=T, c=c):
    N = len(S) - 1
    dt = T / N

    V = np.zeros(N + 1)
    V[0] = init_call_price(S_0=S[0], K=K, r=r, sigma=sigma, T=T)

    total_cost = 0.0
    previous_delta = 0.0

    for k in range(N):
        t = k * dt
        tau = T - t

        d = delta(tau, S[k], K=K, sigma=sigma, T=T, r=r)

        transaction_cost = c * S[k] * abs(d - previous_delta)
        total_cost += transaction_cost

        cash = V[k] - d * S[k] - transaction_cost
        V[k + 1] = d * S[k + 1] + cash * np.exp(r * dt)

        previous_delta = d

    return V, total_cost


def erreur_couverture_couts(N, mu=mu, S_0=S0, K=K, r=R, sigma=SIGMA, T=T, c=c):
    S = sim_S(N, mu, S_0, sigma, T)
    V, total_cost = sim_portefeuille_couts(S, K, r, sigma, T, c)

    payoff = max(S[-1] - K, 0)
    error = V[-1] - payoff

    return error, total_cost



# ------ Boucle Monte-Carlo ------

def mc_erreur_couverture_couts(N, mu=mu, S_0=S0, K=K, r=R, sigma=SIGMA, T=T, c=c, M=M):
    errors = np.zeros(M)
    costs = np.zeros(M)

    for m in range(M):
        error, total_cost = erreur_couverture_couts(N, mu, S_0, K, r, sigma, T, c)
        errors[m] = error
        costs[m] = total_cost

    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    avg_cost = np.mean(costs)

    return mse, rmse, avg_cost


# ------ SIMULATION -------

if __name__ == "__main__":

    N_values_costs = [4, 8, 12, 24, 52, 100, 150, 252]
    cost_rate = 0.001

    rmse_costs = []
    avg_costs = []

    for N in N_values_costs:
        mse_c, rmse_c, avg_cost = mc_erreur_couverture_couts(
            N, mu=mu, S_0=S0, K=K, r=R, sigma=SIGMA, T=T, c=cost_rate, M=M
        )

        rmse_costs.append(rmse_c)
        avg_costs.append(avg_cost)
        print(f"N={N}, RMSE={rmse_c:.6f}, avg_cost={avg_cost:.6f}")

    plt.figure(figsize=(8,5))

    plt.plot(N_values_costs, rmse_costs, marker="o", label="RMSE with transaction costs")
    plt.plot(N_values_costs, avg_costs, marker="o", label="Average transaction costs")

    plt.xlabel("Number of rebalancings N")
    plt.ylabel("Amount")
    plt.title("Hedging error and transaction costs vs rebalancing frequency")

    plt.legend()
    plt.grid(True)
    plt.savefig("plots/transaction_costs_tradeoff.png", dpi=300, bbox_inches="tight")


    total_criterion = np.array(rmse_costs) + np.array(avg_costs)

    plt.figure(figsize=(8,5))

    plt.plot(N_values_costs, total_criterion, marker="o", label="RMSE + average costs")

    plt.xlabel("Number of rebalancings N")
    plt.ylabel("Total criterion")
    plt.title("Trade-off between hedging precision and transaction costs")

    plt.legend()
    plt.grid(True)
    plt.savefig("plots/transaction_costs_optimum.png", dpi=300, bbox_inches="tight")








 