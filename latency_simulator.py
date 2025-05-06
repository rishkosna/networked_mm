import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def run_sim(latency_factor):
    venues = ['V1', 'V2', 'V3']
    assets = ['A1', 'A2', 'A3']
    makers = ['HighRisk', 'LowRisk1', 'LowRisk2']
    beta = 5.0
    T = 1000
    dt = 1.0

    base_latency = {(i, M): np.random.uniform(0.1, 1.0) for i in venues for M in makers}
    latency = {k: v * latency_factor for k, v in base_latency.items()}
    max_lat = max(latency.values())

    # Parameters
    gamma = {'HighRisk': 0.01, 'LowRisk1': 0.0175, 'LowRisk2': 0.02}
    alpha = {k: v/max_lat for k, v in latency.items()}
    eta = 0.01
    spread = {(i, a, M): 1.0 for i in venues for a in assets for M in makers}
    inventory = {(i, a, M): 0 for i in venues for a in assets for M in makers}
    cash = {M: 0.0 for M in makers}
    kappa = {(i, a, M): 1.0 for i in venues for a in assets for M in makers}

    # Underlying Price
    mu, sigma = 0.0005, 0.02
    underlying = pd.DataFrame(index=range(T+1), columns=assets, dtype=float)
    underlying.loc[0] = {a: 100.0 for a in assets}
    for t in range(T):
        for a in assets:
            shock = np.random.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt))
            underlying.loc[t+1, a] = underlying.loc[t, a] * np.exp(shock)

    avg_spread_history = {M: [] for M in makers}

    # Simulation
    for t in range(T):
        orders_t = []
        n_orders = np.random.poisson(10)
        for _ in range(n_orders):
            i = np.random.choice(venues)
            a = np.random.choice(assets)
            costs = {M: 0.5*spread[(i, a, M)] + latency[(i, M)] for M in makers}
            expv = np.array([np.exp(-beta*c) for c in costs.values()])
            probs = expv / expv.sum()
            M = np.random.choice(makers, p=probs)
            orders_t.append((i, a, M))
            mid = underlying.loc[t, a]
            half = 0.5 * spread[(i, a, M)]
            if np.random.rand() < 0.5:
                inventory[(i, a, M)] += 1
                cash[M] -= mid + half
            else:
                inventory[(i, a, M)] -= 1
                cash[M] += mid - half

        # Update kappa
        counts = Counter(orders_t)
        for key, count in counts.items():
            kappa[key] = 0.9*kappa[key] + 0.1*count

        # Spread update
        last_pnl = {M: 0 for M in makers}
        for i in venues:
            for a in assets:
                for M in makers:
                    q = inventory[(i, a, M)]
                    lam = max(kappa[(i, a, M)], 0.1)
                    sigma_star = gamma[M]*q + 2/beta + (1/beta)*np.log(1 + beta/lam)
                    old = spread[(i, a, M)]
                    a_stick = alpha[(i, M)]
                    spread[(i, a, M)] = a_stick*old + (1 - a_stick)*sigma_star

        # Record average half-spread
        for M in makers:
            vals = [spread[(i, a, M)] for i in venues for a in assets]
            avg_spread_history[M].append(np.mean(vals))

    return avg_spread_history

# Run scenarios
low_latency = run_sim(latency_factor=0.1)
high_latency = run_sim(latency_factor=10)
time = range(len(low_latency['HighRisk']))

plt.figure(figsize=(10, 6))
for M, color in zip(['HighRisk', 'LowRisk1', 'LowRisk2'], ['C0', 'C1', 'C2']):
    plt.plot(time, low_latency[M], label=f'{M} (Low Latency)', color=color, linestyle='-')
    plt.plot(time, high_latency[M], label=f'{M} (High Latency)', color=color, linestyle='--')
plt.title('Average Half-Spread: Low vs High Latency')
plt.xlabel('Time Step')
plt.ylabel('Half-Spread')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
