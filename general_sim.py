import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)
venues = ['V1', 'V2', 'V3']
assets = ['A1', 'A2', 'A3']
makers = ['HighRisk', 'LowRisk1', 'LowRisk2']
beta = 5.0       # routing sensitivity
T = 750         # time steps
dt = 1.0

latency = {(i, M): np.random.uniform(0.1, 1.0) for i in venues for M in makers}
max_lat = max(latency.values())

# Market Maker Preferences
gamma = {'HighRisk': 0.01, 'LowRisk1': 0.0175, 'LowRisk2': 0.02}

# parameters
alpha = {(i, M): latency[(i, M)]/max_lat for i in venues for M in makers}
eta = 0.01
spread = {(i, k, M): 1.0 for i in venues for k in assets for M in makers}
inventory = {(i, k, M): 0   for i in venues for k in assets for M in makers}
cash = {M: 0.0 for M in makers}
kappa = {(i, k, M): 1.0 for i in venues for k in assets for M in makers}

# Underlying prices
mu, sigma = 0.002, 0.02
underlying = pd.DataFrame(index=range(T+1), columns=assets, dtype=float)
underlying.loc[0] = {k: 100.0 for k in assets}
for t in range(T):
    for k in assets:
        shock = np.random.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt))
        underlying.loc[t+1, k] = underlying.loc[t, k] * np.exp(shock)

pnl_history = {M: [] for M in makers}
inv_history = {M: [] for M in makers}
avg_spread_history = {M: [] for M in makers}
flow_history = {M: [] for M in makers}

# Simulation
for t in range(T):
    orders_t = []
    flow_count = Counter()
    n_orders = np.random.poisson(10)
    for _ in range(n_orders):
        i = np.random.choice(venues)
        k = np.random.choice(assets)

        # Routing orders via logit over spread + latency
        costs = {M: 0.5*spread[(i, k, M)] + latency[(i, M)] for M in makers}
        expv = np.array([np.exp(-beta*c) for c in costs.values()])
        probs = expv / expv.sum()
        M = np.random.choice(makers, p=probs)
        flow_count[M] += 1
        orders_t.append((i, k, M))

        # Execute buy/sell
        mid = underlying.loc[t, k]
        half = 0.5*spread[(i, k, M)]
        if np.random.rand() < 0.5:
            price = mid + half
            inventory[(i, k, M)] += 1
            cash[M] -= price
        else:
            price = mid - half
            inventory[(i, k, M)] -= 1
            cash[M] += price

    # record flow share
    for M in makers:
        flow_history[M].append(flow_count[M] / max(n_orders,1))

    # Update kappa
    counts = Counter(orders_t)
    for key, count in counts.items():
        kappa[key] = 0.9*kappa[key] + 0.1*count

    # Latency arbitrage
    for k in assets:
        prices = {i: underlying.loc[t, k] for i in venues}
        for i in venues:
            for j in venues:
                if i >= j:
                    continue
                diff = prices[j] - prices[i]
                best_maker = min(makers, key=lambda M: latency[(i, M)] + latency[(j, M)])
                thresh = latency[(i, best_maker)] + latency[(j, best_maker)]
                if diff > thresh:
                    cash[best_maker] += diff

    # Spread update
    last_pnl = {M: pnl_history[M][-1] if t > 0 else 0 for M in makers}
    for i in venues:
        for k in assets:
            for M in makers:
                q = inventory[(i, k, M)]
                lam = max(kappa[(i, k, M)], 0.1)
                sigma_star = gamma[M]*q + 2/beta + (1/beta)*np.log(1 + beta/lam)
                old = spread[(i, k, M)]
                a = alpha[(i, M)]
                spread[(i, k, M)] = a*old + (1 - a)*sigma_star
                # Update α: reduce stickiness if last PnL positive
                alpha[(i, M)] = np.clip(alpha[(i, M)] - eta*np.sign(last_pnl[M]), 0, 1)

    for M in makers:
        inv_val = sum(inventory[(i, k, M)] * underlying.loc[t, k]
                      for i in venues for k in assets)
        pnl = cash[M] + inv_val
        pnl_history[M].append(pnl)
        inv_history[M].append(sum(inventory[(i, k, M)] for i in venues for k in assets))
        sp_vals = [spread[(i, k, M)] for i in venues for k in assets]
        avg_spread_history[M].append(np.mean(sp_vals))

plt.figure(figsize=(8, 4))
for M in makers:
    plt.plot(pnl_history[M], label=M)
plt.title('PnL Over Time')
plt.xlabel('Time Step')
plt.ylabel('PnL')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
for M in makers:
    plt.plot(inv_history[M], label=M)
plt.title('Net Inventory Over Time')
plt.xlabel('Time Step')
plt.ylabel('Inventory')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
for M in makers:
    plt.plot(avg_spread_history[M], label=M)
plt.title('Average Half-Spread Over Time')
plt.xlabel('Time Step')
plt.ylabel('Half-Spread')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
for M in makers:
    plt.plot(flow_history[M], label=M)
plt.title('Flow Share Over Time')
plt.xlabel('Time Step')
plt.ylabel('Fraction of Orders')
plt.legend()
plt.grid()
plt.show()
