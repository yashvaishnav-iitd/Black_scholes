import numpy as np
import matplotlib.pyplot as plt
from pricer import black_scholes, calculate_greeks

K, T, r, sigma = 105.0, 0.5, 0.05, 0.20
spots = np.linspace(70, 140, 200)  # 200 points from $70 to $140

prices = [black_scholes(S, K, T, r, sigma, "call") for S in spots]
deltas = [calculate_greeks(S, K, T, r, sigma, "call")["Delta"] for S in spots]
gammas = [calculate_greeks(S, K, T, r, sigma, "call")["Gamma"] for S in spots]
vegas  = [calculate_greeks(S, K, T, r, sigma, "call")["Vega"] for S in spots]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0,0].plot(spots, prices, color='steelblue')
axes[0,0].axvline(K, color='gray', linestyle='--', label='Strike')
axes[0,0].set_title("Call Option Price vs Stock Price")
axes[0,0].set_xlabel("Stock Price")
axes[0,0].set_ylabel("Option Price")
axes[0,0].legend()

axes[0,1].plot(spots, deltas, color='darkorange')
axes[0,1].axvline(K, color='gray', linestyle='--')
axes[0,1].set_title("Delta vs Stock Price")
axes[0,1].set_xlabel("Stock Price")
axes[0,1].set_ylabel("Delta")

axes[1,0].plot(spots, gammas, color='green')
axes[1,0].axvline(K, color='gray', linestyle='--')
axes[1,0].set_title("Gamma vs Stock Price")
axes[1,0].set_xlabel("Stock Price")
axes[1,0].set_ylabel("Gamma")

axes[1,1].plot(spots, vegas, color='purple')
axes[1,1].axvline(K, color='gray', linestyle='--')
axes[1,1].set_title("Vega vs Stock Price")
axes[1,1].set_xlabel("Stock Price")
axes[1,1].set_ylabel("Vega")

plt.tight_layout()
plt.savefig("greeks_plot.png", dpi=150)
plt.show()