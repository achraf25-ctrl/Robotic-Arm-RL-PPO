import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    data = pd.read_csv("./logs/monitor.csv", skiprows=1)
except FileNotFoundError:
    print("Error")
    exit()


data['rolling_reward'] = data['r'].rolling(window=50).mean()

plt.figure(figsize=(10, 6))


plt.plot(data['l'].cumsum(), data['rolling_reward'], label="PPO Agent", color="#1f77b4", linewidth=2)

plt.title("L-Robot Learning Curve (FetchPickAndPlace-v4)", fontsize=14, fontweight='bold')
plt.xlabel("Total Timesteps", fontsize=12)
plt.ylabel("Episode Reward (Moving Average)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=11)

plt.savefig("learning_curve.png", dpi=300)
print("L-courbe tsauvegardat b smiyat learning_curve.png!")

# Affichi l-graph f l-ecran
plt.show()