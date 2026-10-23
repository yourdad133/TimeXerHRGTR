import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("GTR-main/dataset/electricity.csv")

# 按时间顺序，仅使用训练集
values = df.drop(columns=["date"]).to_numpy(dtype=float)
train = values[:int(len(values) * 0.7)]

def autocorr(x, lag):
    a, b = x[:-lag], x[lag:]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return np.corrcoef(a, b)[0, 1]

daily = np.array([
    autocorr(train[:, c], 24)
    for c in range(train.shape[1])
])

weekly = np.array([
    autocorr(train[:, c], 168)
    for c in range(train.shape[1])
])

plt.figure(figsize=(7, 6))
plt.scatter(daily, weekly, alpha=0.6)
plt.xlabel("ACF at lag 24")
plt.ylabel("ACF at lag 168")
plt.title("Electricity: channel-wise periodicity")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
plt.savefig("periodicity.png", dpi=150, bbox_inches="tight")
plt.show()