import numpy as np
import math
import matplotlib.pyplot as plt

# ----- Model -----
N = 200
u = 0.05            # constant drift per step
Q = 1e-3            # process noise variance
R = 1e-2            # measurement noise variance

rng = np.random.default_rng(42)

# ----- Simulate ground truth and measurements -----
x_true = np.zeros(N)
z = np.zeros(N)
x_true[0] = 0.0
z[0] = x_true[0] + rng.normal(0, math.sqrt(R))
for k in range(1, N):
    x_true[k] = x_true[k-1] + u + rng.normal(0, math.sqrt(Q))
    z[k] = x_true[k] + rng.normal(0, math.sqrt(R))

# ----- Kalman filter (scalar) -----
m = 0.0            # posterior mean at k=0
P = 0.1            # posterior variance at k=0

m_hist = np.zeros(N)
P_hist = np.zeros(N)

for k in range(N):
    # predict
    m_pred = m + u
    P_pred = P + Q

    # update
    v = z[k] - m_pred                 # innovation
    S = P_pred + R                    # innovation variance (H=1)
    K = P_pred / S                    # Kalman gain

    m = m_pred + K * v
    # Joseph form for numerical robustness
    P = (1 - K) * P_pred * (1 - K) + K * R * K

    m_hist[k] = m
    P_hist[k] = P

# ----- Plot -----
t = np.arange(N)
std = np.sqrt(P_hist)
upper = m_hist + 2 * std
lower = m_hist - 2 * std

plt.figure(figsize=(10, 5))
plt.plot(t, x_true, label="truth", linewidth=2)
plt.scatter(t, z, s=12, alpha=0.5, label="measurements")
plt.plot(t, m_hist, label="KF mean", linewidth=2)
plt.fill_between(t, lower, upper, alpha=0.2, label="KF 95% band")
plt.xlabel("time step")
plt.ylabel("state")
plt.title("1D Kalman filter on a random walk with drift")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# prediction

steps = 50

m_k = m
P_k = P

m_k_hist = np.zeros(N)
P_k_hist = np.zeros(N)

for i in range(1,steps+1):
    # predict
    m_pred = m_k + u
    P_pred = P_k + Q

    m_k_hist[i] = m_pred
    P_k_hist[i] = P_pred
