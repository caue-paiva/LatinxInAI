import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# -----------------------------
# 1) Synthetic data
# -----------------------------
rng = np.random.default_rng(0)

N = 300            # total time steps
u = 0.05           # drift per step for the state x
Q = 1e-1           # process noise variance
R = (0.08)**2      # measurement noise variance


# true nonlinear sensor (unknown to the filter; used to generate data)
def h_true(x):
    return np.tanh(0.7 * x) + 0.25 * np.sin(1.6 * x)

x_true = np.zeros(N, dtype=float)
y_meas = np.zeros(N, dtype=float)


#generate syntethic data
x_true[0] = 0.0
y_meas[0] = h_true(x_true[0]) + rng.normal(0, math.sqrt(R))
for k in range(1, N):
    x_true[k] = x_true[k-1] + u + rng.normal(0, math.sqrt(Q))
    y_meas[k] = h_true(x_true[k]) + rng.normal(0, math.sqrt(R))


# -----------------------------
# 2) Tiny PyTorch NN for y = h_theta(x)
# -----------------------------
device = "cpu"

class ObsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x)


net = ObsNet().to(device)

# train using pairs (x_true, y_meas) to learn h(x)
x_t = torch.from_numpy(x_true).float().unsqueeze(1).to(device)
y_t = torch.from_numpy(y_meas).float().unsqueeze(1).to(device)


# time split to respect ordering
split = int(0.7 * N)
x_train, y_train = x_t[:split], y_t[:split]
x_val,   y_val  = x_t[split:], y_t[split:]

opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(800):
    net.train()
    opt.zero_grad()
    y_hat = net(x_train)
    loss = loss_fn(y_hat, y_train)
    loss.backward()
    opt.step()
    # quick early stop on tiny improvement
    if epoch % 100 == 0 and epoch > 0 and loss.item() < 2e-3:
        break

net.eval()

# estimate R from validation residuals as a sanity check
with torch.no_grad():
    resid = (net(x_val) - y_val).cpu().numpy().ravel()
R_est = np.var(resid)
R_used = float(R)  # keep the true R for clarity; swap to R_est to be data driven


# -----------------------------
# 3) EKF with NN observation (scalar)
# -----------------------------
def h_theta_and_jacobian(x_scalar: float):
    """Return y = h_theta(x) and J = dh/dx at x using autograd."""
    xt = torch.tensor([[x_scalar]], dtype=torch.float32, requires_grad=True)
    yt = net(xt)          # shape (1,1)
    y_val = yt.item()
    grad = torch.autograd.grad(yt, xt, retain_graph=False, create_graph=False)[0]
    J = float(grad.item())
    return y_val, J


m_hist = np.zeros(N, dtype=float)   # state mean
P_hist = np.zeros(N, dtype=float)   # state variance
y_pred = np.zeros(N, dtype=float)   # one step predicted measurement mean
S_hist = np.zeros(N, dtype=float)   # predicted measurement variance
v_hist = np.zeros(N, dtype=float)   # innovations



# initial prior
m = 0.0
P = 1.0

m_hist[0] = m
P_hist[0] = P

for k in range(N):
    # Predict
    m_pred = m + u
    P_pred = P + Q

    # Observation linearization at m_pred
    y_hat, J = h_theta_and_jacobian(m_pred)

    # Innovation
    v = y_meas[k] - y_hat
    S = J * J * P_pred + R_used

    # Kalman gain and update
    K = (P_pred * J) / S
    m = m_pred + K * v
    # Joseph scalar form
    P = (1.0 - K * J)**2 * P_pred + (K**2) * R_used

    # store
    m_hist[k] = m
    P_hist[k] = P
    y_pred[k] = y_hat
    S_hist[k] = S
    v_hist[k] = v


# -----------------------------
# 4) Simple multi step forecast
# -----------------------------
H = 40  # horizon
m_f = np.zeros(H, dtype=float)
P_f = np.zeros(H, dtype=float)
y_f = np.zeros(H, dtype=float)
S_f = np.zeros(H, dtype=float)

m_h, P_h = m, P
for h in range(H):
    m_h = m_h + u
    P_h = P_h + Q
    y_h, J_h = h_theta_and_jacobian(m_h)
    m_f[h] = m_h
    P_f[h] = P_h
    y_f[h] = y_h
    S_f[h] = J_h * J_h * P_h + R_used

# -----------------------------
# 5) Plots
# -----------------------------


# -----------------------------
# Compare NN-only vs EKF vs truth and measurements
# -----------------------------
with torch.no_grad():
    y_nn_only = net(torch.from_numpy(x_true).float().unsqueeze(1)).cpu().numpy().ravel()

y_true = np.array([h_true(x) for x in x_true])   # ground truth (noise-free)
y_meas = y_meas                                  # already computed
y_ekf  = y_pred                                  # EKF one-step prediction from earlier

t = np.arange(N)
plt.figure(figsize=(12, 6))
plt.plot(t, y_true, lw=2, label="true y = h_true(x)")
plt.plot(t, y_meas, lw=1, alpha=0.45, label="measured y (noisy)")
plt.plot(t, y_nn_only, lw=2, label="NN-only y = h_theta(x)")
plt.plot(t, y_ekf, lw=2, label="EKF one-step predicted y")
plt.fill_between(t, y_ekf - 1.96*np.sqrt(S_hist), y_ekf + 1.96*np.sqrt(S_hist),
                 alpha=0.18, label="EKF 95% band")
plt.xlabel("time"); plt.ylabel("y")
plt.title("True vs measured vs NN-only vs EKF")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.show()

# RMSEs against ground truth
rmse_meas = np.sqrt(np.mean((y_meas - y_true)**2))
rmse_nn   = np.sqrt(np.mean((y_nn_only - y_true)**2))
rmse_ekf  = np.sqrt(np.mean((y_ekf - y_true)**2))
print(f"RMSE vs true: measured={rmse_meas:.4f}, NN-only={rmse_nn:.4f}, EKF={rmse_ekf:.4f}")


t = np.arange(N)
plt.figure(figsize=(12, 5))
plt.plot(t, y_meas, lw=1, alpha=0.5, label="measured y")
plt.plot(t, [h_true(x) for x in x_true], lw=2, label="true h(x)")
plt.plot(t, y_pred, lw=2, label="EKF predicted y | k-1")
plt.fill_between(t,
                 y_pred - 1.96 * np.sqrt(S_hist),
                 y_pred + 1.96 * np.sqrt(S_hist),
                 alpha=0.2, label="95% band")
plt.legend(); plt.grid(alpha=0.3)
plt.title("Observation space: measured vs EKF one step prediction")
plt.xlabel("time"); plt.ylabel("y")
plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t, x_true, lw=2, label="true x")
plt.plot(t, m_hist, lw=2, label="EKF x mean")
plt.fill_between(t,
                 m_hist - 1.96 * np.sqrt(P_hist),
                 m_hist + 1.96 * np.sqrt(P_hist),
                 alpha=0.2, label="x 95% band")
plt.legend(); plt.grid(alpha=0.3)
plt.title("State estimate with 95% band")
plt.xlabel("time"); plt.ylabel("x")
plt.tight_layout(); plt.show()

# Forecast bands in y
tf = np.arange(N, N + H)
plt.figure(figsize=(12, 4))
plt.plot(t, [h_true(x) for x in x_true], lw=2, label="true h(x) in-sample")
plt.plot(tf, y_f, lw=2, ls="--", label="forecast y")
plt.fill_between(tf,
                 y_f - 1.96 * np.sqrt(S_f),
                 y_f + 1.96 * np.sqrt(S_f),
                 alpha=0.2, label="forecast 95% band")
plt.grid(alpha=0.3); plt.legend()
plt.title("Forecast in observation space")
plt.xlabel("time"); plt.ylabel("y")
plt.tight_layout(); plt.show()

# quick diagnostics
nis = (v_hist**2) / S_hist
print(f"NIS mean ~ 1 if well tuned: {nis.mean():.3f}")
print(f"Final state std: {math.sqrt(P_hist[-1]):.4f}, final obs std: {math.sqrt(S_hist[-1]):.4f}")