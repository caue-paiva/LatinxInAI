# kf_over_last_layer_weights.py
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

rng = np.random.default_rng(0)

# -----------------------------
# 1) Synthetic state and measurements
# -----------------------------
N = 300
u = 0.05         # state drift for x
Qx = 1e-2        # process noise variance for x
R = (0.08)**2    # measurement noise variance

def y_true(x):
    return np.tanh(0.7 * x) + 0.25 * np.sin(1.6 * x)

x_true = np.zeros(N)
y_meas = np.zeros(N)
x_true[0] = 0.0
y_meas[0] = y_true(x_true[0]) + rng.normal(0, math.sqrt(R))
for k in range(1, N):
    x_true[k] = x_true[k-1] + u + rng.normal(0, math.sqrt(Qx))
    y_meas[k] = y_true(x_true[k]) + rng.normal(0, math.sqrt(R))

# -----------------------------
# 2) Train a tiny feature extractor ϕ(x)
#    We train an MLP to predict y, then discard its last layer to get features.
# -----------------------------
device = "cpu"
M = 16  # feature dimension

class FeatNet(nn.Module):
    def __init__(self, m=M):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, m),
            nn.Tanh(),
        )
        self.head = nn.Linear(m, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)

    def features(self, x):
        with torch.no_grad():
            return self.backbone(x)

net = FeatNet(M).to(device)

x_t = torch.from_numpy(x_true).float().unsqueeze(1).to(device)
y_t = torch.from_numpy(y_meas).float().unsqueeze(1).to(device)

split = int(0.7 * N)
x_train, y_train = x_t[:split], y_t[:split]
x_val,   y_val  = x_t[split:], y_t[split:]

opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(800):
    net.train()
    opt.zero_grad()
    pred = net(x_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    opt.step()
    if epoch % 100 == 0 and epoch > 0 and loss.item() < 2e-3:
        break

net.eval()

# Build the pre-computed features phi_k, applying phi to every input
with torch.no_grad():
    Phi = net.features(x_t).cpu().numpy()         # shape (N, M)

phi_train = Phi[:split]
phi_val   = Phi[split:]

# Estimate R from validation residuals of the trained head (sanity check)
with torch.no_grad():
    y_val_hat = net(x_val).cpu().numpy().ravel()
R_est = float(np.var(y_val_hat - y_val.cpu().numpy().ravel()))
R_used = R  # keep ground truth; swap to R_est to be data driven

# -----------------------------
# 3) KF over weights w in y = phi^T w + r
#    State: w_k ∈ R^M
#    Dynamics: w_{k+1} = ρ w_k + q_k, q_k ~ N(0, Qw)
# -----------------------------
#state transition matrix is rho*I, which if rho = 1 is a random walk

rho = 1.0               # AR(1) coefficient; 1.0 gives a random walk
q_scale = 1e-3          # process noise scale for weights
Qw = q_scale * np.eye(M)

mu = np.zeros(M)        # prior mean of weights
P  = 1.0 * np.eye(M)    # prior covariance of weights

mu_hist = np.zeros((N, M))
P_tr_hist = np.zeros(N)         # trace for quick view
y_kf_mean = np.zeros(N)         # predictive mean phi^T μ^-
y_kf_var  = np.zeros(N)         # predictive variance phi^T P^- phi + R
nis_hist  = np.zeros(N)         # normalized innovation squared

for k in range(N):
    phi_k = Phi[k]                     # get phi(x_k) shape (M,)
    y_k   = y_meas[k]

    # Predict in weight space
    mu_pred = rho * mu  #basically a random walk

    F = rho * np.eye(M)
    P_pred = F @ P @ F.T + Qw

    
    # Measurement model: y = phi^T w + r
    H = phi_k.reshape(1, -1)          # 1 x M
    S = H @ P_pred @ H.T + R_used     # scalar 1x1
    K = (P_pred @ H.T) / S            # M x 1

    y_hat = (H @ mu_pred).item()        # scalar
    v = y_k - y_hat                   # innovation

    mu = mu_pred + (K.flatten() * v)
    # Joseph form
    I = np.eye(M)
    P = (I - K @ H) @ P_pred @ (I - K @ H).T + K * R_used * K.T

    # store
    mu_hist[k] = mu
    P_tr_hist[k] = np.trace(P)
    y_kf_mean[k] = y_hat
    y_kf_var[k]  = (H @ P_pred @ H.T + R_used).item()
    nis_hist[k]  = float((v**2) / y_kf_var[k])

# -----------------------------
# 4) Batch BLR on the same features for reference
#    This should match KF with ρ=1 and Qw=0 as N grows.
# -----------------------------
alpha = 0.05                  # prior precision α
beta  = 1.0 / R_used          # obs precision β = 1/σ^2
IM = np.eye(M)                # I_M
Phi_all = Phi                 
y_all = y_meas                

post_preci = alpha * IM + beta * (Phi_all.T @ Phi_all) 
post_cov  = np.linalg.inv(post_preci)
post_mean = beta * post_cov @ (Phi_all.T @ y_all)

# Predictive variance for each sample with BLR
pred_var_blr = (1.0 / beta) + np.einsum("ni,ij,nj->n", Phi_all, post_cov, Phi_all)
pred_mean_blr = Phi_all @ post_mean

# -----------------------------
# 5) Plots and metrics
# -----------------------------
t = np.arange(N)
y_true = np.array([y_true(xx) for xx in x_true])

plt.figure(figsize=(12, 6))
plt.plot(t, y_true, lw=2, label="true y")
plt.plot(t, y_meas, lw=1, alpha=0.45, label="measured y")

# BLR mean + 95% band
plt.plot(t, pred_mean_blr, lw=2, label="BLR batch mean")
plt.fill_between(
    t,
    pred_mean_blr - 1.96 * np.sqrt(pred_var_blr),
    pred_mean_blr + 1.96 * np.sqrt(pred_var_blr),
    alpha=0.15,
    label="BLR 95% band"
)

# KF prior mean + 95% band (prior)
plt.plot(t, y_kf_mean, lw=2, label="KF online mean (weights)")
plt.fill_between(
    t,
    y_kf_mean - 1.96 * np.sqrt(y_kf_var),
    y_kf_mean + 1.96 * np.sqrt(y_kf_var),
    alpha=0.2,
    label="KF 95% band"
)

plt.xlabel("time"); plt.ylabel("y")
plt.title("KF over regression weights vs BLR baseline")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.show()

rmse_meas = np.sqrt(np.mean((y_meas - y_true)**2))
rmse_kf   = np.sqrt(np.mean((y_kf_mean - y_true)**2))
rmse_blr  = np.sqrt(np.mean((pred_mean_blr - y_true)**2))
print(f"RMSE vs true: measured={rmse_meas:.4f}, KF={rmse_kf:.4f}, BLR={rmse_blr:.4f}")
print(f"NIS mean (target ~1): {nis_hist.mean():.3f}")
print(f"Final trace(P) on weights: {P_tr_hist[-1]:.3f}")

plt.figure(figsize=(12,4))
plt.plot(t, nis_hist, lw=1.5)
plt.axhline(1.0, color="k", ls="--", alpha=0.6)
plt.title("NIS over time")
plt.xlabel("time"); plt.ylabel("v^2 / S")
plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

plt.figure(figsize=(12,4))
plt.plot(t, P_tr_hist, lw=1.5)
plt.title("Trace of weight covariance P over time")
plt.xlabel("time"); plt.ylabel("trace(P)")
plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

bw_blr_full      = 2 * 1.96 * np.sqrt(pred_var_blr)
bw_kf_prior_full = 2 * 1.96 * np.sqrt(y_kf_var)

print(f"Avg 95% width (BLR):       {bw_blr_full.mean():.4f}")
print(f"Avg 95% width (KF prior):  {bw_kf_prior_full.mean():.4f}")
