# linear_rnn_kf_sin_to_cos.py
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# -----------------------------
# 0) Repro
# -----------------------------
rng = np.random.default_rng(0)
torch.manual_seed(0)

# 1) Data: sin(x) -> cos(x) 
N = 600
x = np.linspace(0, 8 * math.pi, N)   # same grid
freq_scale = 0.3               

u = np.sin(freq_scale * x)
y_true = np.cos(freq_scale * x)

R_true = (0.05)**2
y_meas = y_true + rng.normal(0, math.sqrt(R_true), size=N)

split = int(0.7 * N)
u_train, y_train = u[:split], y_true[:split]
u_val,   y_val   = u[split:], y_true[split:]

# -----------------------------
# 2) Linear RNN (hidden size H=2)
#    h_k = B h_{k-1} + U u_k
#    y_k = C h_k
# -----------------------------
H = 2

class LinearRNN(nn.Module):
    def __init__(self, hidden_size=H):
        super().__init__()
        self.B = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)   # (H,H)
        self.U = nn.Parameter(torch.randn(hidden_size) * 0.1)                # (H,)
        self.C = nn.Linear(hidden_size, 1, bias=False)                        # in_features=H, out_features=1

    def forward(self, u_seq, h0=None):
        T = u_seq.shape[0]
        h = torch.zeros(H) if h0 is None else h0
        y_hat, h_hist = [], []

        for t in range(T):
            u_t = u_seq[t].item()                    # scalar
            h = self.B @ h + self.U * u_t           # (H,)
            y_hat.append(self.C(h.unsqueeze(0)))    # feed (1,H) into Linear(H,1)
            h_hist.append(h)

        y_hat = torch.vstack(y_hat)                 # (T,1)
        h_hist = torch.stack(h_hist, dim=0)         # (T,H)
        return y_hat, h_hist

# Torch tensors
u_t = torch.from_numpy(u_train).float().unsqueeze(1)   # (T_train, 1)
y_t = torch.from_numpy(y_train).float().unsqueeze(1)   # (T_train, 1)

model = LinearRNN(H)
opt = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# Train
for epoch in range(1000):
    model.train()
    opt.zero_grad()
    y_hat, _ = model(u_t)
    loss = loss_fn(y_hat, y_t)
    loss.backward()
    opt.step()
    if epoch % 200 == 0:
        with torch.no_grad():
            val_pred, _ = model(torch.from_numpy(u_val).float().unsqueeze(1))
            val_loss = nn.functional.mse_loss(val_pred, torch.from_numpy(y_val).float().unsqueeze(1))
        # print(f"epoch {epoch} train={loss.item():.5f} val={val_loss.item():.5f}")

# Freeze
for p in model.parameters():
    p.requires_grad_(False)
model.eval()

# Get full sequence predictions and hidden states from the trained linear RNN
with torch.no_grad():
    y_rnn_hat, h_hist = model(torch.from_numpy(u).float().unsqueeze(1))
y_rnn_hat = y_rnn_hat.squeeze(1).numpy()  # (N,)
h_hist = h_hist.numpy()                    # (N, H)

# Extract learned matrices for KF
B = model.B.detach().numpy()         # (H, H)
U = model.U.detach().numpy()         # (H, 1)
C = model.C.weight.detach().numpy()  # (1, H)
U = U.reshape(-1)

# -----------------------------
# 3) Kalman Filter over hidden state h_k
#    h_k = B h_{k-1} + U u_k + w_k,  w_k ~ N(0, Q)
#    y_k = C h_k + r_k,              r_k ~ N(0, R)
# -----------------------------
Q = 1e-3 * np.eye(H)   # small process noise
R = 0.001            # use the same noise used in y_meas

# Init
m = np.zeros(H)        # prior mean of hidden
P = 1.0 * np.eye(H)    # prior covariance of hidden

m_hist = np.zeros((N, H))
P_tr_hist = np.zeros(N)
y_kf_prior = np.zeros(N)        # C m^- (prior mean in y)
S_prior    = np.zeros(N)        # C P^- C^T + R
y_kf_filt  = np.zeros(N)        # C m (filtered mean in y)
S_filt     = np.zeros(N)        # C P C^T + R

I = np.eye(H)

for k in range(N):
   # predict
    m_pred = (B @ m) + U * u[k]
    P_pred = B @ P @ B.T + Q

    # prior in obs space
    y_hat_prior = float(C @ m_pred)                   # (1,H) @ (H,) -> scalar
    S_k = float(C @ P_pred @ C.T + R)                 # scalar

    # update
    v = y_meas[k] - y_hat_prior
    K = (P_pred @ C.T) / S_k                          # (H,1)
    m = m_pred + (K[:, 0] * v)
    I = np.eye(H)
    P = (I - K @ C) @ P_pred @ (I - K @ C).T + K * R * K.T

    # store
    m_hist[k] = m
    P_tr_hist[k] = np.trace(P)
    y_kf_prior[k] = y_hat_prior
    S_prior[k] = S_k
    y_kf_filt[k] = float(C @ m)
    S_filt[k] = float(C @ P @ C.T + R)

# -----------------------------
# 4) Plots and metrics
# -----------------------------
t = np.arange(N)

plt.figure(figsize=(12, 6))
plt.plot(t, y_true, lw=2, label="true cos(x)")
plt.plot(t, y_meas, lw=1, alpha=0.45, label="measured y")
plt.plot(t, y_rnn_hat, lw=2, label="linear RNN output")
plt.plot(t, y_kf_prior, lw=2, label="KF prior mean")
plt.plot(t, y_kf_filt, lw=2, label="KF filtered mean")
plt.fill_between(t,
                 y_kf_filt - 1.96*np.sqrt(S_filt),
                 y_kf_filt + 1.96*np.sqrt(S_filt),
                 alpha=0.18, label="KF filtered 95%")
plt.xlabel("time step"); plt.ylabel("y")
plt.title("Linear RNN + KF over hidden state: sin(x) -> cos(x)")
plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.show()

# RMSEs
rmse_meas = np.sqrt(np.mean((y_meas - y_true)**2))
rmse_rnn  = np.sqrt(np.mean((y_rnn_hat - y_true)**2))
rmse_kf_p = np.sqrt(np.mean((y_kf_prior - y_true)**2))
rmse_kf_f = np.sqrt(np.mean((y_kf_filt - y_true)**2))
print(f"RMSE vs true: measured={rmse_meas:.4f}  RNN={rmse_rnn:.4f}  KF_prior={rmse_kf_p:.4f}  KF_filt={rmse_kf_f:.4f}")

# NIS for KF (should average near 1 if Q,R well tuned and model is right)
nis = (y_meas - y_kf_prior)**2 / S_prior
print(f"NIS mean ~ 1 if well tuned: {nis.mean():.3f}")

# Hidden covariance trace
plt.figure(figsize=(12, 3.8))
plt.plot(t, P_tr_hist, lw=1.5)
plt.title("Trace of hidden-state covariance P over time")
plt.xlabel("time step"); plt.ylabel("trace(P)")
plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
