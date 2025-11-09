import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter

N = 100000
N_train = 80000
ForecastN = N - N_train
dt  = 0.1          # sampling period
qc1 = 1.0          # spectral density in x-direction
qc2 = 1.0          # spectral density in y-direction
sigma1 = 0.5   # std dev of x-position measurement noise
sigma2 = 0.5   # std dev of y-position measurement noise

rng = np.random.default_rng(123)

I4 = np.eye(4)

# x,y, deltaX, deltaY
x = np.array([0,0,1,1])

H = np.array(
    [
    [1,0,0,0],
    [0,1,0,0]
    ],dtype=float)

R = np.diag([sigma1**2, sigma2**2])

A = np.array([
    [1.0, 0.0, dt,  0.0],
    [0.0, 1.0, 0.0, dt ],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

Q = np.array([
    [qc1*dt**3/3,   0.0,           qc1*dt**2/2,  0.0],
    [0.0,           qc2*dt**3/3,   0.0,          qc2*dt**2/2],
    [qc1*dt**2/2,   0.0,           qc1*dt,       0.0],
    [0.0,           qc2*dt**2/2,   0.0,          qc2*dt]
], dtype=float)



kf = KalmanFilter(
    m_dim = len(x), 
    n_dim =R .shape[0], 
    A = A, 
    H = H, 
    R = R, 
    Q = Q
)


x_truth:list = []
y_truth:list = []

# run the model and create the dataset
for i in range(N):
    #measure
    r_k = rng.multivariate_normal(mean=np.zeros(2), cov=R)

    y = H @ x + r_k
    y_truth.append(y.copy())
    x_truth.append(x.copy())

    #update state
    w_k = rng.multivariate_normal(mean=np.zeros(4), cov=Q)
    x = A @ x + w_k


x_truth_np = np.vstack(x_truth)
y_truth_np = np.vstack(y_truth)

# run KF through train dataset
m_hist, P_hist, v_hist, S_hist = kf.filter(y_truth_np[:N_train])

# last posterior at t = N_train-1
m_last = m_hist[N_train - 1].copy()
P_last = P_hist[N_train - 1].copy()

# run forecast
m_fore = np.zeros((ForecastN, 4))
P_fore = np.zeros((ForecastN, 4, 4))

m_h = m_last.copy()
P_h = P_last.copy()
for h in range(ForecastN):
    m_h = A @ m_h
    P_h = A @ P_h @ A.T + Q
    m_fore[h] = m_h
    P_fore[h] = P_h

# -----------------------------
# 4) Plots: train vs forecast and bands
# -----------------------------
t_all = np.arange(N)
t_tr  = np.arange(N_train)
t_fc  = np.arange(N_train, N)

# position std from covariance
pos_cov_tr = P_hist[:, :2, :2]
std_x_tr = np.sqrt(pos_cov_tr[:, 0, 0])
std_y_tr = np.sqrt(pos_cov_tr[:, 1, 1])

pos_cov_fc = P_fore[:, :2, :2]
std_x_fc = np.sqrt(pos_cov_fc[:, 0, 0])
std_y_fc = np.sqrt(pos_cov_fc[:, 1, 1])

# XY path with train and forecast means
plt.figure(figsize=(6, 6))
plt.plot(x_truth_np[:, 0], x_truth_np[:, 1], lw=2, label="truth path")
plt.plot(m_hist[:, 0], m_hist[:, 1], lw=2, label="KF mean (train)")
plt.plot(m_fore[:, 0], m_fore[:, 1], lw=2, ls="--", label="KF forecast mean")
plt.xlabel("x"); plt.ylabel("y"); plt.title("XY: train vs forecast")
plt.axis("equal"); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.show()

# x over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax1.plot(t_all, x_truth_np[:, 0], lw=2, label="x truth", color="C0")
ax1.plot(t_tr,  m_hist[:, 0], lw=2, label="x KF (train)", color="C1")
ax1.fill_between(t_tr,
                 m_hist[:, 0] - 1.96*std_x_tr,
                 m_hist[:, 0] + 1.96*std_x_tr,
                 alpha=0.2, label="x 95% band (train)", color="C1")
ax1.plot(t_fc,  m_fore[:, 0], lw=2, ls="--", label="x KF forecast", color="C3")
ax1.fill_between(t_fc,
                 m_fore[:, 0] - 1.96*std_x_fc,
                 m_fore[:, 0] + 1.96*std_x_fc,
                 alpha=0.2, label="x 95% band (forecast)", color="C3")
ax1.set_ylabel("x"); ax1.grid(alpha=0.3); ax1.legend(loc="best")

# y over time
ax2.plot(t_all, x_truth_np[:, 1], lw=2, label="y truth", color="C0")
ax2.plot(t_tr,  m_hist[:, 1], lw=2, label="y KF (train)", color="C1")
ax2.fill_between(t_tr,
                 m_hist[:, 1] - 1.96*std_y_tr,
                 m_hist[:, 1] + 1.96*std_y_tr,
                 alpha=0.2, label="y 95% band (train)", color="C1")
ax2.plot(t_fc,  m_fore[:, 1], lw=2, ls="--", label="y KF forecast", color="C3")
ax2.fill_between(t_fc,
                 m_fore[:, 1] - 1.96*std_y_fc,
                 m_fore[:, 1] + 1.96*std_y_fc,
                 alpha=0.2, label="y 95% band (forecast)", color="C3")
ax2.set_xlabel("time step"); ax2.set_ylabel("y")
ax2.grid(alpha=0.3); ax2.legend(loc="best")

fig.suptitle("KF train (0..299) and predict-only forecast (300..499)", y=0.98)
plt.tight_layout(); plt.show()

# -----------------------------
# 5) Forecast metrics vs held-out truth
# -----------------------------
x_true_fc = x_truth_np[N_train:N, 0]
y_true_fc = x_truth_np[N_train:N, 1]

rmse_x_fc = np.sqrt(np.mean((x_true_fc - m_fore[:, 0])**2))
rmse_y_fc = np.sqrt(np.mean((y_true_fc - m_fore[:, 1])**2))

cov_x = np.mean((x_true_fc >= m_fore[:, 0] - 1.96*std_x_fc) &
                (x_true_fc <= m_fore[:, 0] + 1.96*std_x_fc))
cov_y = np.mean((y_true_fc >= m_fore[:, 1] - 1.96*std_y_fc) &
                (y_true_fc <= m_fore[:, 1] + 1.96*std_y_fc))

print(f"Forecast RMSE: x={rmse_x_fc:.3f}, y={rmse_y_fc:.3f}")
print(f"Forecast 95% coverage: x={cov_x*100:.1f}%, y={cov_y*100:.1f}%")