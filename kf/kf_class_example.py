import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter

N = 400
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

m_hist, P_hist, v_hist, S_hist = kf.filter(y_truth_np)

t = np.arange(N)

# if you started storing from i=1, seed the first row for nicer plots
if np.allclose(m_hist[0], 0):
    m_hist[0] = np.array([y_truth_np[0,0], y_truth_np[0,1], 0.0, 0.0])
    # if you also kept P after init, set P_hist[0] = that initial P

# 95% bands for position from the 2x2 position block of P
pos_cov = P_hist[:, :2, :2]
std_x = np.sqrt(pos_cov[:, 0, 0])
std_y = np.sqrt(pos_cov[:, 1, 1])

# ---------- XY path ----------
plt.figure(figsize=(5.5, 5))
plt.plot(x_truth_np[:, 0], x_truth_np[:, 1], lw=2, label="truth path")
plt.plot(m_hist[:, 0], m_hist[:, 1], lw=2, label="KF estimate")
plt.xlabel("x"); plt.ylabel("y"); plt.title("Position in XY")
plt.axis("equal"); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.show()

# ---------- position vs time with 95% bands ----------
fig, (axx, axy) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# x
axx.plot(t, x_truth_np[:, 0], lw=2, label="x truth")
axx.plot(t, m_hist[:, 0], lw=2, label="x KF")
axx.fill_between(t, m_hist[:, 0] - 1.96*std_x, m_hist[:, 0] + 1.96*std_x,
                 alpha=0.2, label="x 95% band")
axx.set_ylabel("x"); axx.grid(alpha=0.3); axx.legend()

# y
axy.plot(t, x_truth_np[:, 1], lw=2, label="y truth")
axy.plot(t, m_hist[:, 1], lw=2, label="y KF")
axy.fill_between(t, m_hist[:, 1] - 1.96*std_y, m_hist[:, 1] + 1.96*std_y,
                 alpha=0.2, label="y 95% band")
axy.set_xlabel("time step"); axy.set_ylabel("y")
axy.grid(alpha=0.3); axy.legend()

fig.suptitle("KF vs ground truth with 95% credible bands", y=0.98)
plt.tight_layout(); plt.show()

# optional quick metrics
rmse_x = np.sqrt(np.mean((x_truth_np[:, 0] - m_hist[:, 0])**2))
rmse_y = np.sqrt(np.mean((x_truth_np[:, 1] - m_hist[:, 1])**2))
print(f"RMSE: x={rmse_x:.3f}, y={rmse_y:.3f}")