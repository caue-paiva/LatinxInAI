import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse



def plot_truth(x_truth_vec):
    x_truth_np = np.vstack(x_truth_vec)   # shape (N, 4)


    # assumes you already have x_truth as shape (N, 4): [x, y, vx, vy]
    # assumes x_truth_np is shape (N, 4): [x, y, vx, vy]
    t = np.arange(len(x_truth_np))

    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # --- position vs time ---
    ax_pos.plot(t, x_truth_np[:, 0], lw=2, label="x")
    ax_pos.plot(t, x_truth_np[:, 1], lw=2, label="y")
    ax_pos.set_ylabel("position")
    ax_pos.set_title("Ground truth — position and velocity vs time")
    ax_pos.grid(alpha=0.3)
    ax_pos.legend()

    # --- velocity vs time ---
    ax_vel.plot(t, x_truth_np[:, 2], lw=2, label="vx")
    ax_vel.plot(t, x_truth_np[:, 3], lw=2, label="vy")
    ax_vel.set_xlabel("time step")
    ax_vel.set_ylabel("velocity")
    ax_vel.grid(alpha=0.3)
    ax_vel.legend()

    plt.tight_layout()
    plt.show()


N = 200
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

#Kalman filter

y0 = y_truth_np[0]
#start from the zero position, with zero velocity
m = np.array([y0[0],y0[1],0,0],dtype=float)

# P0: small-ish on position (measurement-level), large on velocity
pos_var = R.diagonal()            # ~ sensor variances
vel_var = np.array([10.0, 10.0])  # pick large to let data learn velocities
P = np.diag([pos_var[0], pos_var[1], vel_var[0], vel_var[1]])

m_hist = np.zeros((N, 4))
P_hist = np.zeros((N, 4, 4))
v_hist = np.zeros((N, 2))
S_hist = np.zeros((N, 2, 2))

# we infer m0 and P0 from the starting values, so the first prediction is for m1 and P1

for i in range(1,N):
    #prediction
    m_pred = A @ m
    P_pred = A @ P @ A.T + Q


    #update

    #innovation
    v_k = y_truth_np[i] - H @ m_pred

    #innovation variance
    S_k = H @ P_pred @ H.T + R

    #kalman gain
    K = P_pred @ H.T @ np.linalg.solve(S_k, np.eye(2))


    #update step
    m = m_pred + K @ v_k
    
    #Joseph form for numerical robustness
    KH = K @ H
    P = (I4 - KH) @ P_pred @ (I4 - KH).T + K @ R @ K.T

      # store
    m_hist[i] = m
    P_hist[i] = P
    v_hist[i] = v_k
    S_hist[i] = S_k



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