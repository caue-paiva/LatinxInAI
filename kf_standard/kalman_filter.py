import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]

class KalmanFilter:
    
    
    def __init__(
        self,
        m_dim: int,
        n_dim: int,
        A: FloatArray,   # (m_dim, m_dim)
        H: FloatArray,   # (n_dim, m_dim)
        Q: FloatArray,   # (m_dim, m_dim)
        R: FloatArray    # (n_dim, n_dim)
    ) -> None:
        """
        Args:
            m_dim: state dimension (n in many texts)
            n_dim: measurement dimension (m in many texts)
            A: state transition matrix, shape (m_dim, m_dim)
            H: measurement matrix, shape (n_dim, m_dim)
            Q: process noise covariance, shape (m_dim, m_dim)
            R: measurement noise covariance, shape (n_dim, n_dim)
        """
        # Optional runtime shape checks
        if A.shape != (m_dim, m_dim):
            raise ValueError(f"A must be {(m_dim, m_dim)}, got {A.shape}")
        if H.shape != (n_dim, m_dim):
            raise ValueError(f"H must be {(n_dim, m_dim)}, got {H.shape}")
        if Q.shape != (m_dim, m_dim):
            raise ValueError(f"Q must be {(m_dim, m_dim)}, got {Q.shape}")
        if R.shape != (n_dim, n_dim):
            raise ValueError(f"R must be {(n_dim, n_dim)}, got {R.shape}")

        self.m_dim = m_dim
        self.n_dim = n_dim
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
    

    def filter(self, y_ground: list[np.ndarray]):
        Z = np.asarray(y_ground, dtype=float)   # shape (N, n_dim)
        N = Z.shape[0]
        I = np.eye(self.m_dim)

        # init from first measurement [x, y], zero velocity
        z0 = Z[0]
        m = np.array([z0[0], z0[1], 0.0, 0.0], dtype=float)

        pos_var = self.R.diagonal()            # ~ sensor variances
        vel_var = np.array([10.0, 10.0])  # pick large to let data learn velocities
        P = np.diag([pos_var[0], pos_var[1], vel_var[0], vel_var[1]])


        m_hist = np.zeros((N, self.m_dim), dtype=float)
        P_hist = np.zeros((N, self.m_dim, self.m_dim), dtype=float)
        v_hist = np.zeros((N, self.n_dim), dtype=float)
        S_hist = np.zeros((N, self.n_dim, self.n_dim), dtype=float)

        # store the initial posterior/prior (whichever you mean) once
        m_hist[0] = m
        P_hist[0] = P

        for i in range(1,N):
            m_pred = self.A @ m
            P_pred = self.A @ P @ self.A.T + self.Q

            # update
            v_k = Z[i] - self.H @ m_pred                         # innovation
            S_k = self.H @ P_pred @ self.H.T + self.R           # innovation covariance
            K = P_pred @ self.H.T @ np.linalg.solve(S_k, np.eye(self.n_dim))

            m = m_pred + K @ v_k

            # Joseph form
            KH = K @ self.H
            P = (I - KH) @ P_pred @ (I - KH).T + K @ self.R @ K.T

            # store
            m_hist[i] = m
            P_hist[i] = P
            v_hist[i] = v_k
            S_hist[i] = S_k
        
        return  m_hist, P_hist, v_hist, S_hist

