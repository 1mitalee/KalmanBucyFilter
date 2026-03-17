# Simulation of a general SDE, generating Weiner Increments

import numpy as np   
import matplotlib.pyplot as plt

class System:

    def __init__(self, X0, Y0, F, C, ObsDrift, ObsDrift_Jacobian, D ,dt, T_span, Sensor):
        self.X0     = X0 
        self.Y0     = Y0
        self.F      = F # function of time 
        self.C      = C # function of time
        self.h      = lambda t, x: ObsDrift(t, x, Sensor)
        self.H      = lambda t, x: ObsDrift_Jacobian(t, x, Sensor)
        self.D      = D # function of time
        self.dt     = dt
        self.T_span = T_span
        self.dim    = len(X0)
        self.dim_o  = len(Y0)
       

    def drift_f(self, t, x):
        return self.F(t)@x

    def diff_g(self, t, x):
        return self.C(t)

    def drift_f2(self, t, y, x):
        return self.h(t, x)

    def diff_g2(self, t, y, x):
        return self.D(t)

    def EulerMaruyama(self, W, V):
        N       = int(self.T_span / self.dt)
        X_n     = np.zeros((N + 1, self.dim))
        Y_n     = np.zeros((N+1, self.dim_o))

        X_n[0]  = self.X0
        Y_n[0] = self.Y0

        for i in range(1, N + 1):
            t_i    = (i - 1) * self.dt
            X_n[i] = X_n[i-1] + self.drift_f(t_i, X_n[i-1]) * self.dt + W[i-1,:]@np.transpose(self.diff_g(t_i, X_n[i-1])) 
            # m = V[i-1, :]@np.transpose(self.diff_g2(t_i, Y_n[i-1],X_n[i-1]))
            # print(m.shape)

            Y_n[i] = Y_n[i-1] + np.transpose(self.drift_f2(t_i, Y_n[i-1], X_n[i-1])*self.dt) + V[i-1, :]@np.transpose(self.diff_g2(t_i, Y_n[i-1],X_n[i-1]))
            # X_n[i-1] or X[i]?
        return X_n, Y_n

    def NominalTrajectory(self, X0_hat):
        N       = int(self.T_span / self.dt)
        Xnom_n     = np.zeros((N + 1, self.dim))
        Xnom_n[0]  = X0_hat
        for i in range(1, N + 1):
            t_i    = (i - 1) * self.dt
            Xnom_n[i] = Xnom_n[i-1] + self.drift_f(t_i, Xnom_n[i-1]) * self.dt
            # m = V[i-1, :]@np.transpose(self.diff_g2(t_i, Y_n[i-1],X_n[i-1]))
            # print(m.shape)
        return Xnom_n

def WeinerIncrements(t_span, dt, rng):
    N = int(t_span/dt) 
    dW = rng.standard_normal(N) * np.sqrt(dt)
    dW = dW.reshape(N, 1) 
    return dW


def plot_ellipse(ax, P, mu= np.array([0,0]), n_std = 1.0, num_points = 200):

    eigval,eigvec = np.linalg.eigh(P)
    order = eigval.argsort()[::-1]
    eigval=eigval[order]
    eigvec = eigvec[:, order]

    theta = np.linspace(0, 2*np.pi, num_points)
    circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse = mu[:, None] + n_std * eigvec @ np.diag(np.sqrt(eigval))@circle

    ax.plot(ellipse[0], ellipse[1])
    # plt.scatter(mu[0], mu[1], marker='x')
    # plt.gca().set_aspect('equal')
    # plt.show()


