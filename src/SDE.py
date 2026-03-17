# Simulation of a general SDE, generating Weiner Increments

import numpy as np   

class System:

    def __init__(self, f, g, X0, f2, g2, Y0, T_span, dt):
        self.f      = f
        self.g      = g
        self.X0     = X0
        self.T_span = T_span
        self.f2     = f2
        self.g2     = g2
        self.dt     = dt
        self.dim    = len(X0)
        self.dim_o  = len(Y0)

    def drift_f(self, t, x):
        return self.f(t, x)

    def diff_g(self, t, x):
        return self.g(t, x)

    def drift_f2(self, t, y, x):
        return self.f2(t, y,x)

    def diff_g2(self, t, y, x):
        return self.g2(t, y,x)

    def EulerMaruyama(self, W, V):
        N       = int(self.T_span / self.dt)
        X_n     = np.zeros((N + 1, self.dim))
        Y_n     = np.zeros((N+1, self.dim_o))

        X_n[0]  = self.X0
        Y_n[0] = self.Y0

        for i in range(1, N + 1):
            t_i    = (i - 1) * self.dt
            X_n[i] = X_n[i-1] + self.drift_f(t_i, X_n[i-1]) * self.dt + W[i-1,:]@np.transpose(self.diff_g(t_i, X_n[i-1])) 

            Y_n[i] = Y_n[i-1] + self.drift_f2(t_i, Y_n[i-1], X_n[i-1])*self.dt + V[i-1, :]@np.transpose(self.diff_g2(t_i, Y_n[i-1],X_n[i-1]))
            # X_n[i-1] or X[i]?
        return X_n

def WeinerIncrements(t_span, dt, rng):
    N = int(t_span/dt) 
    dW = rng.standard_normal(N) * np.sqrt(dt)
    dW = dW.reshape(N, 1) 
    return dW