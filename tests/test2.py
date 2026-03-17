import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.System import *
from src.EKBF import *
from src.KBF import *
from src.GPSModel import *

#____________________________________________________________________________________________________________________
# NOISE
rng = np.random.default_rng(seed=42)
#ProcessNoise
dW1 = WeinerIncrements(t_span, dt, rng)
dW2 = WeinerIncrements(t_span, dt, rng)
dW3 = WeinerIncrements(t_span, dt, rng)
# print(dW1.shape)
W = np.hstack([dW1, dW2, dW3])
# print(W.shape)
# Measurement noise
dV1 = WeinerIncrements(t_span, dt, rng)
dV2 = WeinerIncrements(t_span, dt, rng)
dV3 = WeinerIncrements(t_span, dt, rng)
dV4 = WeinerIncrements(t_span, dt, rng)
V = np.hstack([dV1, dV2, dV3, dV4])

Y0 = h(0, X0).flatten()
#____________________________________________________________________________________________________________________
GPS_System = System(X0, Y0, F, C, ObsDrift, ObsDrift_Jacobian, D, dt, t_span, Sat_1)
GPS_KBF_filter = Model_KBF(GPS_System, Xe0, S0)
GPS_EKBF_filter = Model_EKBF(GPS_System, Xe0, S0)


# X_n, Y_n, Xe_n, S_n=GPS_EKBF_filter.EKBFEstimate(W, V, dt/100)
X_n, Y_n, X_nom, XeK_n, SK_n = GPS_KBF_filter.KBFEstimate(W, V, dt/100)


t = np.arange(0, t_span, dt)
te = np.arange(0, t_span, dt/100)
N  = len(t)
Ne = len(te)
#____________________________________________________________________________________________________________________
# True VS Estimated Trajectory: well spread satellites

plt.figure(1)
plt.plot(X_n[:,0], X_n[:,1], label='True Trajectory')
plt.plot(XeK_n[:,0], XeK_n[:,1], label='Est. Trajectory')
plt.legend()

plt.figure(2)
plt.plot(t, X_n[:N,0], label='True')
plt.plot(t, XeK_n[:N,0], label='Est.')
plt.legend()

plt.show()
