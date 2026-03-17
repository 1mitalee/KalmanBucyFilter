import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

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
# DATA FOR EXPERIMENT 1
# Baseline Run
# GPS_System = System(X0, Y0, F, C, ObsDrift, ObsDrift_Jacobian, D, dt, t_span, Sat_1)

# GPS_KBF_filter = Model_KBF(GPS_System, Xe0, S0)
# GPS_EKBF_filter = Model_EKBF(GPS_System, Xe0, S0)


# X_n, Y_n, Xe_n, S_n=GPS_EKBF_filter.EKBFEstimate(W, V, dt/100)
# _, _, X_nom, XeK_n, SK_n = GPS_KBF_filter.KBFEstimate(W, V, dt/100)


# t = np.arange(0, t_span, dt)
# te = np.arange(0, t_span, dt/100)
# N  = len(t)
# Ne = len(te)

# np.savez(
# 	"experiment_result.npz",
# 	X_n = X_n,
# 	Y_n = Y_n,
# 	X_nom = X_nom,
# 	Xe_n = Xe_n,
# 	S_n = S_n,
# 	XeK_n = XeK_n,
# 	SK_n = SK_n,
# 	t = t,
# 	te =te,
# 	N=N,
# 	Ne= Ne,
# 	W=W,
# 	V= V
# 	)

# #____________________________________________________________________________________________________________________
# DATA FOR EXPERIMENT 2
# Degenerate Geometry

# GPS_System_2 = System(X0, Y0, F, C, ObsDrift, ObsDrift_Jacobian, D, dt, t_span, Sat_2)

# GPS_KBF_filter_2 = Model_KBF(GPS_System_2, Xe0, S0)
# GPS_EKBF_filter_2 = Model_EKBF(GPS_System_2, Xe0, S0)


# X_n, Y_n, Xe_n, S_n=GPS_EKBF_filter_2.EKBFEstimate(W, V, dt/100)
# _, _, X_nom, XeK_n, SK_n = GPS_KBF_filter_2.KBFEstimate(W, V, dt/100)


# t = np.arange(0, t_span, dt)
# te = np.arange(0, t_span, dt/100)
# N  = len(t)
# Ne = len(te)

# np.savez(
# 	"experiment_result_2.npz",
# 	X_n = X_n,
# 	Y_n = Y_n,
# 	X_nom = X_nom,
# 	Xe_n = Xe_n,
# 	S_n = S_n,
# 	XeK_n = XeK_n,
# 	SK_n = SK_n,
# 	t = t,
# 	te =te,
# 	N=N,
# 	Ne= Ne,
# 	W=W,
# 	V= V
# 	)

# #____________________________________________________________________________________________________________________
# DATA FOR EXPERIMENT 3
GPS_System = System(X0, Y0, F, C, ObsDrift, ObsDrift_Jacobian, D, dt, t_span, Sat_1)

Xe0_new = X0 + np.array([300, 400, 0, 0, 0])

GPS_KBF_filter_3 = Model_KBF(GPS_System, Xe0_new, S0)
GPS_EKBF_filter_3 = Model_EKBF(GPS_System, Xe0_new, S0)


X_n, Y_n, Xe_n, S_n=GPS_EKBF_filter_3.EKBFEstimate(W, V, dt/100)
_, _, X_nom, XeK_n, SK_n = GPS_KBF_filter_3.KBFEstimate(W, V, dt/100)


t = np.arange(0, t_span, dt)
te = np.arange(0, t_span, dt/100)
N  = len(t)
Ne = len(te)

np.savez(
	"experiment_result_3.npz",
	X_n = X_n,
	Y_n = Y_n,
	X_nom = X_nom,
	Xe_n = Xe_n,
	S_n = S_n,
	XeK_n = XeK_n,
	SK_n = SK_n,
	t = t,
	te =te,
	N=N,
	Ne= Ne,
	W=W,
	V= V
	)




