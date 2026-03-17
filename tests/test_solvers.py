import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.SDE import *
from src.ObservationModel import *

# System Parameters

X0 = np.array([0,0,10,2,50])

t_span = 60
dt = 0.01

A = np.array([[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
G = np.array([[0,0,0], [0,0,0], [0.5,0,0], [0,0.5,0], [0,0,0.1]])

R_sqrt = 5 * np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1 ,0],[0, 0, 0, 1]])
R = 25 * np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1 ,0],[0, 0, 0, 1]])

S1 = np.array([[20000,0],[0,20000],[-20000,0],[0,-20000]])
S2 = np.array([[20000, 0],[21000,2000],[22000,-2000],[23000,1000]])

P0 = np.tile((X0[0],X0[1]),(4,1))
Y0 = np.linalg.norm(P0-S1, axis=1)
Y0 = Y0.reshape((1,4))
 

# State Model
def f1(t, x):

	drift = A@x

	return drift

def g1(t,x):

	return G	
# Observation Model

# def f2(t, y):
# 	return ObsDrift(t, y, x, S1)

# def g2(t, y):
# 	return R_sqrt

# DYNAMICS FOR THE OBSERVATIONS ARE INCORRECT --> MAKE THE REQUIRED CHANGES



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
V = np.hstack([dV1, dV2, dV3])

StateSDE = SDE(f1, g1, X0, t_span, dt)

X_n = StateSDE.EulerMaruyama(W)

# ObservationSDE = SDE(f2, g2, Y0, t_span, dt)

t = np.arange(0, t_span+dt, dt)


# # plt.plot(t, X_n[:,0], label='x')
# # plt.plot(t, X_n[:,1], label='y')
# # plt.plot(t, X_n[:,2], label='vx')
# # plt.plot(t, X_n[:,3], label='vy')
# plt.plot(t, X_n[:,4], label='b')

# plt.plot(X_n[:,0], X_n[:,1], label='trajectory')
# plt.legend()
# plt.xlabel('t')
# plt.ylabel('y')
# plt.title('State Dynamics')

# plt.show()

