import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import matplotlib.pyplot as plt
from src.System import *
from src.EKBF import *
from src.KBF import *
from src.GPSModel import *

data = np.load("experiment_result_3.npz")
# print(data.files)

X_n =data["X_n"] 
Y_n =data["Y_n"] 
X_nom =data["X_nom"] 
Xe_n = data["Xe_n"] 
S_n = data["S_n"]
XeK_n =data["XeK_n"] 
SK_n =data["SK_n"]
t =data["t"]
te =data["te"]
N=data["N"]
Ne=data["Ne"]
W=data["W"]
V= data["V"]


fig, ax = plt.subplots()

ax.plot(X_n[:,0], X_n[:,1], label='True Trajectory')
ax.plot(Xe_n[:,0], Xe_n[:,1], label='Est. Trajectory')


T = np.arange(0,60,5)
for i in T:
	n = int(i/dt)
	k = int(100 * n)
	P = S_n[k,:2,:2]
	plot_ellipse(ax, P, Xe_n[n, :2] , 1, 100)

ax.set_xlabel('p_x (m)')
ax.set_ylabel('p_y (m)')
ax.set_title('Trajectory')
ax.legend()
fig.savefig("results/exp3_1.png")
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

fig2, ax2 = plt.subplots()
# Innovation
dV = np.zeros((N,4))
dV_norm = np.zeros(N)
for i in range(N):
	dV[i] = Y_n[i+1]-Y_n[i] - (ObsDrift(dt*i, Xe_n[i], Sat_1)).reshape((4,))*dt
	dV_norm[i] = np.linalg.norm(dV[i])

ax2.plot(t, dV_norm, label='Innovation Norm')
ax2.set_xlabel('t(s)')
ax2.set_ylabel('dv norm')
ax2.set_title('Innovation norm')
ax2.legend()
fig2.savefig("results/exp3_2.png")

# #_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

fig3, ax3 = plt.subplots()
error = X_n - Xe_n
error_pos = error[:, :2]
RMSE = np.linalg.norm(error_pos, axis=1)
error_K = X_n - XeK_n
error_K_pos = error_K[:, :2]
RMSE_K = np.linalg.norm(error_K_pos, axis=1)

RMSE_compare = np.zeros((N,1))
RMSE_K_compare = np.zeros((N,1))
# print(RMSE_compare.shape)
for i in range(N):
	k = 100*i
	RMSE_compare[i] = np.sqrt(S_n[k, 0, 0] + S_n[k, 1, 1])
	RMSE_K_compare[i] = np.sqrt(SK_n[k, 0, 0] + SK_n[k, 1, 1])

ax3.plot(t, RMSE[:N], label = "Empirical RMSE (Monte Carlo)- EKBF")
ax3.plot(t, RMSE_compare, label= "Predicted RMSE - EKBF")
# ax3.plot(t, RMSE[:N], label = "Empirical RMSE (Monte Carlo) - KBF")
# ax3.plot(t, RMSE_compare, label= "Predicted RMSE - KBF")
ax3.set_xlabel('t(s)')
ax3.set_ylabel('RMSE')
ax3.set_title('RMSE')
ax3.legend()
fig3.savefig("results/exp3_3.png")
# #_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

fig4, ax4 = plt.subplots()

ax4.plot(t, RMSE[:N], label = "Empirical RMSE (Monte Carlo)- EKBF")
ax4.plot(t, RMSE_compare, label= "Predicted RMSE - EKBF")
ax4.plot(t, RMSE_K[:N], label = "Empirical RMSE (Monte Carlo) - KBF")
ax4.plot(t, RMSE_K_compare, label= "Predicted RMSE - KBF")
ax4.set_xlabel('t(s)')
ax4.set_ylabel('RMSE')
ax4.set_title('RMSE')
ax4.legend()
fig4.savefig("results/exp3_4.png")

plt.show()
