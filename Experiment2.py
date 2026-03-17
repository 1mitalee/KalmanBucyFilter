import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import matplotlib.pyplot as plt
from src.System import *
from src.EKBF import *
from src.KBF import *
from src.GPSModel import *

data = np.load("experiment_result_2.npz")
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

data_base = np.load("experiment_result.npz")
X_n_base =data_base["X_n"] 
Xe_n_base = data_base["Xe_n"] 


# Experiment 1: Baseline run (well-spread satellites)

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
fig.savefig("results/exp2_1.png")
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
fig2.savefig("results/exp2_2.png")

# #_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

fig3, ax3 = plt.subplots()
error = X_n - Xe_n
error_pos = error[:, :2]
RMSE = np.linalg.norm(error_pos, axis=1)

RMSE_compare = np.zeros((N,1))
# print(RMSE_compare.shape)
for i in range(N):
	k = 100*i
	RMSE_compare[i] = np.sqrt(S_n[k, 0, 0] + S_n[k, 1, 1])

ax3.plot(t, RMSE[:N], label = "Empirical RMSE (Monte Carlo)")
ax3.plot(t, RMSE_compare, label= "Predicted RMSE")
ax3.set_xlabel('t(s)')
ax3.set_ylabel('RMSE')
ax3.set_title('RMSE')
ax3.legend()
fig3.savefig("results/exp2_3.png")
# #_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

fig4, ax4 = plt.subplots()
error_base = X_n_base - Xe_n_base
error_pos_base = error_base[:, :2]
RMSE_base = np.linalg.norm(error_pos_base, axis=1)
ax4.plot(t, RMSE[:N], label = "Empirical RMSE (Degenerate)")
ax4.plot(t, RMSE_base[:N], label= "Empirical RMSE (Base)")
ax4.set_xlabel('t(s)')
ax4.set_ylabel('RMSE')
ax4.set_title('Convergence Speed comparison using RMSE')
ax4.legend()
fig4.savefig("results/exp2_4.png")
# #_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

fig5, ax5 = plt.subplots()
con_num_base = np.zeros((N,1))
con_num_deg = np.zeros((N,1))

for i in range(N):
	t_i = i * dt
	H1 = ObsDrift_Jacobian(t_i, Xe_n_base[i], Sat_1)
	print(H1.shape)
	H1 = H1[:, :2]
	print(H1.shape)
	H2 = ObsDrift_Jacobian(t_i, Xe_n[i], Sat_2)
	H2 = H2[:, :2]
	con_num_base[i] = np.linalg.cond(H1.T @ np.linalg.inv(R) @ H1)
	con_num_deg[i] = np.linalg.cond(H2.T @ np.linalg.inv(R) @ H2)
# ax5.plot(t, con_num_deg, label = "Condition Number (Degenerate)")
# print(con_num_base)
# print(con_num_deg)
ax5.plot(t, con_num_base, label= "Condition Number(Base)")
ax5.set_xlabel('t(s)')
ax5.set_ylabel('Condition Number')
ax5.set_title('Condition Number Comparison')
ax5.legend()
fig5.savefig("results/exp2_5.png")


fig6, ax6 = plt.subplots()
ax6.plot(t, con_num_deg, label= "Condition Number(Deg)")
ax6.set_xlabel('t(s)')
ax6.set_ylabel('Condition Number')
ax6.set_title('Condition Number Comparison')
ax6.legend()
fig6.savefig("results/exp2_6.png")




plt.show()