# Extended Kalman Bucy Filter
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))


import numpy as np 
from GPSModel import *
from System import *

class Model_EKBF:

	def __init__(self, system, Xe0, S0):
		self.system	=system
		self.Xe0	=Xe0
		self.S0		= S0

	def Riccatifunc(self, t, S_prev, xe_prev):
	    F = self.system.F(t)
	    H = self.system.H(t, xe_prev)
	    # R = self.system.D(t) @ np.transpose(self.system.D(t))
	
	    # print("F:\n", F)
	    # print("H:\n", H)
	    # print("R:\n", R)
	    # print("S_prev:\n", S_prev)
	    # print("S@H.T@inv(R)@H@S:\n", S_prev @ H.T @ np.linalg.inv(R) @ H @ S_prev)
	    # print("F@S + S@F.T:\n", F @ S_prev + S_prev @ F.T)
	    Ricc_func = F @ S_prev + S_prev @ F.T - S_prev @ H.T @ np.linalg.inv(R) @ H @ S_prev + Q
	    return Ricc_func

	def RicattiUpdate(self, t, S_prev, xe_prev, h):
		# print(h)
		k1 = self.Riccatifunc(t, S_prev, xe_prev)# (f(t,y))
		# print(k1)
		k2 = self.Riccatifunc(t+h/2, S_prev+k1*(h/2), xe_prev)# (f(t+h/2, y + k1*h/2))
		# print(k2)
		k3 = self.Riccatifunc(t+h/2, S_prev+k2*(h/2), xe_prev)# (f(t+h/2, y + k2*h/2))
		# print(k3)
		k4 = self.Riccatifunc(t+h, S_prev+k3*(h), xe_prev)# (f(t+h, y + k3*h))
		# print(k4)
		S_diff = (h/6)*(k1 + 2*k2 + 2*k3 + k4)
		# print(S_diff)
		S_new = S_prev + S_diff
		S_new = 0.5*(S_new + S_new.T)
		# print(S_new)
		return S_new

	def EstimateUpdate(self, t, S_prev, xe_prev, y_diff, h):
		sys = self.system
		# R = self.system.D(t)@np.transpose(self.system.D(t))
		# CORRECT THE STEP BELOW
		# R_sqrt = np.array([[1, 0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
		K =  S_prev@np.transpose(sys.H(t, xe_prev))@np.linalg.inv(R)
		# d = sys.h(t,x_prev)*sys.dt
		# print(d.shape)
		# y_diff = sys.h(t,x_prev)*h + np.reshape(R_sqrt @ diff_V, (4, 1))
		# print(diff_V.shape)
		# e = R_sqrt@diff_V
		# print(e.shape)
		# print(y_diff)
		# print(xe_prev.shape)
		# a = (sys.F(t) @ xe_prev) * sys.dt
		# b = K
		# c = y_diff - sys.h(t,xe_prev)*sys.dt
		# print(a.shape)
		# print(b.shape)
		# print(c.shape)
		# print((K @(y_diff - sys.h(t, xe_prev)) * h).shape)
		xe_new = xe_prev.reshape(5, 1) + (sys.F(t) @ xe_prev).reshape(5, 1) * h + K @ (y_diff - sys.h(t, xe_prev) * h)
		# print(xe_new.shape)
		# print(xe_new.shape)
		return xe_new

	def EKBFEstimate(self, W, V, h):
		N       = int(self.system.T_span / self.system.dt)
		Ne 		= int(N * (self.system.dt/h))
		X_n, Y_n = self.system.EulerMaruyama(W,V)

		Xe_n	= np.zeros((N+1, self.system.dim))
		Xe_n[0]	= self.Xe0
		S_n 	= np.zeros((Ne+1 ,self.system.dim, self.system.dim))	
		S_n[0]	= self.S0

		for i in range(1,Ne+1):
			t_i		= (i-1)*(h)
			S_n[i]	= self.RicattiUpdate(t_i, S_n[i-1], Xe_n[int((i-1)//100)], h)

			if i % 100 == 0:    
			    k = i // 100   
			    diff_Y = (Y_n[k] - Y_n[k-1]).reshape((4,1))   
			    # print(diff_Y.shape)
			    Xe_n[k] = (self.EstimateUpdate(t_i, S_n[i], Xe_n[k-1], diff_Y, self.system.dt)).flatten()
			
			

		return X_n, Y_n, Xe_n, S_n




