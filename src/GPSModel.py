import numpy as np  

# System Parameters

X0 = np.array([0,0,10,2,50])

t_span = 60
dt = 0.01

A = np.array([[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
G = np.array([[0,0,0], [0,0,0], [0.5,0,0], [0,0.5,0], [0,0,0.1]])

Q = G @ np.transpose(G)

R_sqrt = 5 * np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1 ,0],[0, 0, 0, 1]])
R = 25 * np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1 ,0],[0, 0, 0, 1]])

Sat_1 = np.array([[20000,0],[0,20000],[-20000,0],[0,-20000]])
Sat_2 = np.array([[20000, 0],[21000,2000],[22000,-2000],[23000,1000]])

P0 = np.tile((X0[0],X0[1]),(4,1))
# Y0 = np.linalg.norm(P0-S1, axis=1)
# Y0 = Y0.reshape((1,4))

Xe0 = np.array([20, -10, 0, 0, 0])
S0 = np.diag(np.array([10000, 10000, 400, 400, 40000]))



# State Model
def F(t):
	return A

def C(t):
	return G	

def ObsDrift(t, X, S):

# X: state of the system 
# S: satellite position 4X2
	x = X[0]
	y = X[1]

	p = np.array([x,y])

	P = np.tile(p, (4, 1))

	disp = P - S # dim: 4X2

	dist = np.linalg.norm(disp, axis=1)
	dist = dist.reshape(4,1)

	h = dist + X[4]
	# output: 4X1 
	return h 

def h(t, X):
	return ObsDrift(t, X, Sat_1)


def ObsDrift_Jacobian(t, X, S):
	x = X[0]
	y = X[1]

	p = np.array([x,y])

	P = M = np.tile(p, (4, 1))
	# print(P)

	disp = P - S # dim: 4X2
	# print(disp)

	dist = np.linalg.norm(disp, axis=1)
	dist = dist.reshape(4,1)
	# print(dist)

	H = disp / dist

	# print(H)

	zeros = np.zeros((4, 2))    
	ones  = np.ones((4, 1))

	H = np.hstack([H, zeros, ones])

	# Output 4 X 5
	return H

def H(t, X):
	return ObsDrift_Jacobian(t, X, Sat_1)

# def ObsDrift_Lin(t, X, S, X_hat):

# # X: state of the system 
# # S: satellite position 4X2
# # X_hat: linearization point
# 	error = X - X_hat
# 	error = error.reshape(5,1)
# 	h_hat = ObsDrift(X_hat) + ObsDrift_Jacobian(X_hat, S)@(error)

# 	return h_hat

def D(t):
	return R_sqrt


	

