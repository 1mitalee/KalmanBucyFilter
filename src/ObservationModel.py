import numpy as np  

def ObsDrift(t, Y, X, S):

# X: state of the system 
# S: satellite position 4X2
	x = X[0]
	y = X[1]

	p = np.array([x,y])

	P = np.tile(p, (4, 1))

	disp = P - s # dim: 4X2

	dist = np.linalg.norm(diff, axis=1)
	dist = dist.reshape(4,1)

	h = dist + X[5]
	# output: 4X1 
	return h 

def ObsDrift_Jacobian(t, Y, X, S):
	x = X[0]
	y = X[1]

	p = np.array([x,y])

	P = M = np.tile(p, (4, 1))

	disp = P - s # dim: 4X2

	dist = np.linalg.norm(diff, axis=1)
	dist = dist.reshape(4,1)

	H = P / dist

	zeros = np.zeros((4, 2))    
	ones  = np.ones((4, 1))

	H = np.hstack([H, zeros, ones])

	# Output 4 X 5
	return H

def ObsDrift_Lin(t, Y, X, S, X_hat):

# X: state of the system 
# S: satellite position 4X2
# X_hat: linearization point
	error = X - X_hat
	error = error.reshape(5,1)
	h_hat = ObsDrift(X_hat) + ObsDrift_Jacobian(X_hat, S)@(error)

	return h_hat


	

