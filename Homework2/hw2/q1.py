import numpy as np
import math
import matplotlib.pyplot as plt
import pdb


def Sigmoid(a):
	return 1.0/(1.0 + math.exp(-a))

def logisRegression(X, Y):
	N = X.shape[0]
	D = X.shape[1]
	w = np.zeros((D,1))
	loop_max = 100
	loop = 0

	while (loop < loop_max):
		w_old = w
		grad_E = np.zeros(D)
		H = np.zeros((D,D))
		for i in range(N):
			h_i = Sigmoid((w.T @ X[i,:]))
			#breakpoint()
			H = H - h_i*(1-h_i)*(((X[i,:].reshape(1,D)).T).dot(X[i,:].reshape(1,D)))
			grad_E = grad_E + (Y[i] - h_i)*X[i,:]
			#breakpoint()
		#breakpoint()
		w = w - (np.dot(np.linalg.inv(H), grad_E)).reshape(D,1)
		#breakpoint()

		if (np.linalg.norm(w - w_old) < 1e-4):
			break

	return w

	pass



def main():
	X = np.load('q1x.npy')
	N = X.shape[0]
	Y = np.load('q1y.npy')
	X = np.concatenate((np.ones((N,1)), X), axis=1)

	#breakpoint()
	
	#(b)
	w = logisRegression(X,Y)
	print("w is :")
	print(w)

	#(c)
	plt.figure()

	for i in range(N):
		if Y[i] == 1:
			plt.scatter(x = X[i,1], y = X[i,2], c='r', marker='o')
		else:
			plt.scatter(x = X[i,1], y = X[i,2], c='b', marker='s')

	x1 = np.linspace(min(X[:,1]), max(X[:,1]), num=100)
	x2 = -w[0]/w[2] - w[1]/w[2]*x1
	plt.plot(x1,x2, c='black')

	plt.xlabel('$x_1$')
	plt.ylabel('$x_2$')
	plt.title('q1(c)')
	plt.show()
	#plt.savefig('q1_c.png')

	#breakpoint()


if __name__ == "__main__":
	main()