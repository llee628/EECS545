import numpy as np
import matplotlib.pyplot as plt



def softmax_prob(w,x):
	p_Dw = np.exp(np.dot(w,x.T))
	p_Dw = p_Dw/np.sum(p_Dw)

	#breakpoint()
	return p_Dw.reshape(w.shape[0],1)
	


def grad_ascent(X,Y):
	N = X.shape[0]
	M = X.shape[1]
	K = len(np.unique(Y))
	alpha = 0.0005
	#breakpoint()
	w = np.zeros((K,M))
	#loop = 0
	loop_max = 5000

	for loop in range(loop_max):
		grad_w = np.zeros((K,M))
		w_old = w

		for i in range(N):
			p_Dw = softmax_prob(w,X[i,:])
			Ind = np.zeros((K,1), dtype=int)
			Ind[int(Y[i,0]) - 1] = 1
			
			grad_w = grad_w + X[i,:]*(Ind - p_Dw)
			#breakpoint()
		
		w = w + alpha*grad_w
		w[K-1,:] = 0
		if (np.linalg.norm(w - w_old) < 1e-3):
			break
		#loop += 1
		#print(w)

	#breakpoint()
	print(loop)

	return w

	


def main():
	#Load data
	q2_data = np.load('q2_data.npz')
	q2x_train = q2_data['q2x_train']
	q2y_train = q2_data['q2y_train']
	q2x_test = q2_data['q2x_test']
	q2y_test = q2_data['q2y_test']
	#breakpoint()

	w = grad_ascent(q2x_train, q2y_train)
	#breakpoint()

	#accuracy
	N_test = q2x_test.shape[0]
	correct_num = 0

	for i in range(N_test):
		prob = softmax_prob(w,q2x_test[i,:])
		k = np.argmax(prob)
		k = k + 1
		if (k == q2y_test[i]):
			correct_num += 1

	
	correct_rate = 1.0*correct_num/N_test
	print('The accuracy is: ',100*correct_rate,'%')
	




if __name__ == "__main__":
	main()




