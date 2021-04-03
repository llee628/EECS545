import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.metrics import pairwise_distances  # Don't use other functions in sklearn
import pdb

def train_kmeans(train_data, initial_centroids):
  ##### TODO: Implement here!! #####
  # Hint: pairwise_distances() might be useful
  #breakpoint()
  iter_num = 50
  K = initial_centroids.shape[0]
  for i in range(iter_num):
  	d = pairwise_distances(train_data, initial_centroids)
  	c = np.argmin(d, axis=1)
  	pixel_error = np.min(d, axis=1)
  	print('iteration = {:2d}: error = {:2.2f}'.format(i+1,pixel_error.mean()))

  	for j in range(K):
  		indicator = (j == c)
  		label = train_data[indicator,:]
  		initial_centroids[j] = np.mean(label, axis=0)

  #breakpoint()

  states = {
      'centroids': initial_centroids
  }
  ##### TODO: Implement here!! #####
  return states

def test_kmeans(states, test_data):
  result = {}
  ##### TODO: Implement here!! #####
  compressed_data = test_data.copy() # TODO: modify here!
  centroids = states['centroids']
  K = centroids.shape[0]
  D_matrix = pairwise_distances(test_data, centroids)
  C = np.argmin(D_matrix, axis=1)

  for i in range(K):
  	indicator = (i == C)
  	compressed_data[indicator, :] = centroids[i]
  #breakpoint()


  #breakpoint()
  ##### TODO: Implement here!! #####
  compressed_data_show = compressed_data.reshape(512,512,3).astype(np.uint8)
  plt.imshow(compressed_data_show)
  #plt.show()
  result['pixel-error'] = calculate_error(test_data, compressed_data)
  #breakpoint()
  return result

### DO NOT CHANGE ###
def calculate_error(data, compressed_data):
  assert data.shape == compressed_data.shape
  error = np.sqrt(np.mean(np.power(data - compressed_data, 2)))
  return error
### DO NOT CHANGE ###

# Load data
#breakpoint()
img_small = np.array(imageio.imread('q1data/mandrill-small.tiff')) # 128 x 128 x 3
img_large = np.array(imageio.imread('q1data/mandrill-large.tiff')) # 512 x 512 x 3

ndim = img_small.shape[-1]
train_data = img_small.reshape(-1, ndim).astype(float)
test_data = img_large.reshape(-1, ndim).astype(float)
#breakpoint()

# K-means
num_centroid = 16
initial_centroid_indices = [16041, 15086, 15419,  3018,  5894,  6755, 15296, 11460, 
                            10117, 11603, 11095,  6257, 16220, 10027, 11401, 13404]
initial_centroids = train_data[initial_centroid_indices, :]
states = train_kmeans(train_data, initial_centroids)
result_kmeans = test_kmeans(states, test_data)
print('Kmeans result=', result_kmeans)


from scipy.stats import multivariate_normal  # Don't use other functions in scipy

def gamma_fuc(data, pi, mu, sigma):
	K = mu.shape[0]
	N = data.shape[0]
	norm_distri = np.zeros((K,N))

	for i in range(K):
		norm_distri[i] = multivariate_normal.pdf(data, mean=mu[i], cov=sigma[i])
	#breakpoint()
	temp = pi*norm_distri
	total = np.sum(temp, axis=0, keepdims=True)
	gamma = temp/np.sum(temp, axis=0, keepdims=True)
	return gamma, total

def train_gmm(train_data, init_pi, init_mu, init_sigma):
  ##### TODO: Implement here!! #####
  # Hint: multivariate_normal() might be useful
  N = train_data.shape[0]
  K = init_mu.shape[0]

  pi = init_pi
  mu = init_mu
  sigma = init_sigma

  num_iterate = 50

  for i in range(num_iterate):
  	#E-step
  	gamma, total = gamma_fuc(train_data, pi, mu, sigma)
  	gamma_k_total = np.sum(gamma, axis=1, keepdims=True)
  	likelihood = np.log(total)
  	likelihood = np.sum(likelihood)
  	print('iteration={:2d}: log-likelihood={:6.1f}'.format(i+1, likelihood))

  	#M-step
  	pi = gamma_k_total/N
  	mu = gamma.dot(train_data)/gamma_k_total

  	for j in range(K):
  		x_mean_diff = train_data - mu[[j]]
  		sigma_sum = 0
  		for k in range(N):
  			sigma_sum += gamma[j,k]*((x_mean_diff[[k]].T).dot(x_mean_diff[[k]]))
  		sigma[j] = sigma_sum/gamma_k_total[j]


  init_pi = pi
  init_mu = mu
  init_sigma = sigma
  states = {
      'pi': init_pi,
      'mu': init_mu,
      'sigma': init_sigma,
  }
  ##### TODO: Implement here!! #####
  return states

def test_gmm(states, test_data):
  result = {}
  ##### TODO: Implement here!! #####
  pi = states['pi']
  mu = states['mu']
  sigma = states['sigma']
  K = mu.shape[0]
  gamma, _ = gamma_fuc(test_data, pi, mu, sigma)
  #compressed_data = test_data.copy()

  compressed_data = np.dot(gamma.T, mu)
  plt.imshow(compressed_data.reshape(512,512,3).astype(np.uint8))
  #plt.show()

  ##### TODO: Implement here!! #####
  result['pixel-error'] = calculate_error(test_data, compressed_data)
  return result

# GMM
num_centroid = 5
init_pi = np.ones((num_centroid, 1)) / num_centroid
init_mu = initial_centroids[:num_centroid, :]
init_sigma = np.tile(np.identity(ndim), [num_centroid, 1, 1])*1000.

#breakpoint()
states = train_gmm(train_data, init_pi, init_mu, init_sigma)
#breakpoint()
result_gmm = test_gmm(states, test_data)
print('GMM result=', result_gmm)

print('Means = ')
print(states['mu'])
print('Covariant Matrices:')
print(states['sigma'])

