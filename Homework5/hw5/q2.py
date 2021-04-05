import numpy as np
import matplotlib.pyplot as plt
import time
import pdb

def validate_PCA(states, train_data):
  from sklearn.decomposition import PCA
  pca = PCA()
  pca.fit(train_data)
  true_matrix = pca.components_.T
  true_ev = pca.explained_variance_
  
  output_matrix = states['transform_matrix']
  error = np.mean(np.abs(np.abs(true_matrix) - np.abs(output_matrix)) / np.abs(true_matrix))
  if error > 0.01:
    print('Matrix is wrong! Error=',error)
  else:
    print('Matrix is correct! Error=', error)

  output_ev = states['eigen_vals']
  error = np.mean(np.abs(true_ev - output_ev) / true_ev)
  if error > 0.01:
    print('Variance is wrong! Error=', error)
  else:
    print('Variance is correct! Error=', error)

def train_PCA(train_data):
  ##### TODO: Implement here!! #####
  # Note: do NOT use sklearn here!
  # Hint: np.linalg.eig() might be useful

  N = train_data.shape[0]
  #data preprocess
  normalized_data = train_data - np.mean(train_data, axis=0, keepdims=True)
  #normalized_data /= np.sqrt(np.var(train_data, axis=0, keepdims=True))

  cov = np.cov(normalized_data.T)
  eigen_vals, eigen_vec = np.linalg.eig(cov)
  ordered_eigen_index = np.argsort(eigen_vals)
  ordered_eigen_index = np.flip(ordered_eigen_index)
  transform_matrix = eigen_vec[:, ordered_eigen_index]
  eigen_vals = eigen_vals[ordered_eigen_index]
  #breakpoint()



  states = {
      'transform_matrix': transform_matrix,
      'eigen_vals': eigen_vals
  }
  ##### TODO: Implement here!! #####
  return states, ordered_eigen_index

# Load data
start = time.time()
images = np.load('q2data/q2.npy')
num_data, h, w = images.shape
train_data = images.reshape(num_data, -1)
#breakpoint()

states, ordered_eigen_index = train_PCA(train_data)
print('training time = %.1f sec'%(time.time() - start))

validate_PCA(states, train_data)

print("First 10 principal components:")
print(states['eigen_vals'][:10])

plt.figure(1)
plt.title('Eigenvalues')
plt.plot(ordered_eigen_index, states['eigen_vals'])
plt.show()

#(c)
eigen_vectors = states['transform_matrix']
plt.subplot(2, 5, 1)
mean_image = np.mean(images, axis=0)
plt.imshow(mean_image)

for i in range(9):
  plt.subplot(2, 5, i+2)
  eigen_image = eigen_vectors[:, i].reshape(h, w)
  plt.imshow(eigen_image)

plt.show()

#(d)
eigen_vals = states['eigen_vals']
cumulative_eval = np.cumsum(eigen_vals)
variance_ratio = cumulative_eval/cumulative_eval[-1]
pc_num1 = np.argwhere(variance_ratio >= 0.95)[0,0] + 1
pc_num2 = np.argwhere(variance_ratio >= 0.99)[0,0] + 1
print('')
print('2(d)')
print("We need {:2d} principal components to represent 95% of the total variance".format(pc_num1))
print("The percentage of reduction in dimension is {:2.2f} %".format((1.0 - pc_num1/(h*w))*100))
print('')
print("We need {:2d} principal components to represent 99% of the total variance".format(pc_num2))
print("The percentage of reduction in dimension is {:2.2f} %".format((1.0 - pc_num2/(h*w))*100))


#breakpoint()







