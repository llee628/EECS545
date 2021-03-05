# EECS 545 HW3 Q5
# Your name: Leo (Wei-chung) Lee

# Install scikit-learn package if necessary:
# pip install -U scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import pdb


def readMatrix(filename: str):
    # Use the code below to read files
    with open(filename, 'r') as fd:
        hdr = fd.readline()
        rows, cols = [int(s) for s in fd.readline().strip().split()]
        tokens = fd.readline().strip().split()
        matrix = np.zeros((rows, cols))
        Y = []  
        for i, line in enumerate(fd):
            nums = [int(x) for x in line.strip().split()]
            Y.append(nums[0])
            kv = np.array(nums[1:])
            k = np.cumsum(kv[:-1:2])
            v = kv[1::2]
            matrix[i, k] = v
        return matrix, tokens, np.array(Y)


def evaluate(output, label) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    error = float((output != label).sum()) * 1. / len(output)
    print('Error: {:2.4f}%'.format(100 * error))

    return error

def part_b():
    data_list = [50,100,200,400,800,1400]
    error_list = []

    for N in data_list:
        print("N =",N)
        filename = 'q5_data/MATRIX.TRAIN.'
        filename = filename + str(N)
        dataMatrix_train, tokenlist, category_train = readMatrix(filename)
        dataMatrix_test, tokenlist, category_test = readMatrix('q5_data/MATRIX.TEST')

        
        
        
        #
        #Train
        svc = LinearSVC()
        svc.fit(dataMatrix_train, category_train)
        # Find support vector
        decision_function = svc.decision_function(dataMatrix_train)
        support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
        #breakpoint()

        # Test and evluate
        prediction = svc.predict(dataMatrix_test)
        error = evaluate(prediction, category_test)
        error_list.append(error)
        print("numbers of support vector:",support_vector_indices.size)

    plt.plot(data_list,error_list, 's-')
    plt.xlabel('training set size')
    plt.ylabel('test set error')
    plt.title("5(b)")
    plt.savefig('5_b.png',format='png')
    #plt.show()




def main():
    # Load files
    # Note 1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note 2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    dataMatrix_train, tokenlist, category_train = readMatrix('q5_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q5_data/MATRIX.TEST')

    #Part (a)
    # Train
    #raise NotImplementedError("Implement your code here.")
    
    svc = LinearSVC()
    svc.fit(dataMatrix_train, category_train)
    print("Part(a)")

    # Test and evluate
    #prediction = np.ones(dataMatrix_test.shape[0])  # TODO: This is a dummy prediction.
    prediction = svc.predict(dataMatrix_test)
    evaluate(prediction, category_test)
    print('')

    #Part (b)
    print("part(b)")
    part_b()


if __name__ == '__main__':
    main()
