import numpy as np
import matplotlib.pyplot as plt
import pdb

def readMatrix(file):
    # Use the code below to read files
    fd = open(file, 'r')
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

def nb_train(matrix, category):
    # Implement your algorithm and return 
    state = {}
    N = matrix.shape[1]
    
    ############################
    # Implement your code here #
    ############################

    spam_matrix = np.empty((0,N))
    nonspam_matrix = np.empty((0,N))
    #breakpoint()

    for i in range(len(category)):
        if category[i] == 1:
            spam_matrix = np.vstack((spam_matrix, matrix[i,:]))
        else:
            nonspam_matrix = np.vstack((nonspam_matrix, matrix[i,:]))

    spam_words = np.sum(spam_matrix, axis=1)
    nonspam_words = np.sum(nonspam_matrix, axis=1)
    #breakpoint()
    phi_spam = (np.sum(spam_matrix, axis=0) + 1)/(np.sum(spam_matrix) + N)
    phi_nonspam = (np.sum(nonspam_matrix, axis=0) + 1)/(np.sum(nonspam_matrix) + N)
    phi_y = sum(category)*1.0/len(category)

    #breakpoint()
    state['spam'] = phi_spam
    state['nonspam'] = phi_nonspam
    state['phiy'] = phi_y


    
    return state

def nb_test(matrix, state):
    # Classify each email in the test set (each row of the document matrix) as 1 for SPAM and 0 for NON-SPAM
    output = np.zeros(matrix.shape[0])
    d = matrix.shape[1]
    
    
    ############################
    # Implement your code here #
    ############################
    p_D_spam = matrix @ np.log(state['spam']).reshape(d,1)
    p_D_nonspam = np.dot(matrix, np.log(state['nonspam'].reshape(d,1)))
    p_y = state['phiy']
    #breakpoint()

    target_i = ((p_D_spam + np.log(p_y)) > (p_D_nonspam + np.log(1-p_y)))
    target_i = target_i.reshape(p_D_spam.shape[0])
    output[target_i] = 1
    
    return output

def evaluate(output, label):
    # Use the code below to obtain the accuracy of your algorithm
    error = (output != label).sum() * 1. / len(output)
    print('Error: {:2.4f}%'.format(100*error))
    return error


def part_c():
    dataMatrix_test, tokenlist, category_test = readMatrix('q4_data/MATRIX.TEST')
    train_size = np.array([50, 100, 200, 400, 800, 1400])
    error_list = np.zeros(train_size.size)

    for i, size in enumerate(train_size):
        dataMatrix_train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN.'+ str(size))
        state = nb_train(dataMatrix_train, category_train)
        prediction = nb_test(dataMatrix_test, state)
        print("Training size:",size)
        error_list[i] = evaluate(prediction, category_test)

    plt.plot(train_size,error_list*100)
    plt.xlabel('Training Size')
    plt.ylabel('Error (%)')
    plt.title("4(c)")
    plt.show()
    

def main():
    # Note1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    # Note3: The shape of the data matrix (document matrix): (number of emails) by (number of tokens)

    # Load files
    dataMatrix_train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q4_data/MATRIX.TEST')

    # Train
    state = nb_train(dataMatrix_train, category_train)

    # Test and evluate
    prediction = nb_test(dataMatrix_test, state)
    evaluate(prediction, category_test)

    #(b)
    indicative_list = np.log(state['spam']/state['nonspam'])
    most_five_token_idx = []
    most_five_token = []

    for i in range(5):
        idx = np.argmax(indicative_list)
        most_five_token_idx.append(idx)
        indicative_list[idx] = 0


    for idx in most_five_token_idx:
        most_five_token.append(tokenlist[idx])

    print("The 5 tokens that are most indicative of the SPAM class:")
    print(most_five_token)

    #(c)
    part_c()



if __name__ == "__main__":
    main()
    #part_c()
        
