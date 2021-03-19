from builtins import range
import numpy as np
import math
import pdb


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in the variable out. #
    ###########################################################################
    out = x.dot(w) + b


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = dout.dot(w.T)
    dw = x.T.dot(dout)
    db = np.sum(dout, axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0,x)
    #breakpoint()


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.where(x>0, dout, 0)
    #breakpoint()


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We filter each input with F different filters, where each filter
    spans all C channels and has height H' and width W'.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, H', W')
    Returns a tuple of:
    - out: Output data, of shape (N, F, HH, WW) where H' and W' are given by
      HH = H - H' + 1
      WW = W - W' + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass based on the definition  #
    # of Y in Q1(c).                                                          #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, H_p, W_p = w.shape
    HH = H - H_p + 1
    WW = W- W_p + 1
    out = np.zeros((N, F, HH, WW))
    k = w[:,:,::-1,::-1]
    #breakpoint()

    for n in range(N):
        for f in range(F):
            for i in range(HH):
                for j in range(WW):
                    out[n,f,i,j] = np.sum(x[n, :, i:i+H_p, j:j+W_p]*w[f])

    #breakpoint()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w = cache
    N, C, H, W = x.shape
    F, C, H_p, W_p = w.shape
    HH = H - H_p + 1
    WW = W - W_p + 1
    #pad = 0
    #x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    for n in range(N):
        for f in range(F):
            for i in range(HH):
                for j in range(WW):
                    dx[n,:,i:i+H_p,j:j+W_p] += w[f]*dout[n,f,i,j]
                    dw[f] += x[n,:,i:i+H_p,j:j+W_p]*dout[n,f,i,j]
    #breakpoint()


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here and we can assume that the dimension of
    input and stride will not cause problem here. Output size is given by
    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_prime = 1 + (H - pool_height) // stride
    W_prime = 1 + (W - pool_width) // stride
    #breakpoint()
    out = np.zeros((N,C,H_prime,W_prime))
    
    for n in range(N):
        for c in range(C):
            for i in range(H_prime):
                for j in range(W_prime):
                    out[n,c,i,j] = np.amax(x[n,c,i*stride:i*stride+pool_height, j*stride:j*stride+pool_width])


    # for n in range(1, N):
    #     out_n = np.zeros((C,H_prime,W_prime))
    #     for i in range(H_prime):
    #         for j in range(W_prime):
    #             out_n[:,i,j] = np.amax(x[n,:,i*stride:i*stride+pool_height, j*stride:j*stride+pool_width])
    #     breakpoint()
    #     out = np.stack((out,out_n))
    #     breakpoint()

    # breakpoint()



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x = cache[0]
    pool_param = cache[1]
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_prime = 1 + (H - pool_height) // stride
    W_prime = 1 + (W - pool_width) // stride

    #dx = np.zeros((N,C,H_prime,W_prime))
    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(H_prime):
                for j in range(W_prime):
                    window = x[n,c,i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
                    ind = np.unravel_index(np.argmax(window, axis=None), window.shape)
                    window = np.zeros_like(window)
                    window[ind] = 1.0
                    dx[n,c,i*stride:i*stride+pool_height, j*stride:j*stride+pool_width] += dout[n,c,i,j]*window

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the j-th
      class for the i-th input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the cross-entropy loss averaged over N samples.
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Implement the softmax loss
    ###########################################################################
    N = x.shape[0]
    temp = np.exp(x)
    sigma_temp = np.sum(temp, axis=1, keepdims=True)
    p = temp/sigma_temp

    loss = 0
    Indicator = np.zeros_like(x)

    for i in range(N):
        Indicator[i, y[i]] = 1
        loss += np.log(p[i, y[i]])

    loss = -loss/N
    dx = (p - Indicator)/N

    #loss = -(np.sum(y.reshape(N,1)*np.log(p)))
    #loss = loss/N

    #dx = (p - y.reshape(N,1))/N
    


    

    #breakpoint()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
