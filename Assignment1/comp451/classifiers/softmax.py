from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg, regtype='L2'):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    - regtype: Regularization type: L1 or L2

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    r = scores.shape[0]
    softmax = np.zeros_like(scores)
    exps = np.exp(scores)
    for i in range(r):
        softmax[i] = exps[i] / np.sum(exps[i])
    logs = -np.log(softmax[range(X.shape[0]),y])
    loss = np.sum(logs) / X.shape[0]
    if(regtype =='L2'):
        loss += reg * np.sum(W*W)
    if(regtype =='L1'):
        loss += reg * np.sum(np.absolute(W))
    
    softmax[np.arange(X.shape[0]), y] -= 1
 
    for i in range(X.shape[0]):
        for j in range( X.shape[1]):
            for k in range(W.shape[1]):
                dW[j, k] += X[i, j] * softmax[i, k] 

    dW /=  X.shape[0]
    dW += reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, regtype='L2'):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    r = scores.shape[0]
    exps = np.exp(scores)
    sums = np.sum(exps, axis=1)
    softmax = exps[:] / sums[:, None]
    loss -= np.sum(np.log(softmax[np.arange(X.shape[0]), y]))
    loss /= X.shape[0]
    if(regtype =='L2'):
        loss += reg * np.sum(W*W)
    if(regtype =='L1'):
        loss += reg * np.sum(np.absolute(W))

    softmax[np.arange(X.shape[0]), y] -= 1
    dW = X.T.dot(softmax)/ X.shape[0]
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
