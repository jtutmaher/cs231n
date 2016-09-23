import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
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

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the gradient to zero.
  num_train = len(y)
  # Calculate softmax loss elements
  ypred = X.dot(W)
  numerator = np.exp(ypred)
  denominator = np.sum(numerator,axis=1)
  # Divide 2d array (numerators) by 1d array (denominators)
  ypred_softmax = numerator/denominator[:,None]
  L_i = np.log(ypred_softmax)
  # Calculate total loss
  loss = -np.sum([L_i[i,y[i]] for i in range(num_train)])
  # Calculate batch gradient
  dW = __batch_gradient(ypred_softmax,y,W,X)
  # Average loss over training set
  loss/=num_train
  # Get regularization
  loss += 0.5*reg*np.sum(W*W)
  # Apply regularization correction
  dW += reg * W

  return loss, dW

def __batch_gradient(ypred,yact,W,X):
  # Initialize gradient matrix
  dW = np.zeros(W.shape)
  # Determine batch number and classes number
  num_train = len(yact)
  num_classes = ypred.shape[1]
  # Iterate through training data to determine gradient matrix
  for i in range(num_train):
    # Set up one hot notation for actual data
    one_hot_vec = np.zeros(num_classes)
    one_hot_vec[yact[i]] = 1
    scale = ypred[i,:] - one_hot_vec
    dW += np.outer(X[i,:],scale)
  # Average gradient matrix over batch data
  dW /= num_train

  return dW

def __stochastic_gradient(ypred,yact,W,X):
  # Determine batch number and classes number
  num_train = len(yact)
  num_classes = ypred.shape[1]
  # Get random element
  rand = np.random.randint(0,num_train)
  # Set up one hot notation for actual data
  one_hot_vec = np.zeros(num_classes)
  one_hot_vec[yact[rand]] = 1
  # Construct gradient matrix
  scale = ypred[rand,:] - one_hot_vec
  dW = np.outer(X[rand,:],scale)
  return dW


def softmax_loss_vectorized(W, X, y, reg):
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
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

