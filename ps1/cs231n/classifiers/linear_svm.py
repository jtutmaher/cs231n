import numpy as np
from random import shuffle

def L_i_vectorize(x,y,W):
  #PREDICTIONS
  scores = x.dot(W)
  #LOSS FUNCTION
  margins = scores-scores[y]+1
  margins[margins<0] = 0
  margins[y] = 0
  loss = np.sum(margins)
  #GRADIENT MATRIX
  margins[margins>0] = 1
  dW = np.outer(x,margins)
  dW[:,y] = -np.count_nonzero(margins) * x
  return loss, dW

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  #num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    L_i,dW_i = L_i_vectorize(X[i],y[i],W)
    loss+=L_i
    dW+=dW_i
    #scores = X[i].dot(W)
    #correct_class_score = scores[y[i]]
    #for j in xrange(num_classes):
    #  if j == y[i]: # If correct class
    #    continue
    #  margin = scores[j] - correct_class_score + 1 # note delta = 1
    #  if margin > 0:
    #    loss += margin
    #    dW[:, j] += X[i]
    #    dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Average gradients as well
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Add regularization to the gradient
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  W = W.T
  num_train = X.shape[0]
  W_tot = np.repeat(W[:,:,np.newaxis],num_train,axis=2)
  #test = W_tot[0,:,:]
  y_predict = np.einsum("ijk,ij->ik",X,W_tot)
  print y_predict.shape

  L = 0
  dW = np.zeros(W_tot.shape)

  return L, dW

def helloworld():
  print("here")
