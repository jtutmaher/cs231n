"""
------------------------------------------------------------------
Jake Tutmaher
September 20, 2016
------------------------------------------------------------------
"""

import numpy as np

"""
K - nearest neighbors (KNN) Class, PSET 1: CS231. KNN is a "memory
only" approach - analogous to a lookup method for a given training
set. Typical accuracies of 40 percent (best case).
--
"""

class knn:
    def __init__(self):
        pass

    def train(self,X,Y):
        """
        Store Training Data and Labels in Class
        :param X: 2D numpy array (floats), training images
        :param Y: 2D numpy array (ints), training labels
        :return: N/A
        """
        self.Xtr = X
        self.Ytr = Y

    def predict(self,dist,k=1):
        """
        Find Min Value Indices in distance array, return
        corresponding values for training labels
        :param dist: 2D array (floats), distance array
        :param k: Int, number of nearest neighbors
        :return: 1D array (floats), predictions
        """
        idx = np.argsort(dist,axis=1)
        predict = [np.argmax(np.bincount(self.Ytr[idx[x,:k]]))
                   for x in range(dist.shape[0])]
        return np.array(predict)

    def distance(self,X):
        """
        Compute the distance matrix for the test set
        :param X: 2D array (floats), test images
        :return: 2D array (floats), num_test x num_train
        """
        #GET NUM TEST IMAGES
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        #INITIALIZE DIST ARRAY
        darray = np.zeros((num_test,num_train))
        #COMPUTE DIST ARRAY - 1 LOOP
        for x in range(num_train):
            currIm = self.Xtr[x,:]
            slice = np.repeat(currIm[:,np.newaxis],num_test,axis=1)
            diff = np.sum(np.abs((slice - X.transpose())),axis=0)
            darray[:,x] = diff
        return darray

    def check(self,X):
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        #INITIALIZE DISTANCE ARRAY
        darray = np.zeros((num_test,num_train))
        for x in range(num_train):
            for y in range(num_test):
                darray[y,x] = np.sum(np.abs(self.Xtr[x,:]-X[y,:]))

        return darray

    def accuracy(self, Ypred, Yact):
        """
        Get Accuracy of KNN Training Run
        :param Ypred: 1D array (floats), predicted values for test
        :param Yact: 1D array (floats), actual values for test
        :return: Float, number of correct predictions
        """
        num_correct = np.sum(Ypred==Yact)
        return np.divide(num_correct,float(Yact.size))

"""
Support Vector Machine class. Generalization of a binary support vector machine
algorithm. Utilizes the generalized loss function for arbitrary classification.
"""

class svm:
    def __init__(self):
        self.model = []
        pass

    def __loss_i(self,x, y, W):
        """
        Computes total loss and gradient matrix for a single training
        note: meant as private method.
        instance
        :param y: float, Correct class
        :param W: 2D array (floats), weight matrix
        :return: float & 2D array (floats), loss for trial and gradient
                 matrix
        """
        # Predictions
        scores = x.dot(W)
        # Loss term
        margins = scores - scores[y] + 1
        margins[margins < 0] = 0
        margins[y] = 0
        # Total loss for trial
        loss = np.sum(margins)
        # Corrent for single piecewise term
        margins[margins > 0] = 1
        # Generate gradient matrix
        dW = np.outer(x, margins)
        # Scaled for row corresponding to correct class
        dW[:, y] = -np.count_nonzero(margins) * x
        return loss, dW

    def loss(self,W,X,y,reg):
        """
        Computes total loss and gradient matrix averages for all trials
        :param W: 2D array (floats), weight matrix
        :param X: 2D array (floats), feature vectors for all trials
        :param y: 1D array (floats), correct classes for all trials
        :param reg: regularization factor for l2 correction
        :return: float & 2D array (floats), loss and gradient matrix
        """
        # Get weight matrix and initialize gradients
        dW = np.zeros(W.shape)
        loss = 0.0
        # Iterate through training set
        num_train = X.shape[0]
        for i in xrange(num_train):
            L_i, dW_i = self.__loss_i(X[i], y[i], W)
            loss += L_i
            dW += dW_i
        # Average loss
        loss /= num_train
        # Average gradients as well
        dW /= num_train
        # Add regularization to the loss.
        loss += 0.5 * reg * np.sum(W * W)
        # Add regularization to the gradient
        dW += reg * W
        # Return
        return loss,dW

    def train(self,Xtr,Ytr,learning_rate=1e-7, reg=5e4,batch_size=500,classes=10):
        """
        Train SVM linear model based on training set
        :param Xtr: 2D array (floats), training data
        :param Ytr: 1D array (int), training lables
        :param learning_rate: float, correction factor to gradient
        :param reg: float, for l2 regularization
        :param batch_size: int, num images for SGD
        :param classes: int, num classes for SVM model
        :return: 2D array, final weight matrix "model"
        """
        # Training size
        num_train = Xtr.shape[0]
        # Random SVM weight matrix of small numbers
        W = np.random.randn(Xtr.shape[1], classes) * 0.0001
        loss_array = []
        # Iterate through batches
        for x in range(0,num_train,batch_size):
            batch_x = Xtr[x:x+batch_size,:]
            batch_y = Ytr[x:x+batch_size]
            # Get loss and gradient
            loss,dW = self.loss(W,batch_x,batch_y,reg)
            # Apply correction
            loss_array.append(loss)
            W -= learning_rate*dW
        # Store model for testing
        self.model = W
        return loss_array

    def predict(self,X):
        """
        Make predictions based on Current Model
        :param X: 2D array (floats), set of test/validation images
        :return: 1D array (floats), predictions
        """
        W = self.model
        y = X.dot(W)
        predict = np.argmax(y,axis=1)
        return predict

    def accuracy(self,Ypr,Yact):
        return np.mean(Ypr==Yact)

    def model(self):
        return self.model


