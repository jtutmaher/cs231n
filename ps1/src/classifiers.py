import numpy as np

"""
--
Jake Tutmaher
September 20, 2016

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

class svm:
    def __init__(self):
        pass

    def svm_loss_naive(self,W,X,y,reg):
        dW = np.zeros(W.shape)  # initialize the gradient as zero

        # compute the loss and the gradient
        num_classes = W.shape[1]
        num_train = X.shape[0]
        loss = 0.0
        for i in xrange(num_train):
            scores = X[i].dot(W)
            correct_class_score = scores[y[i]]
            for j in xrange(num_classes):
                if j == y[i]:
                    continue
                margin = scores[j] - correct_class_score + 1  # note delta = 1
                dW[i,j] = margin
                if margin > 0:
                    loss += margin

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        dW[dW<0] = 0
        loss /= num_train

        # Add regularization to the loss.
        loss += 0.5 * reg * np.sum(W * W)

        #############################################################################
        # TODO:                                                                     #
        # Compute the gradient of the loss function and store it dW.                #
        # Rather that first computing the loss and then computing the derivative,   #
        # it may be simpler to compute the derivative at the same time that the     #
        # loss is being computed. As a result you may need to modify some of the    #
        # code above to compute the gradient.                                       #
        #############################################################################


        return loss, dW

