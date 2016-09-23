"""
------------------------------------------------------------------
Jake Tutmaher
September 20, 2016
------------------------------------------------------------------
"""

import numpy as np

"""
K - nearest neighbors (KNN) class. KNN is a "memory only"
approach - analogous to a lookup method for a given training
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
Support vector machine class. Generalization of a binary support vector machine
algorithm. Utilizes the generalized loss function for arbitrary classification.
"""

class svm:
    def __init__(self,featurelength,numclasses):
        self.weights = np.random.rand(featurelength,numclasses)*0.0001
        self.model = np.zeros((featurelength,numclasses))
        return

    def train(self,Xtr,Ytr,learning_rate=1e-7, reg=5e4,batch_size=100,classes=10):
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
        # Training size and loss array initialization
        num_train = Xtr.shape[0]
        Weights = np.copy(self.weights)
        loss_array = []
        # Iterate through batches - train model
        for x in range(0,num_train,batch_size):
            batch_x = Xtr[x:x+batch_size,:]
            batch_y = Ytr[x:x+batch_size]
            loss,dW = self.loss(Weights,batch_x,batch_y,reg)
            loss_array.append(loss)
            Weights -= learning_rate*dW
        # Store model for testing
        self.model = Weights
        return loss_array

    def loss(self,W,X,y,reg):
        """
        Computes total loss and gradient matrix averages for all trials
        :param W: 2D array (floats), weight matrix
        :param X: 2D array (floats), feature vectors for all trials
        :param y: 1D array (floats), correct classes for all trials
        :param reg: regularization factor for l2 correction
        :return: float & 2D array (floats), loss and gradient matrix
        """
        # Iterate through training set
        num_train = X.shape[0]
        loss,dW = self.__batch_gradient(X,y,W)
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

    def predict(self,X):
        """
        Make predictions based on Current Model
        :param X: 2D array (floats), set of test/validation images
        :return: 1D array (floats), predictions
        """
        y = X.dot(self.model)
        predict = np.argmax(y,axis=1)
        return predict

    def weights(self):
        return self.weights

    def accuracy(self,Ypr,Yact):
        return np.mean(Ypr==Yact)

    def model(self):
        return self.model

    def __batch_gradient(self,X,y,W):
        """
        Computes total loss and gradient matrix for a single training
        note: meant as private method.
        instance
        :param y: float, Correct class
        :param W: 2D array (floats), weight matrix
        :return: float & 2D array (floats), loss for trial and gradient
                 matrix
        """
        #Training Size
        num_train = X.shape[0]
        # Predictions
        scores = X.dot(W)
        # Loss term
        correct_scores = scores[np.arange(num_train),y]
        # Compute loss function for all terms
        margins = scores - correct_scores[:,None] + 1
        margins[margins < 0] = 0
        margins[np.arange(num_train),y] = 0
        # Total loss for trial
        loss = np.sum(margins)
        # Generate Gradient scaling terms
        new_margins = np.copy(margins)
        # Set finite margins to 1 - counting function
        new_margins[new_margins > 0] = 1
        new_margins[np.arange(num_train),y] = - np.sum(new_margins,axis=1)
        # Generate gradient matrix
        dW = X.T.dot(new_margins)

        return loss, dW

"""
Softmax classifier
"""

class softmax:
    def __init__(self,featurelength,classes):
        """
        Initialize weights and model for softmax class
        :param featurelength: float, length of feature vectors
        :param classes: float, number of classes
        """
        self.W = np.random.randn(featurelength,classes)*0.0001
        self.model = np.zeros((featurelength,classes))
        return

    def train(self,Xtr,Ytr,learning_rate=1e-7, reg=5e4,batch_size=100,grad="batch"):
        """
        Train softmax classifier based on training set
        :param Xtr: 2D array (floats), training data
        :param Ytr: 1D array (int), training lables
        :param learning_rate: float, correction factor to gradient
        :param reg: float, for l2 regularization
        :param batch_size: int, num images for SGD
        :param classes: int, num classes for SVM model
        :return: 2D array, final weight matrix "model"
        """
        # Training size and loss array initialization
        num_train = Xtr.shape[0]
        Weights = np.copy(self.W)
        loss_array = []
        # Iterate through batches - train model
        for x in range(0,num_train,batch_size):
            batch_x = Xtr[x:x+batch_size,:]
            batch_y = Ytr[x:x+batch_size]
            loss,dW = self.cross_entropy_loss(Weights,batch_x,batch_y,reg,grad)
            loss_array.append(loss)
            Weights -= learning_rate*dW
        # Store model for testing
        self.model = Weights
        return loss_array

    def predict(self,X):
        """
        Make predictions based on Current Model
        :param X: 2D array (floats), set of test/validation images
        :return: 1D array (floats), predictions
        """
        y = X.dot(self.model)
        predict = np.argmax(y,axis=1)
        return predict

    def cross_entropy_loss(self,W,X, y, reg,grad):
        """
        Cross entropy loss and gradient descent method
        :param W: 2D array (floats), initial weights
        :param X: 2D array (floats), batch feature vectors
        :param y: 1D array (int), list of classes
        :param reg: float, regularization factor
        :param grad: string, gradient descent method
        :return: float & 2D array (floats), batch loss and gradient
                 matrix
        """
        # Determine number of training data
        num_train = len(y)
        # Calculate softmax loss elements
        ypred = X.dot(W)
        numerator = np.exp(ypred)
        denominator = np.sum(numerator, axis=1)
        # Divide 2d array (numerators) by 1d array (denominators)
        ypred_softmax = numerator / denominator[:, None]
        L_i = np.log(ypred_softmax)
        # Calculate total loss
        loss = -np.sum([L_i[i, y[i]] for i in range(num_train)])
        # Calculate gradient
        if grad=="stochastic":
            dW = self.__stochastic_gradient(ypred_softmax, y, W, X)
        else:
            dW = self.__batch_gradient(ypred_softmax, y, W, X)
        # Average loss over training set
        loss /= num_train
        # Get regularization
        loss += 0.5 * reg * np.sum(W * W)
        # Apply regularization correction
        dW += reg * W

        return loss, dW

    def weights(self):
        return self.weights

    def accuracy(self, Ypr, Yact):
        return np.mean(Ypr == Yact)

    def model(self):
        return self.model

    def __batch_gradient(self,ypred, yact, W, X):
        """
        Batch gradient method - average gradient over a number of
        feature vectors - no regularization
        :param ypred: 2D array (floats), predicted (softmax normalized) classes
        :param yact: 1D array (int), actual classes
        :param W: 2D array (floats), current weight matrix
        :param X: 2D array (floats), batch feature vectors
        :return: 2D array (floats), batch averaged gradient correction
        """
        # Determine batch number
        num_train = len(yact)
        # Construct actuals matrix via one-hot notation
        yact_mat = np.zeros(ypred.shape)
        yact_mat[np.arange(num_train),yact] = 1
        # Compute scaling coefficients - from gradient of loss function
        scale = ypred - yact_mat
        dW = X.T.dot(scale)
        # Average gradient matrix over batch data
        dW /= num_train

        return dW

    def __stochastic_gradient(self,ypred, yact, W, X):
        """
        Stochastic gradient method - gradient for random feature vector
        in batch - no regularization
        :param ypred: 2D array (floats), predicted (softmax normalized) classes
        :param yact: 1D array (int), actual classes
        :param W: 2D array (floats), current weight matrix
        :param X: 2D array (floats), batch feature vectors
        :return: 2D array (floats), batch averaged gradient correction
        """
        # Determine batch number and classes number
        num_train = len(yact)
        num_classes = ypred.shape[1]
        # Get random element
        rand = np.random.randint(0, num_train)
        # Set up one hot notation for actual data
        one_hot_vec = np.zeros(num_classes)
        one_hot_vec[yact[rand]] = 1
        # Construct gradient matrix - no averaging since stochastic
        scale = ypred[rand, :] - one_hot_vec
        dW = np.outer(X[rand, :], scale)
        return dW


