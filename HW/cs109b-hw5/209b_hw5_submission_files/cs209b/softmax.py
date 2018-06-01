import numpy as np
from random import shuffle
from numpy import linalg as LA

def softmax_loss_naive(W, b, X, Y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (c, n) containing weights.
    - X: A numpy array of shape (m ,n) containing a minibatch of data.
    - Y: A numpy array of shape (m, c) containing training labels using a one-hot
            encoding, y[i, ci] = 1 means that X[i,:] has label ci.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - dictionary of gradients with respect to W and b.
    """
    c = W.shape[0]
    n = W.shape[1]
    m = X.shape[0]
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    z = np.matmul(W, X.T) + np.tile(b, (1, m))
    exp_z = np.exp(z)
    exp_z = exp_z / exp_z.max(axis=0) # normalize the exponents
    exp_z_sum = np.sum(exp_z, axis=0)
    exp_z_sum = exp_z_sum.reshape((1, exp_z_sum.shape[0]))
    softmax_matrix = exp_z / np.tile(exp_z_sum, (c, 1))
    ll_matrix = -np.log(softmax_matrix)
#     pred_matrix = (softmax_matrix == softmax_matrix.max(axis=0, keepdims=1)).astype(float)
    loss = np.sum(np.multiply(Y.T, ll_matrix)) / m + 0.5*reg*np.sum(W**2)
    dW = np.matmul((softmax_matrix - Y.T), X) / m + (reg*W)
    db = np.matmul((softmax_matrix - Y.T), np.ones((m, 1)))/m

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    grads = {"dW": dW, "db": db}
    return loss, grads


def softmax_loss_vectorized(W, b, X, Y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (c, n) containing weights.
    - b: A numpy array of shape (c, 1) containing the bias elements.
    - X: A numpy array of shape (m ,n) containing a minibatch of data.
    - Y: A numpy array of shape (m, c) containing training labels using a one-hot
        encoding, y[i, ci] = 1 means that X[i,:] has label ci.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - dictionary of gradients with respect to W and b.
    """
    c = W.shape[0]
    n = W.shape[1]
    m = X.shape[0]
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    z = np.matmul(W, X.T) + np.tile(b, (1, m))
    exp_z = np.exp(z)
    exp_z = exp_z / exp_z.max(axis=0) # normalize the exponents
    exp_z_sum = np.sum(exp_z, axis=0)
    exp_z_sum = exp_z_sum.reshape((1, exp_z_sum.shape[0]))
    softmax_matrix = exp_z / np.tile(exp_z_sum, (c, 1))
    ll_matrix = -np.log(softmax_matrix)
#     pred_matrix = (softmax_matrix == softmax_matrix.max(axis=0, keepdims=1)).astype(float)
    loss = np.sum(np.multiply(Y.T, ll_matrix)) / m + 0.5*reg*np.sum(W**2)
    dW = np.matmul((softmax_matrix - Y.T), X) / m + (reg*W)
    db = np.matmul((softmax_matrix - Y.T), np.ones((m, 1)))/m

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    assert(W.shape == (c, n))
    assert(b.shape == (c, 1))
    assert(loss.shape == ())
    grads = {"dW": dW, "db": db}
    return loss, grads


def optimize(W, b, X, Y, num_iterations=100, learning_rate=1e-5, reg=0,
             batch_size=100, print_cost = False):
    """
    This function optimizes W and b by running a stochastic gradient descent algorithm using mini-batches.

    Arguments:
    W -- weights, a numpy array of size (c, n)
    b -- bias, of size (c, 1)
    X -- data of size (m, n)     (m is the number of examples)
    Y -- true "label" vector with one-hot encoding of size (m, c)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    reg -- (float) regularization strength.
    batch_size -- (integer) number of training examples to use at each step.
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights W and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization
    """
    m, n = X.shape
    c = Y.shape[1]
    costs = []

    for i in range(num_iterations):
        chose_idx = np.random.choice(m, size=batch_size)
        X_batch = X[chose_idx, :]    # should have shape (batch_size, n)
        y_batch = Y[chose_idx, :]     # should have shape (batch_size, c)

        #########################################################################
        # TODO: Implement stochastic gradient descent.                          #
        #########################################################################
        _, grads = softmax_loss_vectorized(W, b, X_batch, y_batch, reg)
        dW, db = grads["dW"], grads["db"]
        W -= learning_rate*dW
        b -= learning_rate*db
        cost, _ = softmax_loss_vectorized(W, b, X_batch, y_batch, reg)
        #########################################################################
        #                          END OF YOUR CODE                             #
        #########################################################################

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    params = {"W": W, "b": b}
    grads = {"dW": dW, "db": db}

    return params, grads, costs


def predict(W, b, X):
    '''
    Infer 0-c encoding labelss using the learned softmax regression parameters (W, b)

    Arguments:
    W -- weights, a numpy array of size (c, n)
    b -- bias, a numpy array of size (c, 1)
    X -- data of size (m, n)

    Returns:
    Y_prediction -- a numpy array containing all predictions for the examples in X with size (m,).
    '''
    c = W.shape[0]
    m = X.shape[0]
    n = X.shape[1]
    Y_prediction = np.zeros((m,))

    #############################################################################
    # TODO: Compute the scores and return a prediciton.                         #
    #############################################################################
    z = np.matmul(W, X.T) + np.tile(b, (1, m))
    exp_z = np.exp(z)
    exp_z = exp_z / exp_z.max(axis=0) # normalize the exponents
    exp_z_sum = np.sum(exp_z, axis=0)
    exp_z_sum = exp_z_sum.reshape((1, exp_z_sum.shape[0]))
    softmax_matrix = exp_z / np.tile(exp_z_sum, (c, 1))
    Y_prediction = np.argmax(softmax_matrix, axis=0)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    assert(Y_prediction.shape == (m,))
    return Y_prediction


def model(X_train, Y_train, X_val, Y_val, num_iterations = 2000, learning_rate = 1e-5, reg=0, batch_size = 100, print_cost = False):
    """
    Builds the logistic regression model.

    Arguments:
    X_train -- training set represented by a numpy array of shape (m_train, n)
    Y_train -- training labels represented by a numpy array (vector) of shape (m_train, c)
    X_val -- validation set represented by a numpy array of shape (m_val, n)
    Y_val -- validation labels represented by a numpy array (vector) of shape (m_val, c)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()

    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    m_train = X_train.shape[0]
    m_val = X_val.shape[0]
    n = X_train.shape[1]
    c = Y_train.shape[1]

    #############################################################################
    # TODO: Put the model together: initialize parameters, optimize, retrieve   #
    # the solution and determine predictions on the training and test sets.     #
    #############################################################################
    W = np.random.randn(c, n) * (1/n)
    b = np.random.randn(c, 1) * (1/n)
    params, grads, costs = optimize(W, b, X_train, Y_train, num_iterations, learning_rate, reg, batch_size, print_cost)
    W = params['W']
    b = params['b']
    Y_prediction_train = predict(W, b, X_train)
    Y_prediction_val = predict(W, b, X_val)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    # Print train/test Errors
    y_train = np.argmax(Y_train, axis=1)   # convert one-hot encoding to class number
    y_val = np.argmax(Y_val, axis=1)
    train_acc = np.mean(Y_prediction_train == y_train)   # accuracy
    val_acc= np.mean(Y_prediction_val == y_val)
    print("train accuracy: {}".format(train_acc))
    print("validation accuracy: {}".format(val_acc))
    d = {"costs": costs,               # save the classifier parameters
         "Y_pred_test": val_acc,
         "Y_pred_train" : train_acc,
         "W" : W,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d
