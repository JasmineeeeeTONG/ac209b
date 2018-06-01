import numpy as np
from cs209b.optim import *


def loss_2layer_net(params, X, Y=None, reg=0.0, dropout=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - params: Network parameters
    - X: Input data of shape (m, n0). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      a one-hot enconding vector of size c. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.
    - dropout: the dropout probability on the input and hidden layers. If
      droput=0, there is no dropout and neurons are always active.

    Returns:
    If Y is None, return a matrix scores of shape (m, c) where A2[i, c] is
    the score of class c on input X[i].

    If Y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function.
    """
    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    m, n0 = X.shape      # m: number of training examples; n0: input dimension
    n1 = W1.shape[0]     # n1: input dimension to hidden layer
    c = W2.shape[0]      # c: output dimension
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in A2, which should be an array of shape (m, c).         #
    # Eventually, you will have to account to the dropout prob.                 #
    #############################################################################
    # Compute the forward pass

    # === input - hidden ===
    D0 = np.random.choice([0, 1], size=X.shape, p=[dropout, 1-dropout]) # dropout probability mask
    X_D0 = np.multiply(X, D0) # input after dropout
    z1 = (np.matmul(W1, X_D0.T) + np.tile(b1, (1, m)))/(1-dropout) # layer1 linear sum
    z1_zero_stack = np.zeros((2, z1.shape[0], z1.shape[1]))
    z1_zero_stack[0] = z1
    z1_zero_stack[1] = np.zeros(z1.shape)
    a1 = np.max(z1_zero_stack, axis=0) # layer1 activation - ReLU

    # === hidden - output ===
    D1 = np.random.choice([0, 1], size=a1.shape, p=[dropout, 1-dropout])
    a1_D1 = np.multiply(a1, D1) # hidden layer after dropout
    z2 = (np.matmul(W2, a1_D1) + np.tile(b2, (1, m)))/(1-dropout) # layer2 linear sum
    z2_exp = np.exp(z2)
    z2_exp = z2_exp / z2_exp.max(axis=0) # normalize the exponents
    z2_exp_sum = np.sum(z2_exp, axis=0)
    z2_exp_sum = z2_exp_sum.reshape((1, z2_exp_sum.shape[0]))
    a2 = z2_exp / np.tile(z2_exp_sum, (c, 1)) # layer2 activation - softmax
    ll_matrix = -np.log(a2)
#     pred_matrix = (a2 == a2.max(axis=0, keepdims=1)).astype(float)
    A2 = z2.T # forward scores

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if Y is None:
      return A2

    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    # Compute the loss
    loss = np.sum(np.multiply(Y.T, ll_matrix))/m + 0.5*reg*(np.sum(W1**2)+np.sum(W2**2))

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # Backward pass: compute gradients
    dW2 = np.matmul((a2 - Y.T), a1_D1.T)/m + (reg*W2)
    db2 = np.matmul((a2 - Y.T), np.ones((m, 1)))/m
    da1 = np.matmul((a2.T - Y), W2)
    dRelu_mask = np.where(a1.T > 0, 1, 0)
    da1_relu = np.multiply(dRelu_mask, da1) # derivative - ReLU
    dW1 = np.matmul(da1_relu.T, X_D0)/m + (reg*W1)
    db1 = np.matmul(da1_relu.T, np.ones((m, 1)))/m

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    grads = {}
    grads["dW2"] = dW2
    grads["dW1"] = dW1
    grads["db2"] = db2
    grads["db1"] = db1

    return loss, grads

def train_2layer_net(params, X, Y, X_val, Y_val, optimizer="sgd", batch_size=200,
            dropout=0.0, learning_rate=1e-3, reg=5e-6, num_iters=100,
            learning_rate_decay=1.0, beta=0.9, beta1=0.9, beta2=0.999,
            epsilon=1e-8, verbose=False, seed=None):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - optimizer: descent technique "sgd", "momentum", "adam" or "rmsprop".
    - batch_size: Number of training examples to use per step.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_epochs: Number of epochs.
    - beta: Momentum hyperparameter
    - beta1: Exponential decay hyperparameter for the past gradients estimates
    - beta2: Exponential decay hyperparameter for the past squared gradients estimates
    - epsilon: hyperparameter preventing division by zero in Adam updates
    - verbose: boolean; if true print progress during optimization.
    """
    m = X.shape[0]
    iterations_per_epoch = max(m / batch_size, 1)

    # Keep track of merit values.
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    # Initializes velocity or weighted averages, according to optimizer.
    v, s = init_optimizer(params, optimizer)
    t = 0           # initializing the counter required for Adam update.

    for it in range(num_iters):

        # Random minibatch of training data and labels
        idx_batch = np.random.choice(np.arange(m),size=batch_size)
        X_batch = X[idx_batch]
        Y_batch = Y[idx_batch]

        # Compute loss and gradients using the current minibatch
        loss, grads = loss_2layer_net(params, X_batch, Y=Y_batch, reg=reg, dropout=dropout)
        loss_history.append(loss)

        # Update parameters
        if optimizer == "sgd":
            params = sgd(params, grads, learning_rate)
        elif optimizer == "momentum":
            params, v = sgd_momentum(params, grads, v, beta, learning_rate)
        elif optimizer == "rmsprop":
            params, s = rmsprop(params, grads, s, beta2, learning_rate, epsilon)
        elif optimizer == "adam":
            t = t + 1 # Adam counter
            params, v, s = adam(params, grads, v, s, t, learning_rate,
                                beta1, beta2, epsilon)

        if verbose and it % 100 == 0:
            print('iter %d / %d: loss %f' % (it, num_iters, loss))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            y_batch = np.argmax(Y_batch, axis=1)
            y_val = np.argmax(Y_val, axis=1)
            train_acc = (predict(params, X_batch) == y_batch).mean()
            val_acc = (predict(params, X_val) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
      'params': params,
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

def predict(params, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    m, n0 = X.shape      # m: number of training examples; n0: input dimension
    n1 = W1.shape[0]     # n1: input dimension to hidden layer
    c = W2.shape[0]
    ###########################################################################
    # TODO: Implement this function; it should be similar to a forward pass.  #
    ###########################################################################

    z1 = np.matmul(W1, X.T) + np.tile(b1, (1, m)) # layer1 linear sum
    z1_zero_stack = np.zeros((2, z1.shape[0], z1.shape[1]))
    z1_zero_stack[0] = z1
    z1_zero_stack[1] = np.zeros(z1.shape)
    a1 = np.max(z1_zero_stack, axis=0) # layer1 activation - ReLU
    z2 = np.matmul(W2, a1) + np.tile(b2, (1, m)) # layer2 linear sum
    z2_exp = np.exp(z2)
    z2_exp = z2_exp / z2_exp.max(axis=0) # normalize the exponents
    z2_exp_sum = np.sum(z2_exp, axis=0)
    z2_exp_sum = z2_exp_sum.reshape((1, z2_exp_sum.shape[0]))
    a2 = z2_exp / np.tile(z2_exp_sum, (c, 1)) # layer2 activation - softmax
    y_pred = np.argmax(a2, axis=0)

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred
