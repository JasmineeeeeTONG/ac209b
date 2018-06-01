import numpy as np
import math

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts a dictionary params and
the gradient of the loss with respect to those elements and produces the next
set of weights in the same dictionary params.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates.
"""

def init_optimizer(params, optimizer):
    """
    Initializes velocity, exponential weighted average of the gradient, or the
    exponentially weighted average of the squared gradient, according to the
    optimizer. Returns None if the parameter is not used.
    """
    v, s = {}, {}
    if optimizer == "sgd":
        pass       # no inizialization necessary
    elif optimizer == "momentum":
        v["dW1"] = np.zeros_like(params["W1"])
        v["db1"] = np.zeros_like(params["b1"])
        v["dW2"] = np.zeros_like(params["W2"])
        v["db2"] = np.zeros_like(params["b2"])
    elif optimizer == "rmsprop":
        s["dW1"] = np.zeros_like(params["W1"])
        s["db1"] = np.zeros_like(params["b1"])
        s["dW2"] = np.zeros_like(params["W2"])
        s["db2"] = np.zeros_like(params["b2"])
    elif optimizer == "adam":
        v["dW1"] = np.zeros_like(params["W1"])
        v["db1"] = np.zeros_like(params["b1"])
        v["dW2"] = np.zeros_like(params["W2"])
        v["db2"] = np.zeros_like(params["b2"])
        s["dW1"] = np.zeros_like(params["W1"])
        s["db1"] = np.zeros_like(params["b1"])
        s["dW2"] = np.zeros_like(params["W2"])
        s["db2"] = np.zeros_like(params["b2"])
    return v, s

def sgd(params, grads, learning_rate=1e-3):
    """
    Performs vanilla stochastic gradient descent.
    - learning_rate: Scalar learning rate.
    """
    ###########################################################################
    # TODO: Implement the stochastic gradient descent update rule.            #
    ###########################################################################
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2']
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    params['W1'] = W1
    params['b1'] = b1
    params['W2'] = W2
    params['b2'] = b2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return params


def sgd_momentum(params, grads, v, beta=0.9, learning_rate=1e-3):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - v: python dictionary containing current velocity: v["dW1"], v["db1"], etc.
    - beta: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    """
    ###########################################################################
    # TODO: Implement the momentum update formula. You should use and update  #
    # the velocity v.                                                         #
    ###########################################################################
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2']
    V_dW1, V_db1, V_dW2, V_db2 = v['dW1'], v['db1'], v['dW2'], v['db2']
    V_dW1 = beta * V_dW1 + (1-beta) * dW1
    V_db1 = beta * V_db1 + (1-beta) * db1
    V_dW2 = beta * V_dW2 + (1-beta) * dW2
    V_db2 = beta * V_db2 + (1-beta) * db2
    W1 -= learning_rate * V_dW1
    b1 -= learning_rate * V_db1
    W2 -= learning_rate * V_dW2
    b2 -= learning_rate * V_db2
    v['dW1'] = V_dW1
    v['db1'] = V_db1
    v['dW2'] = V_dW2
    v['db2'] = V_db2
    params['W1'] = W1
    params['b1'] = b1
    params['W2'] = W2
    params['b2'] = b2

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return params, v


def rmsprop(params, grads, s, beta2=0.99, learning_rate=1e-3, epsilon=1e-8):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: scalar learning rate.
    - beta2: scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    """
    ###########################################################################
    # TODO: Implement the RMSprop update formula.                                                #
    ###########################################################################
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2']
    s_dW1, s_db1, s_dW2, s_db2 = s['dW1'], s['db1'], s['dW2'], s['db2']
    s_dW1 = beta2 * s_dW1 + (1-beta2) * (dW1**2)
    s_db1 = beta2 * s_db1 + (1-beta2) * (db1**2)
    s_dW2 = beta2 * s_dW2 + (1-beta2) * (dW2**2)
    s_db2 = beta2 * s_db2 + (1-beta2) * (db2**2)
    W1 -= learning_rate * dW1 / (np.sqrt(s_dW1)+epsilon)
    b1 -= learning_rate * db1 / (np.sqrt(s_db1)+epsilon)
    W2 -= learning_rate * dW2 / (np.sqrt(s_dW2)+epsilon)
    b2 -= learning_rate * db2 / (np.sqrt(s_db2)+epsilon)
    s['dW1'] = s_dW1
    s['db1'] = s_db1
    s['dW2'] = s_dW2
    s['db2'] = s_db2
    params['W1'] = W1
    params['b1'] = b1
    params['W2'] = W2
    params['b2'] = b2

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return params, s


def adam(params, grads, v, s, t, learning_rate=0.01, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - v: Adam variable, moving average of the first gradient, python dictionary
    - s: Adam variable, moving average of the squared gradient, python dictionary
    - t: Iteration number.
    """
    vc = {}        # Initializing first moment estimate, v corrected
    sc = {}        # Initializing second moment estimate, s corrected
    ###########################################################################
    # TODO: Implement the Adam update formula for all elements in params.     #
    ###########################################################################
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2']

    V_dW1, V_db1, V_dW2, V_db2 = v['dW1'], v['db1'], v['dW2'], v['db2']
    V_dW1 = beta1 * V_dW1 + (1-beta1) * dW1
    V_db1 = beta1 * V_db1 + (1-beta1) * db1
    V_dW2 = beta1 * V_dW2 + (1-beta1) * dW2
    V_db2 = beta1 * V_db2 + (1-beta1) * db2
    v['dW1'] = V_dW1
    v['db1'] = V_db1
    v['dW2'] = V_dW2
    v['db2'] = V_db2
    vc_dW1 = V_dW1 / (1-beta1)
    vc_db1 = V_db1 / (1-beta1)
    vc_dW2 = V_dW2 / (1-beta1)
    vc_db2 = V_db2 / (1-beta1)


    s_dW1, s_db1, s_dW2, s_db2 = s['dW1'], s['db1'], s['dW2'], s['db2']
    s_dW1 = beta2 * s_dW1 + (1-beta2) * (dW1**2)
    s_db1 = beta2 * s_db1 + (1-beta2) * (db1**2)
    s_dW2 = beta2 * s_dW2 + (1-beta2) * (dW2**2)
    s_db2 = beta2 * s_db2 + (1-beta2) * (db2**2)
    s['dW1'] = s_dW1
    s['db1'] = s_db1
    s['dW2'] = s_dW2
    s['db2'] = s_db2
    sc_dW1 = s_dW1 / (1-beta2)
    sc_db1 = s_db1 / (1-beta2)
    sc_dW2 = s_dW2 / (1-beta2)
    sc_db2 = s_db2 / (1-beta2)

    W1 -= learning_rate * vc_dW1 / (np.sqrt(sc_dW1)+epsilon)
    b1 -= learning_rate * vc_db1 / (np.sqrt(sc_db1)+epsilon)
    W2 -= learning_rate * vc_dW2 / (np.sqrt(sc_dW2)+epsilon)
    b2 -= learning_rate * vc_db2 / (np.sqrt(sc_db2)+epsilon)
    params['W1'] = W1
    params['b1'] = b1
    params['W2'] = W2
    params['b2'] = b2

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return params, v, s


def random_mini_batches(X, Y, batch_size = 64, seed = 0):
    """
    DEPRECATED: uses to much memory.
    Creates a list of random minibatches from (X, Y)

    Arguments:
    - X: input data, of shape (m, n0)
    - Y: true "label" vector (1 for blue dot / 0 for red dot), of shape (m, c)
    - batch_size: size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)            # To control "random" minibatches
    m = X.shape[0]                  # number of training examples
    c = Y.shape[1]                  # number of classes
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, c))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/batch_size)
    for k in range(num_complete_minibatches):
        batch_X = shuffled_X[k * batch_size:(k + 1) * batch_size, :]
        batch_Y = shuffled_Y[k * batch_size:(k + 1) * batch_size, :]
        mini_batch = (batch_X, batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % batch_size != 0:
        end = m - batch_size * math.floor(m / batch_size)
        batch_X = shuffled_X[num_complete_minibatches * batch_size:, :]
        batch_Y = shuffled_Y[num_complete_minibatches * batch_size:, :]
        mini_batch = (batch_X, batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
