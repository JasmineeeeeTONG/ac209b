import keras
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time


#--------  plot_decision_boundary
# A function that visualizes the decision boundaries of a given model
# Input: 
#      x (predictors)
#      y (labels)
#      model (classification model)
#      title (title for plot)
#      ax (axis to plot on)

def plot_decision_boundary(x, y, model, title, ax):
    # Create mesh
    # Interval of points for biomarker 1
    min0 = np.min(x[:,0])
    max0 = np.max(x[:,0])
    interval0 = np.arange(min0, max0, (max0-min0)/100.0)
    n0 = np.size(interval0)
    
    # Interval of points for biomarker 2
    min1 = np.min(x[:,1])
    max1 = np.max(x[:,1])
    interval1 = np.arange(min1, max1, (max1-min1)/100.0)
    n1 = np.size(interval1)

    # Create mesh grid of points
    x1, x2 = np.meshgrid(interval0, interval1)
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    xx = np.concatenate((x1, x2), axis=1)

    # Predict on mesh of points
    ybar = 1.0*(model.predict(xx)>0.5)
    yy = ybar.reshape((n0, n1))
    
    # Plot decision surface
    x1 = x1.reshape(n0, n1)
    x2 = x2.reshape(n0, n1)

    #if(y != None):
    if(y.any() != None):
        # Plot scatter plot of data
        ybar = yy.reshape(-1,)
        ax.scatter(xx[ybar==0,0], xx[ybar==0,1], c='red', cmap=plt.cm.coolwarm, alpha = 0.1)
        ax.scatter(xx[ybar==1,0], xx[ybar==1,1], c='blue', cmap=plt.cm.coolwarm, alpha = 0.1)
    
    # Label axis, title
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')



#--------  plot_data
# A function that visualizes data
# Input: 
#      x (predictors)
#      y (labels)

def plot_data(x, y, ax, title):
    xpos = x[y==1,:]
    xneg = x[y==0,:]
    
    ax.plot(xneg[:,0], xneg[:,1], 'r.', label = 'negative', markersize=1)
    ax.plot(xpos[:,0], xpos[:,1], 'b.', label = 'positive', markersize=1)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.legend(loc = 'upper right')


#--------  plot_learning_curve
# A function that visualizes the training and test accuracies as function of epochs
# Input: 
#      results
#      ax

def plot_learning_curve(results, ax):
    ax.plot(results.history['acc'], label='Train')
    ax.plot(results.history['val_acc'], label='Test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='best')

    
# Logging class
# Call back once in few batches

class NBatchLogger(Callback):
    def __init__(self, display, x, y, model):
        self.seen = 0
        self.display = display
        self.x = x
        self.y = y
        self.model = model

    def on_epoch_end(self, batch, logs={}):
        self.seen += 1
        if self.seen % self.display == 0:
            # you can access loss, accuracy in self.params['metrics']
            print('{}: {:.3f} / {:.3f}'.format(\
                            self.seen,\
                            logs['acc'],\
                            logs['val_acc'])) 
            
 