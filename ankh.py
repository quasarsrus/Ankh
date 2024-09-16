# -*- coding: utf-8 -*-
"""
*ANKH(Custom Neural Network)*
"""

import numpy as np
import random
import warnings

"""#Multi-Layered Perceptron"""

class MLP(object):
  def __init__(self,prx,layer_dims,activation_type,loss_fcn, initialiser):
    if len(layer_dims) != 1 + len(activation_type):
      raise ValueError("Number of layers and activations do not match")

    if initialiser not in {"he_uniform", "xavier_normal", "glorot_uniform", "pre-existing", "normal"}:
      raise NameError("Incorrect initialiser")
    self.master_cache = []
    self.activation_type = activation_type
    self.loss_fcn = loss_fcn
    self.layer_dims = layer_dims
    self.weights = dict()
    self.bias = dict()
    j = 0

    for i in range(len(layer_dims)-1):

        if initialiser == "he_uniform":
          iw = np.sqrt(6/(layer_dims[i])).astype(np.float32)
          self.weights["W"+str(i+1)] = np.random.uniform(-iw, iw, size=(layer_dims[i+1],layer_dims[i])).astype(np.float32)
          self.bias["b"+str(i+1)] = np.zeros(layer_dims[i+1]).astype(np.float32)

        elif initialiser == "xavier_normal":
          iw = np.sqrt(2/((layer_dims[i+1] + layer_dims[i])))
          self.weights["W"+str(i+1)] = (np.random.normal(0, iw, size=(layer_dims[i+1],layer_dims[i])))
          self.bias["b"+str(i+1)] = (np.zeros(layer_dims[i+1]))

        elif initialiser == "glorot_uniform":
          iw = np.sqrt(6/((layer_dims[i+1] + layer_dims[i]))).astype(np.float32)
          self.weights["W"+str(i+1)] = (np.random.uniform(-iw, iw, size=(layer_dims[i+1],layer_dims[i]))).astype(np.float32)
          self.bias["b"+str(i+1)] = (np.zeros(layer_dims[i+1])).astype(np.float32)

        elif initialiser == "pre-existing":
          self.weights["W"+str(i+1)] = np.transpose(prx[j])
          self.bias["b"+str(i+1)] = np.transpose(prx[j+1])
          j += 2

        elif initialiser == "normal":
          self.weights["W"+str(i+1)] = np.random.normal(0,size=(layer_dims[i+1],layer_dims[i])) * np.sqrt(2/layer_dims[i])
          self.bias["b"+str(i+1)] = np.zeros(layer_dims[i+1])


  def forward_pass(self, input,layer):
    output = np.dot(input, np.transpose(self.weights["W"+str(layer)])) + \
              np.transpose(np.expand_dims(self.bias["b"+str(layer)], axis=-1).astype(np.float32))

    return output

  def activation(self, Z, name = "linear"):

    if name not in {"relu", "lrelu", "linear", "sigmoid", "tanh"}:
      raise NameError("Unavailable activation function")

    if name == "relu":
      A = np.maximum(np.zeros(np.shape(Z),dtype = np.float32),Z)
      return A
    elif name == "lrelu":
      A = np.maximum(np.zeros(np.shape(Z),dtype = np.float32) + 0.1,Z)
      return A
    elif name == "sigmoid":
      warnings.filterwarnings('ignore')
      A = 1./(1.+np.exp(-Z))
      return A
    elif name == "tanh":
      A = np.tanh(Z)
      return A
    elif name == "linear":
      return Z

  def forward_propagation(self, input):

    linear_cache = []
    activation_cache = []
    self.master_cache = []
    T = input
    for i in range(len(self.layer_dims)-1):
      activation_cache.append(T)
      Z = self.forward_pass(T,i+1)
      linear_cache.append(Z)
      T = self.activation(Z, self.activation_type[i])

    self.master_cache.append(activation_cache)
    self.master_cache.append(linear_cache)
    return T

  def compute_loss(self, Y, Yhat):
    if self.loss_fcn == "mse" or self.loss_fcn == "Mean_Squared_error":
      mse = np.mean((Y - Yhat) ** 2)
      return mse
    if self.loss_fcn == "binary_cross_entropy":
      epsilon = 1e-8
      bce = -(np.sum(Y * np.log(Yhat + epsilon) + (1 - Y) * np.log(1 - Yhat + epsilon))) / Y.shape[0]
      return bce

  def derivative_activation(self, Z, name = "linear"):

    if name not in {"relu", "lrelu", "linear", "sigmoid", "tanh"}:
      raise NameError("Unavailable activation function")

    if name == "relu":
      Z[Z<=0] = 0
      Z[Z>0] = 1
      A = Z
      return A
    elif name == "lrelu":
      Z[Z<=0.1] = 0.1
      Z[Z>0] = 1
      A = Z
      return A
    elif name == "sigmoid":
      P = self.activation(Z, name = "sigmoid")
      A = P*(1-P)
      return A
    elif name == "tanh":
      A = 1 - (self.activation(Z, name = "tanh"))**2
      return A
    elif name == "linear":
      return np.ones(np.shape(Z), dtype = np.float32)

  def output_loss_derivative(self, Yhat, Y):
    Yhat = Yhat.astype(np.float32)
    if self.loss_fcn == "mse" or self.loss_fcn == "Mean_Squared_error":
      m = np.shape(Y)[0]
      return -2*(Y-Yhat) / m

    if self.loss_fcn == "binary_cross_entropy":
      epsilon = 1e-8
      return (Yhat - Y)/(Yhat*(1 - Yhat) + epsilon)

  def backward_pass(self,delE,X,W):

    m = np.shape(delE)[0]
    delW = np.dot(np.transpose(X), delE) / m
    delb = np.sum(delE,0) / m
    delx = np.dot(delE,W)

    return delW, delb, delx

  def back_propagation(self, Yhat, Y):
    gradients = dict()
    delE = self.derivative_activation(self.master_cache[1][-1], name = self.activation_type[-1]) * self.output_loss_derivative(Yhat, Y)
    for i in range(len(self.layer_dims), 1, -1):
      gradients["delW"+str(i-1)], gradients["delb"+str(i-1)], gradients["delx"+str(i-1)] = \
                              self.backward_pass(delE, self.master_cache[0][i-2],self.weights["W"+str(i-1)])

      if i > 2:
        delE = self.derivative_activation(self.master_cache[1][i-3], name = self.activation_type[i-3]) * gradients["delx"+str(i-1)]

    return gradients

  def update_weights(self,grad_params, learning_rate, optimiser):

    if optimiser == "gradient_descent" or optimiser == "mini_batch_gradient_descent" or optimiser == "stochastic_gradient_descent":

      for i in range(len(self.layer_dims)-1):
        self.weights["W"+str(i+1)] -= learning_rate * np.transpose(grad_params["delW"+str(i+1)])
        self.bias["b"+str(i+1)] -= learning_rate * np.transpose(grad_params["delb"+str(i+1)])

    elif optimiser == "gradient_descent_with_momentum":

      for i in range(len(self.layer_dims)-1):
        self.weights["W"+str(i+1)] -= learning_rate * grad_params["v_dw"+str(i+1)]
        self.bias["b"+str(i+1)] -= learning_rate * grad_params["v_db"+str(i+1)]

    elif optimiser == "rmsprop":
      grad_params_rms, grad_params_bp = grad_params
      epsilon = 1e-7
      for i in range(len(self.layer_dims)-1):
        self.weights["W"+str(i+1)] -= learning_rate * np.transpose(grad_params_bp["delW"+str(i+1)]) / (np.sqrt(grad_params_rms["s_dw"+str(i+1)]) \
                                                                                                       + epsilon)
        self.bias["b"+str(i+1)] -= learning_rate * np.transpose(grad_params_bp["delb"+str(i+1)]) / (np.sqrt(grad_params_rms["s_db"+str(i+1)]) \
                                                                                                    + epsilon)

    elif optimiser == "Adam":
      epsilon = 1e-7
      for i in range(len(self.layer_dims)-1):

        self.weights["W"+str(i+1)] -= (learning_rate * grad_params["v_dw_corrected"+str(i+1)] / \
                                       (np.sqrt(grad_params["s_dw_corrected"+str(i+1)]) + epsilon))
        self.bias["b"+str(i+1)] -= (learning_rate * grad_params["v_db_corrected"+str(i+1)] /  \
                                       (np.sqrt(grad_params["s_db_corrected"+str(i+1)]) + epsilon))

    else:
      raise NameError("Incorrect/Unavailable Optimiser")

  def train_model(self,x_train, y_train, lr = 0.001, epochs = 1, verbose = True, optimiser = "gradient_descent", shuffle = True, \
                  steps_per_epoch = None, batch_size = None):

    total_losses = []

    if optimiser not in {"gradient_descent", "mini_batch_gradient_descent", "stochastic_gradient_descent", "gradient_descent_with_momentum",\
                         "rmsprop", "Adam"}:
      raise NameError("Incorrect/Unavailable Optimiser")

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    if batch_size == None and steps_per_epoch == None:
      batch_size = int(np.ceil(x_train.shape[0] / 2))
      steps_per_epoch = int(np.ceil(np.shape(x_train)[0]/batch_size))

    elif batch_size == None and steps_per_epoch != None:
      batch_size = int(np.ceil(np.shape(x_train)[0]/steps_per_epoch))

    elif batch_size != None and steps_per_epoch == None:
      steps_per_epoch = int(np.ceil(np.shape(x_train)[0]/batch_size))

    if optimiser == "gradient_descent":

      shuffled_index = [i for i in range(int(np.shape(x_train)[0]))]

      for i in range(epochs):

        if shuffle:
          random.shuffle(shuffled_index)

        yhat = self.forward_propagation(x_train[shuffled_index[:],:])
        loss = self.compute_loss(y_train[shuffled_index[:],:], yhat)
        grad = self.back_propagation(yhat, y_train[shuffled_index[:],:])

        self.update_weights(grad,lr,optimiser)

        total_losses.append(loss)

        if verbose:
          print("Iteration:",i+1,";", "Cost:",loss)

      return total_losses

    if optimiser == "mini_batch_gradient_descent":

      shuffled_index = [i for i in range(int(np.shape(x_train)[0]))]

      for i in range(epochs):

        marker = 0
        loss = 0
        batch = batch_size - 1
        if shuffle:
          random.shuffle(shuffled_index)

        for k in range(steps_per_epoch):

          yhat = self.forward_propagation(x_train[shuffled_index[marker:batch],:])
          loss += self.compute_loss(y_train[shuffled_index[marker:batch],:], yhat)
          loss = loss.astype(np.float32)
          grad = self.back_propagation(yhat, y_train[shuffled_index[marker:batch],:])
          self.update_weights(grad,lr,optimiser)

          if np.shape(x_train)[0] - batch < batch_size:
            marker += batch_size
            batch += np.shape(x_train)[0] - batch
            if marker >= np.shape(x_train)[0]:
              break
          else:
            marker += batch_size
            batch += batch_size

        total_losses.append(loss/steps_per_epoch)

        if verbose:
          print("Iteration:",i+1,";", "Cost:",loss/steps_per_epoch )

      return total_losses

    if optimiser == "stochastic_gradient_descent":

        shuffled_index = [i for i in range(int(np.shape(x_train)[0]))]

        for i in range(epochs):
          random.shuffle(shuffled_index)

          for j in range(len(shuffled_index)):
            yhat = self.forward_propagation(np.atleast_2d(x_train[shuffled_index[j],:]))
            loss = self.compute_loss(np.atleast_2d(y_train[shuffled_index[j],:]), yhat)
            grad = self.back_propagation(yhat, np.atleast_2d(y_train[shuffled_index[j],:]))
            self.update_weights(grad,lr,optimiser)
          if verbose:
            print("Iteration:",i+1,";", "Cost:",loss )

    if optimiser == "gradient_descent_with_momentum":
      beta = 0.9
      momentum = dict()
      shuffled_index = [i for i in range(int(np.shape(x_train)[0]))]

      for i in range(len(self.layer_dims)-1):
        momentum["v_dw"+str(i+1)] = np.zeros(np.shape(self.weights["W"+str(i+1)]),dtype = np.float32)
        momentum["v_db"+str(i+1)] = np.zeros(np.shape(self.bias["b"+str(i+1)]),dtype = np.float32)

      for i in range(epochs):

        marker = 0
        loss = 0
        batch = batch_size - 1
        if shuffle:
          random.shuffle(shuffled_index)

        for k in range(steps_per_epoch):

          yhat = self.forward_propagation(x_train[shuffled_index[marker:batch],:])
          loss += self.compute_loss(y_train[shuffled_index[marker:batch],:], yhat)
          loss = loss.astype(np.float32)
          grad = self.back_propagation(yhat, y_train[shuffled_index[marker:batch],:])

          for j in range(len(self.layer_dims)-1):
            momentum["v_dw"+str(j+1)] = beta * momentum["v_dw"+str(j+1)] + np.transpose((1 - beta) * grad["delW"+str(j+1)])
            momentum["v_db"+str(j+1)] = beta * momentum["v_db"+str(j+1)] + np.transpose((1 - beta) * grad["delb"+str(j+1)])

          self.update_weights(momentum,lr,optimiser)

          if np.shape(x_train)[0] - batch < batch_size:
              marker += batch_size
              batch += np.shape(x_train)[0] - batch
              if marker >= np.shape(x_train)[0]:
                break
          else:
            marker += batch_size
            batch += batch_size

        total_losses.append(loss/steps_per_epoch)

        if verbose:
          print("Epoch:",i+1,";", "Cost:",loss/steps_per_epoch )

      return total_losses

    if optimiser == "rmsprop":

      beta = 0.999
      rms = dict()
      shuffled_index = [i for i in range(int(np.shape(x_train)[0]))]

      for i in range(len(self.layer_dims)-1):
        rms["s_dw"+str(i+1)] = np.zeros(np.shape(self.weights["W"+str(i+1)]),dtype = np.float32)
        rms["s_db"+str(i+1)] = np.zeros(np.shape(self.bias["b"+str(i+1)]),dtype = np.float32)

      for i in range(epochs):

        marker = 0
        loss = 0
        batch = batch_size - 1
        if shuffle:
          random.shuffle(shuffled_index)

        for k in range(steps_per_epoch):

          yhat = self.forward_propagation(x_train[shuffled_index[marker:batch],:])
          loss += self.compute_loss(y_train[shuffled_index[marker:batch],:], yhat)
          loss = loss.astype(np.float32)
          grad = self.back_propagation(yhat, y_train[shuffled_index[marker:batch],:])

          for j in range(len(self.layer_dims)-1):
            rms["s_dw"+str(j+1)] = beta * rms["s_dw"+str(j+1)] + np.transpose((1 - beta) * (grad["delW"+str(j+1)]) ** 2)
            rms["s_db"+str(j+1)] = beta * rms["s_db"+str(j+1)] + np.transpose((1 - beta) * (grad["delb"+str(j+1)]) ** 2)

          self.update_weights((rms,grad),lr,optimiser)

          if np.shape(x_train)[0] - batch < batch_size:
            marker += batch_size
            batch += np.shape(x_train)[0] - batch
            if marker >= np.shape(x_train)[0]:
              break
          else:
            marker += batch_size
            batch += batch_size

        total_losses.append(loss/steps_per_epoch)

        if verbose:
          print("Epoch:",i+1,";", "Cost:",loss/steps_per_epoch)

      return total_losses

    if optimiser == "Adam":

      beta2 = 0.999
      beta1 = 0.9
      adam_opt = dict()
      shuffled_index = [i for i in range(int(np.shape(x_train)[0]))]

      for i in range(len(self.layer_dims)-1):
        adam_opt["v_db"+str(i+1)] = np.zeros(np.shape(self.bias["b"+str(i+1)]), dtype = np.float32)
        adam_opt["v_dw"+str(i+1)] = np.zeros(np.shape(self.weights["W"+str(i+1)]),dtype = np.float32)
        adam_opt["s_dw"+str(i+1)] = np.zeros(np.shape(self.weights["W"+str(i+1)]),dtype = np.float32)
        adam_opt["s_db"+str(i+1)] = np.zeros(np.shape(self.bias["b"+str(i+1)]),dtype = np.float32)

      for j in range(epochs):
        marker = 0
        loss = 0
        batch = batch_size - 1
        if shuffle:
          random.shuffle(shuffled_index)

        for k in range(steps_per_epoch):

          yhat = self.forward_propagation(x_train[shuffled_index[marker:batch],:])
          loss += self.compute_loss(y_train[shuffled_index[marker:batch],:], yhat)
          loss = loss.astype(np.float32)
          grad = self.back_propagation(yhat, y_train[shuffled_index[marker:batch],:])

          for i in range(len(self.layer_dims)-1):

            adam_opt["v_dw"+str(i+1)] = beta1 * adam_opt["v_dw"+str(i+1)] + np.transpose((1 - beta1) * grad["delW"+str(i+1)])
            adam_opt["v_db"+str(i+1)] = beta1 * adam_opt["v_db"+str(i+1)] + np.transpose((1 - beta1) * grad["delb"+str(i+1)])
            adam_opt["s_dw"+str(i+1)] = beta2 * adam_opt["s_dw"+str(i+1)] + np.transpose((1 - beta2) * (grad["delW"+str(i+1)] ) ** 2)
            adam_opt["s_db"+str(i+1)] = beta2 * adam_opt["s_db"+str(i+1)] + np.transpose((1 - beta2) * (grad["delb"+str(i+1)] ) ** 2)

            adam_opt["v_dw_corrected"+str(i+1)] = adam_opt["v_dw"+str(i+1)] / (1 - beta1 ** (j+1))
            adam_opt["v_db_corrected"+str(i+1)] = adam_opt["v_db"+str(i+1)] / (1 - beta1 ** (j+1))
            adam_opt["s_dw_corrected"+str(i+1)] = adam_opt["s_dw"+str(i+1)] / (1 - beta2 ** (j+1))
            adam_opt["s_db_corrected"+str(i+1)] = adam_opt["s_db"+str(i+1)] / (1 - beta2 ** (j+1))

          self.update_weights(adam_opt,lr,optimiser)
          if np.shape(x_train)[0] - batch < batch_size:
            marker += batch_size
            batch += np.shape(x_train)[0] - batch
            if marker >= np.shape(x_train)[0]:
              break
          else:
            marker += batch_size
            batch += batch_size

        total_losses.append(loss/steps_per_epoch)

        if verbose:
          print("Epoch:",j+1,";", "Cost:",loss/steps_per_epoch )

      return total_losses
