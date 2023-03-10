# Actor

from __future__ import print_function, division
import numpy as np
import math
from keras.initializers import identity
from keras.models import model_from_json, Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
import tensorflow 
import keras.backend as keras_backend

class Actor(object):
    """
    Object representing the actor network, which approximates the function:
        
        u(s) -> a
        
    where u(actually mew) is the deterministic policy mapping from states to
    actions a. 
    """
    def __init__(self, tensorflow_session, state_size, action_size,
                 hidden_units = (300, 600), learning_rate = 0.0001, batch_size = 64,
                 tau = 0.001):
        """
        Constructor for the Actor network
        
        :param tensorflow_session: the tensorflow session
        :param state_size: An integer denoting the dimensionality of the states in the current problem
        :param action_size: An integer denoting the dimensionality of the actions in the current problem
        :param hidden_units: An iterable defining the number of hidden units in each layer
        :param learning_rate: An flood denoting the speed at which the network will learn.
        :param batch_size: An integer denoting the batch size
        :param tau: A flot denoting the rare at which the target model will track the main model:
            
            target_weights= tau * main_weights + (1 - tau) * target_weights
        
        """
        # Store parameters
        self._tensorflow_session = tensorflow_session
        self._batch_size = batch_size
        self._tau = tau
        self._learning_rate = learning_rate
        self._hidden = hidden_units
        
        keras_backend.set_session(tensorflow_session)
        
        # Generate the main model
        self._model, self._model_weights, self._model_input = self._generate_model()
        self._target_model, self._target_weights, self._target_state = self._generate_model()
            
        # Generate tensors to hold the gradients for our Policy Gradient update
        self._action_gradients = tensorflow.placeholder(tensorflow.float32,
                                                        [None, action_size])
        self._parameter_gradients = tensorflow.gradients(self._model_output,
                                                         self._model_weights,
                                                         -self._action_gradients)
        self._gradients = zip(self._parameter_gradients, self._model_weights)
        
        # Define the optimization function
        self._optimize = tensorflow.train.AdamOptimizer(learning_rate).apply_gradients(self._gradients)
        
        # And initialise all tensorflow variables
        self._tensorflow_session.run(tensorflow.initialize_all_variables())
        
    def train(self, states, action_gradients):
        """
        Updates the weights of the main network
        
        :param states: The states of the input to the network
        :param action_gradients: The gradients of the actions to update the network
        :return: None
        """
        self._tensorflow_session.run(self._optimize, feed_dict = {self._states: states,
                                                                  self._action_gradients: action_gradients})
    
    def train_target_model(self):
        """
        Updates the weights of the target network to slowly track the main network.
        
        the speed at which target network tracks the main network is defined by tau,
        given in the constructor to this class. Formally, the tracking function 
        is defined as:
            
            target_weights = tau * main_weights + (1 - tau) * target_weights
        
        :return: None
        """
        main_weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()
        target_weights = [self._tau * main_weights + (1 - self._tau) *
                          target_weight for main_weight, target_weight in
                          zip(actor_weights, actor_target_weights)]
        self._target_model.set_weights(target_weights)
        
    def _generate_model(self):
        """
        Generates the model based on the hyperparameters defined in the constructor.
        
        :return: at tuple containing references to the model, weights, and input later
        """
        input_layer = Input(shape = [self._state_size])
        layer = Dense(self._hidden[0], activation= 'relu')(input_layer)
        layer = Dense(self._hidden[1], activation = 'relu')(layer)
        output_layer = Dense(self._action_size, activation='sigmoid')(layer)
        model = Model(input = input_layer, output = output_layer)
        return model, model.trainable_weights. input_layer