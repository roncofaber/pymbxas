#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:50:12 2025

@author: roncofaber
"""

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

class StackedAutoencoder:
    def __init__(self, input_dim, hidden_dims, learning_rate=0.0001):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.encoder = None
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        # Define the encoder layers
        input_layer = Input(shape=(self.input_dim,))
        encoded = input_layer
        for hidden_dim in self.hidden_dims:
            encoded = Dense(hidden_dim, activation='relu')(encoded)

        # Define the decoder layers
        decoded = encoded
        for hidden_dim in reversed(self.hidden_dims[:-1]):
            decoded = Dense(hidden_dim, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)

        # Compile the autoencoder
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='mse')

        # Create the encoder model
        self.encoder = Model(input_layer, encoded)
        return autoencoder

    def train(self, X, epochs=512, batch_size=64):
        history = self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)


    def encode(self, X):
        return self.encoder.predict(X)
    
    def predict(self, X):
        return self.encoder.predict(X)
