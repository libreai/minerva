"""
Copyright 2017 Libre AI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function, division

import numpy as np
import keras.layers
from keras import optimizers
from keras.engine.topology import Layer
from keras.layers import Activation, Lambda
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Convolution1D, Dense
from keras.models import Input, Model

"""
Using Keras to implement the Nowcast Network (NNX) for timeseries prediction.
NNX is based on Convolutional Neural Networks following a WaveNet architecture.
"""


class NNX:
    def wave_net_activation(self, x):
        # type: (Layer) -> Layer
        """The WaveNet activation as described in https://deepmind.com/blog/wavenet-generative-model-raw-audio/

        Params:
            x: The layer we want to apply the activation to

        Returns:
            A new layer with the wavenet activation applied
        """
        tanh_out = Activation('tanh')(x)
        sigm_out = Activation('sigmoid')(x)
        return keras.layers.multiply([tanh_out, sigm_out])

    def residual_block(self, x, s, i, nb_filters, kernel_size, dropout_rate=0):
        """Defines the residual block for the WaveNet

        Params:
            x: The previous layer in the model
            s: The stack index i.e. which stack in the overall network
            dilation: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.

        Returns:
            A tuple where the first element is the residual model layer, and the second
            is the skip connection.
        """
        original_x = x
        conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                      dilation_rate=i, padding='causal',
                      name='dilated_conv_%d_tanh_s%d' % (i, s))(x)

        x = self.wave_net_activation(conv)

        x = SpatialDropout1D(dropout_rate, name='spatial_dropout1d_%d_s%d_%f' % (i, s, dropout_rate))(x)

        # 1x1 conv.
        x = Convolution1D(nb_filters, 1, padding='same')(x)
        # residual
        res_x = keras.layers.add([original_x, x])

        return res_x, x

    def process_dilations(self, dilations):
        def is_power_of_two(num):
            return num != 0 and ((num & (num - 1)) == 0)

        if all([is_power_of_two(i) for i in dilations]):
            return dilations

        else:
            new_dilations = tuple([2 ** i for i in dilations])
            print('Adjusting dilations from {} to {} (powers of two).'.format(dilations, new_dilations))
            return new_dilations

    def nnx(self,
            input_layer,
            nb_filters=64,
            kernel_size=2,
            nb_stacks=1,
            dilations=(1, 2, 4, 8, 16, 32),
            use_skip_connections=True,
            dropout_rate=0.0,
            return_sequences=True):
        """Creates a WaveNet layer.
        Params:
            input_layer: A tensor of shape (batch_size, timesteps, input_dim).
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.

        Returns:
            A Nowcasting Network Layer (NNX) layer.
        """
        x = input_layer
        x = Convolution1D(nb_filters, 1, padding='causal', name='initial_conv')(x)
        skip_connections = []
        for s in range(nb_stacks):
            for i in dilations:
                x, skip_out = self.residual_block(x, s, i, nb_filters, kernel_size, dropout_rate)
                skip_connections.append(skip_out)
        if use_skip_connections:
            x = keras.layers.add(skip_connections)
        x = Activation('relu')(x)

        if not return_sequences:
            output_slice_index = -1
            x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)
        return x

    def compile_nnx_model(self,
                          num_feat,  # type: int
                          num_classes,  # type: int
                          nb_filters,  # type: int
                          kernel_size,  # type: int
                          dilations,  # type: tuple[int]
                          nb_stacks,  # type: int
                          max_len,  # type: int
                          use_skip_connections=True,  # type: bool
                          return_sequences=True,
                          regression=True,  # type: bool
                          dropout_rate=0.05,  # type: float
                          nb_outputs=1
                          ):
        # type: (...) -> keras.Model
        """Creates a compiled WaveNet model for a given task (i.e. regression or classification).

        Parameters:
            :param num_feat: A tensor of shape (batch_size, timesteps, input_dim).
            :param num_classes: The size of the final dense layer, how many classes we are predicting.
            :param nb_filters: The number of filters to use in the convolutional layers.
            :param kernel_size: The size of the kernel to use in each convolutional layer.
            :param dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            :param nb_stacks : The number of stacks of residual blocks to use.
            :param max_len: The maximum sequence length, use None if the sequence length is dynamic.
            :param use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            :param return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            :param regression: Whether the output should be continuous or discrete.
            :param dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            :param nb_outputs: The output dimension, often equal to the number of inputs.

        Returns:
            :return A compiled Keras NNX model.
        """

        dilations = self.process_dilations(dilations)

        input_layer = Input(name='input_layer', shape=(max_len, num_feat))

        x = self.nnx(input_layer, nb_filters, kernel_size, nb_stacks, dilations,
                     use_skip_connections, dropout_rate, return_sequences)

        if regression:
            # regression
            preds = Dense(nb_outputs, activation='linear', name='output_dense')(x)

            model = Model(inputs=input_layer, outputs=preds)

            adam = optimizers.Adam(lr=0.002, clipnorm=1.)
            model.compile(adam, loss='mean_squared_error', metrics=['mae', 'mse'])
        else:
            # classification
            x = Dense(num_classes)(x)
            x = Activation('softmax', name='output_softmax')(x)
            output_layer = x
            model = Model(input_layer, output_layer)

            adam = optimizers.Adam(lr=0.002, clipnorm=1.)
            model.compile(adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def make_timeseries_instances(self, timeseries, window_size):
        """Make input features and prediction targets from a `timeseries` for use in machine learning.

        Parameters:
            :param ndarray timeseries: Either a simple vector, or a matrix of shape ``(timestep, series_num)``, i.e., time is axis 0 (the
              row) and the series is axis 1 (the column).
            :param int window_size: The number of samples to use as input prediction features (also called the lag or lookback).

        Returns:
            :return: A tuple of `(X, y, q)`.  `X` are the inputs to a predictor, a 3D ndarray with shape
              ``(timeseries.shape[0] - window_size, window_size, timeseries.shape[1] or 1)``.  For each row of `X`, the
              corresponding row of `y` is the next value in the timeseries.  The `q` or query is the last instance, what you would use
              to predict a hypothetical next (unprovided) value in the `timeseries`.
        """
        timeseries = np.asarray(timeseries)
        assert 0 < window_size < timeseries.shape[0]

        X = []
        y = []

        for start in range(0, timeseries.shape[0] - window_size):
            X.append(np.array(timeseries[start: start + window_size]))
            y.append(np.array(timeseries[start + window_size]))

        X = np.atleast_3d(X)
        y = np.asarray(y)

        print("X", X.shape)
        print("y", y.shape)

        q = np.atleast_3d([timeseries[-window_size:]])
        return X, y, q

    def nowcast(self, timeseries, window_size):
        nb_samples, nb_series = timeseries.shape

        timeseries = np.atleast_2d(timeseries)
        if timeseries.shape[0] == 1:
            timeseries = timeseries.T  # Convert 1D vectors to 2D column vectors

        X, y, q = self.make_timeseries_instances(timeseries, window_size)

        # epochs = int(np.sqrt(X.shape[0] * X.shape[1]))
        epochs = X.shape[0] * X.shape[1]
        print("epochs ", epochs)

        test_size = int(0.2 * nb_samples)
        x_train, x_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

        model = self.compile_nnx_model(num_feat=x_train.shape[2],
                                       num_classes=0,
                                       nb_filters=4,
                                       kernel_size=5,
                                       dilations=[1, 2, 4, 8],
                                       nb_stacks=8,
                                       max_len=x_train.shape[1],
                                       use_skip_connections=True,
                                       regression=True,
                                       dropout_rate=0.001,
                                       nb_outputs=nb_series,
                                       return_sequences=False
                                       )

        print(model.summary())

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=128, verbose=False)

        predictions = model.predict(q).squeeze()
        score = model.evaluate(x_test, y_test, verbose=0)
        print('(loss, mean_absolute_error, mean_squared_error) = ', score)
        return model, predictions


