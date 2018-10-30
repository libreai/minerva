"""
Using Keras to implement a Simple Nowcast Network (NNS) for timeseries prediction.
NNS is based on Convolutional Neural Networks following a simplified WaveNet architecture.
"""
from __future__ import print_function, division

import numpy as np
from keras.layers import Input, Convolution1D, Dense, Flatten
from keras.models import Model


class NNS:
    def compile_nns_model(self, window_size, kernel_size, nb_input_series=1, nb_outputs=1, filters=4):
        """:Return: a Keras Model for predicting the next value in a timeseries given a fixed-size lookback window of previous values.

        The model can handle multiple input timeseries (`nb_input_series`) and multiple prediction targets (`nb_outputs`).

        :param int window_size: The number of previous timeseries values to use as input features.  Also called lag or lookback.
        :param int nb_input_series: The number of input timeseries; 1 for a single timeseries.
          The `X` input to ``fit()`` should be an array of shape ``(n_instances, window_size, nb_input_series)``; each instance is
          a 2D array of shape ``(window_size, nb_input_series)``.  For example, for `window_size` = 3 and `nb_input_series` = 1 (a
          single timeseries), one instance could be ``[[0], [1], [2]]``. See ``make_timeseries_instances()``.
        :param int nb_outputs: The output dimension, often equal to the number of inputs.
          For each input instance (array with shape ``(window_size, nb_input_series)``), the output is a vector of size `nb_outputs`,
          usually the value(s) predicted to come after the last value in that input instance, i.e., the next value
          in the sequence. The `y` input to ``fit()`` should be an array of shape ``(n_instances, nb_outputs)``.
        :param int kernel_size: the size (along the `window_size` dimension) of the sliding window that gets convolved with
          each position along each instance. The difference between 1D and 2D convolution is that a 1D filter's "height" is fixed
          to the number of input timeseries (its "width" being `kernel_size`), and it can only slide along the window
          dimension.  This is useful as generally the input timeseries have no spatial/ordinal relationship, so it's not
          meaningful to look for patterns that are invariant with respect to subsets of the timeseries.
        :param int filters: The number of different filters to learn (roughly, input patterns to recognize).
        """

        # The first conv layer learns `filters` filters (aka kernels), each of size ``(kernel_size, nb_input_series)``.
        # Its output will have shape (None, window_size - kernel_size + 1, filters), i.e., for each position in
        # the input timeseries, the activation of each filter at that position.

        inputs = Input(shape=(window_size, nb_input_series))
        x = Convolution1D(filters, 1, padding='causal', name='initial_conv')(inputs)
        x = Convolution1D(activation='relu', filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=1)(x)
        x = Convolution1D(activation='relu', filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=2)(x)
        x = Convolution1D(activation='relu', filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=4)(x)
        x = Flatten()(x)
        preds = Dense(nb_outputs, activation='linear')(x)

        model = Model(inputs=inputs, outputs=preds)

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])

        return model


    def make_timeseries_instances(self, timeseries, window_size):
        """Make input features and prediction targets from a `timeseries` for use in machine learning.

        :return: A tuple of `(X, y, q)`.  `X` are the inputs to a predictor, a 3D ndarray with shape
          ``(timeseries.shape[0] - window_size, window_size, timeseries.shape[1] or 1)``.  For each row of `X`, the
          corresponding row of `y` is the next value in the timeseries.  The `q` or query is the last instance, what you would use
          to predict a hypothetical next (unprovided) value in the `timeseries`.
        :param ndarray timeseries: Either a simple vector, or a matrix of shape ``(timestep, series_num)``, i.e., time is axis 0 (the
          row) and the series is axis 1 (the column).
        :param int window_size: The number of samples to use as input prediction features (also called the lag or lookback).
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
        """Create a 1D CNN regressor to predict the next value in a `timeseries` using the preceding `window_size` elements
        as input features and evaluate its performance.

        :param ndarray timeseries: Timeseries data with time increasing down the rows (the leading dimension/axis).
        :param int window_size: The number of previous timeseries values to use to predict the next.
        """
        kernel_size = 5
        filters = 4
        timeseries = np.atleast_2d(timeseries)
        if timeseries.shape[0] == 1:
            timeseries = timeseries.T  # Convert 1D vectors to 2D column vectors

        nb_samples, nb_series = timeseries.shape
        model = self.compile_nns_model(window_size=window_size, kernel_size=kernel_size, nb_input_series=nb_series,
                                       nb_outputs=nb_series, filters=filters)

        X, y, q = self.make_timeseries_instances(timeseries, window_size)

        # epochs = int(np.sqrt(X.shape[0] * X.shape[1]))
        epochs = X.shape[0] * X.shape[1]
        print("epochs ", epochs)

        test_size = int(0.2 * nb_samples)
        x_train, x_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

        print(model.summary())

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=128, verbose=False)

        predictions = model.predict(q).squeeze()
        score = model.evaluate(x_test, y_test, verbose=0)
        print('(loss, mean_absolute_error, mean_squared_error) = ', score)
        return model, predictions
