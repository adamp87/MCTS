"""
Implementation of a replica of proposed Deep Neural Network by DeepMind (class AlphaNet)
Class that inherits AlphaNet, retraining and inference with non-freezed models both on CPU and GPU (DNNPredict)

Author: AdamP 2020-2020
"""

import os

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, ReLU, add
from tensorflow.keras.optimizers import Adam


class AlphaNet:
    """
    Replica of the neural network proposed by DeepMind - AlphaZero
    """
    def __init__(self, input_dim, output_dim):

        self.num_layers = 5  # AlphaZero: 40
        self.hidden_layers = [{'filters': 128, 'kernel_size': (3, 3)} for _ in range(self.num_layers)]  # AlphaZero: 256

        self.input_dim = input_dim
        self.output_dim = output_dim[0]*output_dim[1]*output_dim[2]  # flattened
        self.reg_const = 0.0001
        self.learning_rate = 0.001

        self.model = self._build_model()

    def residual_layer(self, input_block, filters, kernel_size):

        x = self.conv_layer(input_block, filters, kernel_size)

        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=3)(x)
        x = add([input_block, x])
        x = ReLU()(x)

        return x

    def conv_layer(self, x, filters, kernel_size):

        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=3)(x)
        x = ReLU()(x)

        return x

    def value_head(self, x):

        x = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=3)(x)
        x = ReLU()(x)
        x = Flatten()(x)

        x = Dense(
            256,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = ReLU()(x)

        x = Dense(
            1,
            use_bias=False,
            activation='tanh',
            kernel_regularizer=regularizers.l2(self.reg_const),
            name='value_head'
        )(x)

        return x

    def policy_head(self, x):

        x = Conv2D(
            filters=2,
            kernel_size=(1, 1),
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=3)(x)
        x = ReLU()(x)
        x = Flatten()(x)

        x = Dense(
            self.output_dim,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(0.0001),
            name='policy_head'
        )(x)

        return x

    def _build_model(self):

        # input layer
        main_input = Input(shape=self.input_dim, name='main_input')

        # first convolution layer
        x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

        # hidden layers
        for h in self.hidden_layers[1:]:
            x = self.residual_layer(x, h['filters'], h['kernel_size'])

        # value and policy output layers
        vh = self.value_head(x)
        ph = self.policy_head(x)

        model = tf.keras.Model(inputs=[main_input], outputs=[vh, ph])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': tf.nn.softmax_cross_entropy_with_logits},
                      optimizer=Adam(learning_rate=self.learning_rate),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5}
                      )

        return model


class DNNPredict(AlphaNet):
    """
    Implements prediction and retraining of non-freezed models.
    """
    def __init__(self, log, input_dim, output_dim, **kwargs):
        AlphaNet.__init__(self, input_dim, output_dim)
        self.log = log

    def predict(self, state):
        """
        Prediction of value and policy based on an input state.
        Can be executed on non-freezed models both on CPU and GPU.
        :param state: Input tensor of a game state.
        :return: value: single value describing the chance to win
        :return: policy: tensor describing which next position should be investigated
        """
        value, policy = self.model.predict(state)
        return value[0, 0], policy

    def retrain(self, args, database):
        """Performs retraining of the model"""
        fit = None
        self.log.info("Retraining")
        for _ in tqdm(range(args.train_epochs)):  # select different data for each epoch
            state, policy, value = database.load(args.train_sample_size)
            policy.shape = (policy.shape[0], policy.shape[1] * policy.shape[2] * policy.shape[3])
            targets = {'value_head': value, 'policy_head': policy}

            fit = self.model.fit(state, targets, epochs=1, verbose=0, validation_split=0, batch_size=32)
            self.log.debug("Loss: {0}, Value: {1}, Policy: {2}".format(fit.history['loss'],
                                                                       fit.history['value_head_loss'],
                                                                       fit.history['policy_head_loss']))
        self.log.info("Final Loss: {0}, Value: {1}, Policy: {2}".format(fit.history['loss'],
                                                                        fit.history['value_head_loss'],
                                                                        fit.history['policy_head_loss']))

    def save(self, path):
        """Save model weight"""
        self.model.save_weights(os.path.join(path, 'weights', 'weights'))
        tf.saved_model.save(self.model, os.path.join(path, 'saved'))

    def load(self, path):
        """Load model weight"""
        self.model.load_weights(os.path.join(path, 'weights', 'weights'))
