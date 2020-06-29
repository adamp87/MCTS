import os
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, ReLU, LeakyReLU, add
from tensorflow.keras.optimizers import Adam


class ResidualCNN:
    """
    Replica of the neural network proposed by DeepMind - AlphaZero
    """
    def __init__(self, input_dim, output_dim):

        self.num_layers = 20  # AlphaZero: 40
        self.hidden_layers = [{'filters': 128, 'kernel_size': (3, 3)} for i in range(self.num_layers)]  # AlphaZero: 256

        self.input_dim = input_dim
        self.output_dim = output_dim[0]*output_dim[1]*output_dim[2]  # flattened
        self.reg_const = 0.0001
        self.learning_rate = 0.1

        self.model = self._build_model()

    def residual_layer(self, input_block, filters, kernel_size):

        x = self.conv_layer(input_block, filters, kernel_size)

        x = Conv2D(
            filters=filters
            , kernel_size=kernel_size
            , data_format="channels_first"
            , padding='same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)

        x = add([input_block, x])

        x = ReLU()(x)

        return (x)

    def conv_layer(self, x, filters, kernel_size):

        x = Conv2D(
            filters=filters
            , kernel_size=kernel_size
            , data_format="channels_first"
            , padding='same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = ReLU()(x)

        return (x)

    def value_head(self, x):

        x = Conv2D(
            filters=1
            , kernel_size=(1, 1)
            , data_format="channels_first"
            , padding='same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(
            256
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(0.1)
        )(x)

        x = LeakyReLU()(x)

        x = Dense(
            1
            , use_bias=False
            , activation='tanh'
            , kernel_regularizer=regularizers.l2(0.01)
            , name='value_head'
        )(x)

        return (x)

    def policy_head(self, x):

        x = Conv2D(
            filters=2
            , kernel_size=(1, 1)
            , data_format="channels_first"
            , padding='same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = ReLU()(x)

        x = Flatten()(x)

        x = Dense(
            self.output_dim
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(0.0001)
            , name='policy_head'
        )(x)

        return (x)

    def _build_model(self):

        main_input = Input(shape=self.input_dim, name='main_input')

        x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self.residual_layer(x, h['filters'], h['kernel_size'])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        model = Model(inputs=[main_input], outputs=[vh, ph])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': tf.nn.softmax_cross_entropy_with_logits},
                      optimizer=Adam(learning_rate=self.learning_rate),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5}
                      )

        return model


class DNNPredict(ResidualCNN):
    def __init__(self, input_dim, output_dim):
        ResidualCNN.__init__(self, input_dim, output_dim)

    def predict(self, input):
        value, policy = self.model.predict(input)
        return value[0, 0], policy

    def retrain(self, log, args, database):
        log.info("Retraining")
        n_states = database.get_state_count()
        for i in tqdm(range(args.train_epochs)):  # select different data for each epoch
            idx = np.random.choice(np.arange(0, n_states, 1), np.min((args.train_sample_size, n_states)), replace=False)
            idx = np.sort(idx)  # hdf5 requires sorted index
            state = database.datafile["state"][idx, :, :, :]
            policy = database.datafile["policy"][idx, :, :, :]
            value = database.datafile["value"][idx, :]
            policy.shape = (policy.shape[0], policy.shape[1] * policy.shape[2] * policy.shape[3])
            targets = {'value_head': value, 'policy_head': policy}

            fit = self.model.fit(state, targets, epochs=1, verbose=0, validation_split=0, batch_size=64)  # 32
            log.debug("Loss: {0}, Value: {1}, Policy: {2}".format(fit.history['loss'],
                                                                  fit.history['value_head_loss'],
                                                                  fit.history['policy_head_loss']))
        log.info("Final Loss: {0}, Value: {1}, Policy: {2}".format(fit.history['loss'],
                                                                   fit.history['value_head_loss'],
                                                                   fit.history['policy_head_loss']))

    def save(self, path):
        self.model.save_weights(os.path.join(path, 'weights', 'weights'))
        tf.saved_model.save(self.model, os.path.join(path, 'saved'))

    def load(self, path):
        self.model.load_weights(os.path.join(path, 'weights', 'weights'))
