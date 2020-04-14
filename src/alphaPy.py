import os
import logging
import threading
import subprocess
from argparse import ArgumentParser

import zmq
import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt

log = logging.getLogger('')
log.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s --%(levelname)s-- %(message)s')
# set up logging to file
log_file = logging.FileHandler('/home/adamp/Documents/Codes/Hearts/data/chess_1.log')
log_file.setFormatter(log_formatter)
log_file.setLevel(logging.DEBUG)
log.addHandler(log_file)
# define a Handler which writes INFO messages or higher to console
log_console = logging.StreamHandler()
log_console.setFormatter(log_formatter)
log_console.setLevel(logging.INFO)
log.addHandler(log_console)


def plot_state(state, policy):
    t = 0
    fig, ax = plt.subplots(1, 2, figsize=(9, 6))
    img = np.zeros((8, 8))
    for i in range(6):
        img += state[6 * t + i, :, :] * (i + 1)
        img -= state[6 * 8 + 6 * t + i, :, :] * (i + 1)
    im = ax[0].imshow(img)
    fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(policy)
    fig.colorbar(im, ax=ax[1])
    plt.tight_layout(0.1)
    plt.show()


def debug_state(state, policy):
    fig, ax = plt.subplots(3, 3)
    ax = ax.flatten()
    for t in range(8):
        img = np.zeros((8, 8))
        for i in range(6):
            img += state[6 * t + i, :, :] * (i + 1)
            img -= state[6 * 8 + 6 * t + i, :, :] * (i + 1)
        im = ax[t].imshow(img)
        fig.colorbar(im, ax=ax[t])
    im = ax[8].imshow(policy)
    fig.colorbar(im, ax=ax[8])
    plt.show()

    for t in range(8):
        rep = state[12 * 8 + t, :, :]
        print("REP1 t:{0}, min:{1:4.2f}, max:{1:4.2f}".format(t, np.min(rep), np.max(rep)))
        rep = state[12 * 8 + 8 + t, :, :]
        print("REP2 t:{0}, min:{1:4.2f}, max:{1:4.2f}".format(t, np.min(rep), np.max(rep)))

    data = state[14 * 8 + 0, :, :]
    print("COLR min:{1:4.2f}, max:{1:4.2f}".format(np.min(data), np.max(data)))
    data = state[14 * 8 + 1, :, :]
    print("MOVN min:{1:4.2f}, max:{1:4.2f}".format(np.min(data), np.max(data)))
    data = state[14 * 8 + 2, :, :]
    print("P1CL min:{1:4.2f}, max:{1:4.2f}".format(np.min(data), np.max(data)))
    data = state[14 * 8 + 3, :, :]
    print("P1CR min:{1:4.2f}, max:{1:4.2f}".format(np.min(data), np.max(data)))
    data = state[14 * 8 + 4, :, :]
    print("P2CL min:{1:4.2f}, max:{1:4.2f}".format(np.min(data), np.max(data)))
    data = state[14 * 8 + 5, :, :]
    print("P2CR min:{1:4.2f}, max:{1:4.2f}".format(np.min(data), np.max(data)))
    data = state[14 * 8 + 6, :, :]
    print("NOAC min:{1:4.2f}, max:{1:4.2f}".format(np.min(data), np.max(data)))


class DNNStatePolicyHandler(threading.Thread):
    def __init__(self, zmq_context, datafilepath_hdf, dims, port="5557"):
        threading.Thread.__init__(self)
        self.dims = dims
        self.socket = zmq_context.socket(zmq.REP)
        self.socket.bind("tcp://*:{0}".format(port))

        self.data_state = []
        self.data_policy = []

        if not h5py.is_hdf5(datafilepath_hdf):
            datafile = h5py.File(datafilepath_hdf, "w")
            datafile.create_dataset("state", dtype=np.float32,
                                    shape=(0, dims[0], dims[1], dims[2]),
                                    maxshape=(None, dims[0], dims[1], dims[2]),
                                    chunks=(42, dims[0], dims[1], dims[2]),
                                    compression="gzip", compression_opts=1)
            datafile.create_dataset("policy", dtype=np.float32,
                                    shape=(0, dims[1], dims[2]),
                                    maxshape=(None, dims[1], dims[2]),
                                    chunks=(42 * dims[0], dims[1], dims[2]),
                                    compression="gzip", compression_opts=1)
            datafile.create_dataset("value", dtype=np.float32,
                                    shape=(0, 1),
                                    maxshape=(None, 1),
                                    chunks=(1024 * 1024, 1),
                                    compression="gzip", compression_opts=1)
            datafile.create_dataset("game_idx", dtype=np.uint32,
                                    shape=(0, 1),
                                    maxshape=(None, 1),
                                    chunks=(1024 * 1024, 1),
                                    compression="gzip", compression_opts=1)
            datafile.close()
            log.info("Database have been created at {0}".format(datafilepath_hdf))
        self.datafile = h5py.File(datafilepath_hdf, "a")

    def store_and_reset(self, game_idx, game_result):
        dset = self.datafile["state"]
        dset_idx = dset.shape[0]
        dset.resize(dset_idx+len(self.data_state), axis=0)
        dset[dset_idx:, :, :, :] = self.data_state

        dset = self.datafile["policy"]
        dset_idx = dset.shape[0]
        dset.resize(dset_idx+len(self.data_state), axis=0)
        dset[dset_idx:, :, :] = self.data_policy

        data_z = np.empty(len(self.data_state), dtype=np.float32)
        data_z[::2] = game_result[0]
        data_z[1::2] = game_result[1]

        dset = self.datafile["value"]
        dset_idx = dset.shape[0]
        dset.resize(dset_idx+len(self.data_state), axis=0)
        dset[dset_idx:, 0] = data_z

        data_game_idx = np.zeros(len(self.data_state), dtype=np.uint32) + game_idx
        dset = self.datafile["game_idx"]
        dset_idx = dset.shape[0]
        dset.resize(dset_idx+len(self.data_state), axis=0)
        dset[dset_idx:, 0] = data_game_idx

        self.datafile.flush()
        self.data_state.clear()
        self.data_policy.clear()

    def get_game_count(self):
        dset = self.datafile["game_idx"]
        idx = dset.shape[0]
        if idx == 0:
            return 0
        return int(dset[idx-1])

    def get_state_count(self):
        return self.datafile["state"].shape[0]

    def run(self):
        def send_ok(socket):
            #  Send reply back to client
            dummy = np.ones(2, dtype=np.int8)
            dummy[0] = 4
            dummy[1] = 2
            data = np.array(dummy).tobytes()
            socket.send(data)

        while True:
            try:
                #  receive state
                message = self.socket.recv()
                state = np.frombuffer(message, dtype=np.float32)
                state.shape = (self.dims[0], self.dims[1], self.dims[2])
                send_ok(self.socket)

                # receive policy
                message = self.socket.recv()
                policy = np.frombuffer(message, dtype=np.float32)
                policy.shape = (self.dims[1], self.dims[2])
                send_ok(self.socket)

                self.data_state.append(state)
                self.data_policy.append(policy)
            except zmq.error.ContextTerminated:
                self.socket.close()
                return


class DNNPredict(threading.Thread):
    def __init__(self, zmq_context, dims, port="5555"):
        threading.Thread.__init__(self)
        self.dims = dims
        self.socket = zmq_context.socket(zmq.REP)
        self.socket.bind("tcp://*:{0}".format(port))
        self.model = ResidualCNN((self.dims[0], self.dims[1], self.dims[2]), (self.dims[1], self.dims[2]))
        self.port = port

    def save_weight(self, path):
        self.model.model.save_weights(path)

    def load_weight(self, path):
        self.model.model.load_weights(path)

    def run(self):
        while True:
            try:
                # get state
                message = self.socket.recv()
                state = np.frombuffer(message, dtype=np.float32)
                state.shape = (1, self.dims[0], self.dims[1], self.dims[2])

                # predict
                value, policy = self.model.model.predict(state)

                # send prediction
                data = np.empty((1, self.dims[1]*self.dims[2]+1), dtype=np.float32)
                data[0, :self.dims[1]*self.dims[2]] = policy
                data[0, self.dims[1]*self.dims[2]] = value
                data = np.array(data).tobytes()
                self.socket.send(data)

            except zmq.error.ContextTerminated:
                self.socket.close()
                return


class ResidualCNN:
    def __init__(self, input_dim, output_dim):

        self.hidden_layers = [
            {'filters': 75, 'kernel_size': (4, 4)}
            , {'filters': 75, 'kernel_size': (4, 4)}
            , {'filters': 75, 'kernel_size': (4, 4)}
            , {'filters': 75, 'kernel_size': (4, 4)}
            , {'filters': 75, 'kernel_size': (4, 4)}
            , {'filters': 75, 'kernel_size': (4, 4)}
        ]
        self.num_layers = len(self.hidden_layers)

        self.input_dim = input_dim
        self.output_dim = output_dim[0]*output_dim[1]  # TODO
        self.reg_const = 0.0001
        self.learning_rate = 0.1
        self.value_dense_node_count = 20

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

        x = LeakyReLU()(x)

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
        x = LeakyReLU()(x)

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
            self.value_dense_node_count
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = LeakyReLU()(x)

        x = Dense(
            1
            , use_bias=False
            , activation='tanh'
            , kernel_regularizer=regularizers.l2(self.reg_const)
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
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(
            self.output_dim
            , use_bias=False
            , activation='linear'
            , kernel_regularizer=regularizers.l2(self.reg_const)
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


def execute_game(cmd_args):
    last_line = ""
    process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while process.poll() is None:
        while True:
            output = process.stdout.readline().decode()
            if not output:
                break
            output = output.replace('\n', '')  # remove newline
            last_line = output
            log.debug(output)

    if last_line == "White Wins!":
        game_result = [1, -1]
    elif last_line == "Black Wins!":
        game_result = [-1, 1]
    elif last_line == "Even!":
        game_result = [0, 0]
    else:
        log.error("Error on game result")
        return [0, 0]
    return game_result


def self_play(args, best_model, curr_model, database):
    log.info("Playing")
    total_games = database.get_game_count()
    for self_play_idx in tqdm(range(total_games+1, args.self_plays+total_games+1)):
        log.debug("Executing self play id: {0}".format(self_play_idx))
        if np.random.randint(0, 2, 1, dtype=int)[0] == 0:
            p_white = best_model.port
            p_black = curr_model.port
        else:
            p_white = curr_model.port
            p_black = best_model.port
        p_white = "tcp://localhost:{0}".format(p_white)
        p_black = "tcp://localhost:{0}".format(p_black)
        seed = np.random.randint(0, np.iinfo(np.int32).max, 1, dtype=int)[0]

        cmd_args = [args.path_to_exe, "portW", p_white, "portB", p_black,
                    "seed", str(seed), "deterministic", "0", "p0", "800", "p1", "800"]
        game_result = execute_game(cmd_args)
        database.store_and_reset(self_play_idx, game_result)


def retrain(args, model, database):
    log.info("Retraining")
    n_states = database.get_state_count()
    for i in tqdm(range(args.train_epochs)):  # select different data for each epoch
        idx = np.random.choice(np.arange(0, n_states, 1), np.min((args.train_sample_size, n_states)))
        idx = np.sort(idx)  # hdf5 requires sorted index
        idx = np.unique(idx)  # hdf5 does not allow repeating indices
        state = database.datafile["state"][idx, :, :, :]
        policy = database.datafile["policy"][idx, :, :]
        value = database.datafile["value"][idx, :]
        policy.shape = (policy.shape[0], policy.shape[1] * policy.shape[2])
        targets = {'value_head': value, 'policy_head': policy}

        fit = model.fit(state, targets, epochs=1, verbose=0, validation_split=0, batch_size=32)
        log.debug("Loss: {0}, Value: {1}, Policy: {2}".format(fit.history['loss'],
                                                              fit.history['value_head_loss'],
                                                              fit.history['policy_head_loss']))
    log.info("Final Loss: {0}, Value: {1}, Policy: {2}".format(fit.history['loss'],
                                                               fit.history['value_head_loss'],
                                                               fit.history['policy_head_loss']))


def evaluate(args, best_model, curr_model):
    log.info("Evaluating")
    scores = np.array([0, 0], dtype=np.int)  # best, current
    for game_idx in tqdm(range(args.eval_plays)):
        if np.random.randint(0, 2, 1, dtype=int)[0] == 0:
            idx_best = 0
            idx_curr = 1
            p_white = best_model.port
            p_black = curr_model.port
        else:
            idx_best = 1
            idx_curr = 0
            p_white = curr_model.port
            p_black = best_model.port
        p_white = "tcp://localhost:{0}".format(p_white)
        p_black = "tcp://localhost:{0}".format(p_black)

        cmd_args = [args.path_to_exe, "portW", p_white, "portB", p_black,
                    "deterministic", "1", "p0", "1600", "p1", "1600"]
        game_result = execute_game(cmd_args)
        if game_result[idx_best] == 1:
            scores[0] += 1
            log.debug("Evaluation game {0} result: best wins".format(game_idx))
        if game_result[idx_curr] == 1:
            scores[1] += 1
            log.debug("Evaluation game {0} result: current wins".format(game_idx))
    log.info("Result of evaluation: best wins {0}, current wins {1}".format(scores[0], scores[1]))
    decision = scores[1] > args.eval_plays * 0.55  # is current player won more than 55% of all games
    return decision


if __name__ == '__main__':
    chess_dims = (119, 8, 8)
    connect4_dims = (9, 6, 7)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tensorflow logging level
    parser = ArgumentParser()
    parser.add_argument("--human_play", action='store_true', help="Start server for human play")
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=100, help="Total number of iterations to run")
    parser.add_argument("--self_plays", type=int, default=120, help="Number of self play games to execute")
    parser.add_argument("--eval_plays", type=int, default=100, help="Number of evaluation play games to execute")
    parser.add_argument("--path_to_exe", type=str, default="/home/adamp/Documents/Codes/Hearts/build/release/Chess", help="Path to CPP MCTS exe")
    parser.add_argument("--path_to_database", type=str, default="/home/adamp/Documents/Codes/Hearts/data/chess.hdf", help="Path to HDF database")
    parser.add_argument("--train_epochs", type=int, default=300, help="Number of epochs for training")
    parser.add_argument("--train_sample_size", type=int, default=256, help="Number of game states to use for training")
    args = parser.parse_args()

    log.info("TensorFlow V: {0}, CUDA: {1}".format(tf.__version__, tf.test.is_built_with_cuda()))

    for gpu in tf.config.list_physical_devices('GPU'):
        log.info("GPU: {0}".format(gpu))
        tf.config.experimental.set_memory_growth(gpu, True)

    context = zmq.Context(1)
    best_model = DNNPredict(context, connect4_dims, port="5555")
    curr_model = DNNPredict(context, connect4_dims, port="5556")
    database = DNNStatePolicyHandler(context, args.path_to_database, connect4_dims, port="5557")

    if not os.path.isdir("/home/adamp/Documents/Codes/Hearts/models/best_0"):
        best_model.save_weight("/home/adamp/Documents/Codes/Hearts/models/best_0/weights")
        best_model.model.model.summary()
        log.info("Created new weights")
    best_model.load_weight("/home/adamp/Documents/Codes/Hearts/models/best_{0}/weights".format(args.iteration))
    curr_model.load_weight("/home/adamp/Documents/Codes/Hearts/models/best_{0}/weights".format(args.iteration))

    database.start()
    best_model.start()
    curr_model.start()

    if args.human_play:
        log.info("Server is running...")
        log.info("Playing against version: {0}".format(args.iteration))
        try:
            best_model.join()  # hang thread
        except KeyboardInterrupt:
            context.term()
            database.join()
            best_model.join()
            curr_model.join()
            exit(0)

    try:
        for iteration_idx in range(args.iteration+1, args.iteration+args.total_iterations+1):
            log.info("Starting iteration: {0}".format(iteration_idx))
            self_play(args, best_model, curr_model, database)
            retrain(args, curr_model.model.model, database)
            curr_model.save_weight("/home/adamp/Documents/Codes/Hearts/models/save_{0}/weights".format(iteration_idx))
            if evaluate(args, best_model, curr_model):
                log.info("New best model have been found in iteration: {0}".format(iteration_idx))
                curr_model.save_weight("/home/adamp/Documents/Codes/Hearts/models/best_{0}/weights".format(iteration_idx))
                best_model.load_weight("/home/adamp/Documents/Codes/Hearts/models/best_{0}/weights".format(iteration_idx))
    except KeyboardInterrupt:
        context.term()
        database.join()
        best_model.join()
        curr_model.join()
