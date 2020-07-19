import os
import threading
import subprocess
from argparse import ArgumentParser

import zmq
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from Alpha4.database import Database
from Alpha4.model_rt import DNNPredictRT as Predict
from Alpha4.logger import get_logger, add_file_logger
from Alpha4.connect4 import Connect4 as Problem


def plot_state(state, policy):
    import matplotlib.pyplot as plt
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
    import matplotlib.pyplot as plt
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


class DNNStatePolicyHandler(threading.Thread, Database):
    def __init__(self, log, datafilepath_hdf, dims_state, dims_policy, zmq_context, port="5557"):
        threading.Thread.__init__(self)
        Database.__init__(self, log, datafilepath_hdf, dims_state, dims_policy)
        self.socket = zmq_context.socket(zmq.REP)
        self.socket.bind("tcp://*:{0}".format(port))

        self.dims_state = dims_state
        self.dims_policy = dims_policy
        self.data_state = []
        self.data_policy = []

    def store_and_reset(self, game_idx, game_result):
        self.store(game_idx, self.data_state, self.data_policy, game_result)

        self.datafile.flush()
        self.data_state.clear()
        self.data_policy.clear()

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
                state.shape = (self.dims_state[2], self.dims_state[0], self.dims_state[1])
                send_ok(self.socket)

                # receive policy
                message = self.socket.recv()
                policy = np.frombuffer(message, dtype=np.float32)
                policy.shape = (self.dims_policy[2], self.dims_policy[0], self.dims_policy[1])
                send_ok(self.socket)

                # NCWH to NWHC
                state = np.transpose(state, axes=(1, 2, 0))
                policy = np.transpose(policy, axes=(1, 2, 0))

                self.data_state.append(state)
                self.data_policy.append(policy)
            except zmq.error.ContextTerminated:
                self.socket.close()
                return


class DNNPredict(threading.Thread, Predict):
    def __init__(self, input_dim, output_dim, zmq_context, port="5555"):
        threading.Thread.__init__(self)
        Predict.__init__(self, input_dim, output_dim)
        self.socket = zmq_context.socket(zmq.REP)
        self.socket.bind("tcp://*:{0}".format(port))
        self.port = port

    def run(self):
        while True:
            try:
                # get state
                message = self.socket.recv()
                state = np.frombuffer(message, dtype=np.float32)
                state.shape = (1, self.input_dim[2], self.input_dim[0], self.input_dim[1])
                state = np.transpose(state, axes=(0, 2, 3, 1))  # NCWH to NWHC

                # predict
                value, policy = self.predict(state)

                # NWHC to NCWH
                policy.shape = self.output_dim
                policy = np.transpose(policy, axes=(2, 0, 1))
                policy.shape = (policy.size, )

                # send prediction
                data = np.empty((1, self.output_dim[0]*self.output_dim[1]*self.output_dim[2]+1), dtype=np.float32)
                data[0, :self.output_dim[0]*self.output_dim[1]*self.output_dim[2]] = policy
                data[0, self.output_dim[0]*self.output_dim[1]*self.output_dim[2]] = value
                data = np.array(data).tobytes()
                self.socket.send(data)

            except zmq.error.ContextTerminated:
                self.socket.close()
                return


def execute_game(log, cmd_args):
    last_line = ""
    process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while process.poll() is None:
        while True:
            output = process.stdout.readline().decode()
            if not output:
                break
            output = output.replace('\n', '')  # remove newline
            output = output.replace('\r', '')  # remove newline
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


def self_play(log, args, best_model, curr_model, database):
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
        game_result = execute_game(log, cmd_args)
        database.store_and_reset(self_play_idx, game_result)


def evaluate(log, args, best_model, curr_model):
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
        game_result = execute_game(log, cmd_args)
        if game_result[idx_best] == 1:
            scores[0] += 1
            log.debug("Evaluation game {0} result: best wins".format(game_idx))
        if game_result[idx_curr] == 1:
            scores[1] += 1
            log.debug("Evaluation game {0} result: current wins".format(game_idx))
    log.info("Result of evaluation: best wins {0}, current wins {1}".format(scores[0], scores[1]))
    decision = scores[1] > args.eval_plays * 0.55  # is current player won more than 55% of all games
    return decision


def main():
    # chess_dims = (119, 8, 8)
    db_name = Problem.name+'.hdf'
    exe_name = "Connect4"
    dims_state = Problem.dims_state
    dims_policy = Problem.dims_policy
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tensorflow logging level

    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="Root dir of project", required=True)
    parser.add_argument("--human_play", action='store_true', help="Start server for human play")
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=100, help="Total number of iterations to run")
    parser.add_argument("--self_plays", type=int, default=120, help="Number of self play games to execute")
    parser.add_argument("--eval_plays", type=int, default=100, help="Number of evaluation play games to execute")
    parser.add_argument("--path_to_exe", type=str, default=exe_name, help="Path to CPP MCTS exe")
    parser.add_argument("--path_to_database", type=str, default=db_name, help="Path to HDF database")
    parser.add_argument("--train_epochs", type=int, default=300, help="Number of epochs for training")
    parser.add_argument("--train_sample_size", type=int, default=256, help="Number of game states to use for training")
    args = parser.parse_args()

    log = get_logger()
    add_file_logger(log, os.path.join(args.root_dir, 'data', 'connect4.log'))
    if args.path_to_database == 'connect4.hdf':
        args.path_to_database = os.path.join(args.root_dir, 'data', 'connect4.hdf')
    if args.path_to_exe == 'Connect4':
        args.path_to_exe = os.path.join(args.root_dir, 'build', 'release', 'Connect4')

    log.info("TensorFlow V: {0}, CUDA: {1}".format(tf.__version__, tf.test.is_built_with_cuda()))
    for gpu in tf.config.list_physical_devices('GPU'):
        log.info("GPU: {0}".format(gpu))
        tf.config.experimental.set_memory_growth(gpu, True)

    context = zmq.Context(1)
    best_model = DNNPredict(dims_state, dims_policy, context, port="5555")
    curr_model = DNNPredict(dims_state, dims_policy, context, port="5556")
    database = DNNStatePolicyHandler(log, args.path_to_database, dims_state, dims_policy, context, port="5557")

    if not os.path.isdir(os.path.join(args.root_dir, 'models', 'best_0')):
        best_model.save(os.path.join(args.root_dir, 'models', 'best_0'))
        best_model.model.model.summary()
        log.info("Created new weights")
    best_model.load(os.path.join(args.root_dir, 'models', 'best_{0}'.format(args.iteration)))
    curr_model.load(os.path.join(args.root_dir, 'models', 'best_{0}'.format(args.iteration)))

    # Test Code
    # curr_model.load_weight(os.path.join(project_dir, 'models', 'save_1', 'weights'))
    # value, policy = curr_model.model.model.predict(np.array(database.datafile["state"]), batch_size=512)
    # print(np.unique(value, return_counts=True))

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
            self_play(log, args, best_model, curr_model, database)
            curr_model.retrain(args, curr_model.model.model, database)
            curr_model.save(os.path.join(args.root_dir, 'models', 'save_{0}'.format(iteration_idx)))
            if evaluate(log, args, best_model, curr_model):
                log.info("New best model have been found in iteration: {0}".format(iteration_idx))
                curr_model.save(os.path.join(args.root_dir, 'models', 'best_{0}'.format(iteration_idx)))
                best_model.load(os.path.join(args.root_dir, 'models', 'best_{0}'.format(iteration_idx)))
    except KeyboardInterrupt:
        pass
    context.term()
    database.join()
    best_model.join()
    curr_model.join()


if __name__ == '__main__':
    main()
