"""
Main function to set up AlphaZero framework and to execute it
Functions for self-play simulation and self-play evaluation.

Author: AdamP 2020-2020
"""

import os
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from Alpha4.mcts import MCTS
# from Alpha4.model import DNNPredict as Predict
# from Alpha4.model_rt import DNNPredictRT as Predict
from Alpha4.model_lite import DNNPredictLite as Predict
from Alpha4.database import Database
from Alpha4.connect4 import Connect4 as Game
from Alpha4.logger import get_logger, add_file_logger


def self_play(args, best_model, curr_model, database, log):
    """Simulate games with stochastic decisions and store simulated data for DNN training"""
    log.info("Playing")
    total_games = database.get_game_count()
    for self_play_idx in tqdm(range(total_games+1, args.self_plays+total_games+1)):
        log.debug("Executing self play id: {0}".format(self_play_idx))

        # select white, black
        best_idx, curr_idx = np.random.choice([0, 1], 2, replace=False)
        models = [None, None]
        models[best_idx] = best_model
        models[curr_idx] = curr_model

        # init random generator
        seed = np.random.randint(0, np.iinfo(np.int32).max, 1, dtype=int)[0]
        np.random.seed(seed)

        # simulate one game
        states = [[], []]
        policies = [[], []]
        mcts = [MCTS(log), MCTS(log)]  # tree for both players
        game = Game(models[0], models[1])  # representation of game
        while not game.is_finished():
            # execute search for current player, returns action to select, current state of game and computed policy
            idx_ai = game.get_player()
            action, state, policy = mcts[idx_ai].execute(800, game, is_deterministic=False)

            # update tree and game
            for player_idx in range(2):
                mcts[player_idx].update(action)
            game.update(action)

            # store input state and output policy for DNN training
            states[idx_ai].append(state)
            policies[idx_ai].append(policy)
            log.debug("State of game {0}:{1}{2}".format(self_play_idx, os.linesep, game))
            # log.debug("Input DNN State:{0}{1}".format(os.linesep, game.get_game_state_dnn()))

        # get end result and store training data
        result = game.get_result()
        for player_idx in range(2):
            # add end state for training
            policies[player_idx].append(np.zeros(shape=Game.dims_policy))
            states[player_idx].append(game.get_game_state_dnn(player_idx, player_idx))
            # generate value vector for each state
            results = np.zeros(len(states[player_idx])) + result[player_idx]
            database.store(self_play_idx, states[player_idx], policies[player_idx], results)


def evaluate(args, best_model, curr_model, log):
    """Execute games with deterministic decisions and evaluate based on game outcomes"""
    log.info("Evaluating")
    scores = np.array([0, 0], dtype=np.int)  # result of evaluations: [best, current]
    for game_idx in tqdm(range(args.eval_plays)):

        # select white, black
        best_idx, curr_idx = np.random.choice([0, 1], 2, replace=False)
        models = [None, None]
        models[best_idx] = best_model
        models[curr_idx] = curr_model

        # execute one game
        mcts = [MCTS(log), MCTS(log)]
        game = Game(models[0], models[1])
        while not game.is_finished():
            # execute search for current player
            idx_ai = game.get_player()
            action = mcts[idx_ai].execute(1600, game, is_deterministic=True)

            # update tree and game
            for player_idx in range(2):
                mcts[player_idx].update(action)
            game.update(action)
            log.debug("State of evaluation game {0}:{1}{2}".format(game_idx, os.linesep, game))
        result = game.get_result()

        if result[best_idx] == 1:
            scores[0] += 1
            log.debug("Evaluation game {0} result: best wins".format(game_idx))
        elif result[curr_idx] == 1:
            scores[1] += 1
            log.debug("Evaluation game {0} result: current wins".format(game_idx))
        else:
            log.debug("Evaluation game {0} result: game was even".format(game_idx))

    log.info("Result of evaluation: best wins {0}, current wins {1}".format(scores[0], scores[1]))
    decision = scores[1] > args.eval_plays * 0.55  # is current player won more than 55% of all games
    return decision


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tensorflow logging level

    # handle input arguments
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="Root dir of project", required=True)
    # parser.add_argument("--human_play", action='store_true', help="Start server for human play")
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=10, help="Total number of iterations to run")
    parser.add_argument("--self_plays", type=int, default=250, help="Number of self play games to execute")
    parser.add_argument("--eval_plays", type=int, default=60, help="Number of evaluation play games to execute")
    parser.add_argument("--path_to_database", type=str, default=Game.name+'.hdf', help="Path to HDF file")
    parser.add_argument("--train_epochs", type=int, default=300, help="Number of epochs for training")
    parser.add_argument("--train_sample_size", type=int, default=256, help="Number of game states to use for training")
    args = parser.parse_args()

    # set up logging
    log = get_logger()
    add_file_logger(log, os.path.join(args.root_dir, 'data', Game.name+'.log'))
    if args.path_to_database == Game.name+'.hdf':
        args.path_to_database = os.path.join(args.root_dir, 'data', Game.name+'.hdf')

    # set up TF GPU
    log.info("TensorFlow V: {0}, CUDA: {1}".format(tf.__version__, tf.test.is_built_with_cuda()))
    for gpu in tf.config.list_physical_devices('GPU'):
        log.info("GPU: {0}".format(gpu))
        tf.config.experimental.set_memory_growth(gpu, True)

    # open database and init models
    database = Database(log, args.path_to_database, Game.dims_state, Game.dims_policy)
    tf_lite_args = {"database": database, "delegate": "libedgetpu.so.1", "device": "usb:0", "compile_tpu": True}
    curr_model = Predict(log, Game.dims_state, Game.dims_policy, **tf_lite_args)
    best_model = Predict(log, Game.dims_state, Game.dims_policy, **tf_lite_args)

    # load model weights and if available frozen models
    if not os.path.isdir(os.path.join(args.root_dir, 'models', 'best_0')):
        best_model.save(os.path.join(args.root_dir, 'models', 'best_0'))
        best_model.model.summary()
        log.info("Created new weights")
    best_model.load(os.path.join(args.root_dir, 'models', 'best_{0}'.format(args.iteration)))
    curr_model.load(os.path.join(args.root_dir, 'models', 'best_{0}'.format(args.iteration)))

    # Test Code, only for DNNPredict
    # curr_model.load(os.path.join(args.root_dir, 'models', 'save_1'))
    # value, policy = curr_model.predict(np.array(database.datafile["state"]))
    # print(np.unique(value, return_counts=True))
    # exit(0)

    try:  # main loop for AlphaZero workflow
        for iteration_idx in range(args.iteration+1, args.iteration+args.total_iterations+1):
            log.info("Starting iteration: {0}".format(iteration_idx))
            self_play(args, best_model, curr_model, database, log)
            curr_model.retrain(args, database)
            curr_model.save(os.path.join(args.root_dir, 'models', 'save_{0}'.format(iteration_idx)))
            if evaluate(args, best_model, curr_model, log):
                log.info("New best model have been found in iteration: {0}".format(iteration_idx))
                curr_model.save(os.path.join(args.root_dir, 'models', 'best_{0}'.format(iteration_idx)))
                best_model.load(os.path.join(args.root_dir, 'models', 'best_{0}'.format(iteration_idx)))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
