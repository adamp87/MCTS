import os
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from Alpha4.mcts import MCTS
from Alpha4.model_lite import DNNPredictLite as Predict
from Alpha4.database import Database
from Alpha4.connect4 import Connect4 as Problem
from Alpha4.logger import get_logger, add_file_logger


def self_play(args, best_model, curr_model, database, log):
    log.info("Playing")
    total_games = database.get_game_count()
    for self_play_idx in tqdm(range(total_games+1, args.self_plays+total_games+1)):
        log.debug("Executing self play id: {0}".format(self_play_idx))

        best_idx, curr_idx = np.random.choice([0, 1], 2, replace=False)
        models = [None, None]
        models[best_idx] = best_model
        models[curr_idx] = curr_model
        seed = np.random.randint(0, np.iinfo(np.int32).max, 1, dtype=int)[0]
        np.random.seed(seed)

        states = []
        policies = []
        mcts = [MCTS(), MCTS()]
        problem = Problem(models[0], models[1])
        while not problem.is_finished():
            idx_ai = problem.get_player()
            idx_op = (idx_ai + 1) % 2
            action, state, policy = mcts[idx_ai].execute(800, problem, False, log)
            mcts[idx_op].update(action, log)
            problem.update(action)

            states.append(state)
            policies.append(policy)
            log.debug("State of game {0}:{1}{2}".format(self_play_idx, os.linesep, problem))
            # log.debug("Input DNN State:{0}{1}".format(os.linesep, problem.get_game_state_dnn()))
        result = problem.get_result()
        result = np.tile(result, int(np.ceil(len(states)/2)))[:len(states)]

        database.store(self_play_idx, states, policies, result)


def evaluate(args, best_model, curr_model, log):
    log.info("Evaluating")
    scores = np.array([0, 0], dtype=np.int)  # best, current
    for game_idx in tqdm(range(args.eval_plays)):

        best_idx, curr_idx = np.random.choice([0, 1], 2, replace=False)
        models = [None, None]
        models[best_idx] = best_model
        models[curr_idx] = curr_model

        mcts = [MCTS(), MCTS()]
        problem = Problem(models[0], models[1])
        while not problem.is_finished():
            idx_ai = problem.get_player()
            idx_op = (idx_ai + 1) % 2
            action, _, _ = mcts[idx_ai].execute(1600, problem, True, log)
            mcts[idx_op].update(action, log)
            problem.update(action)
        result = problem.get_result()

        if result[best_idx] == 1:
            scores[0] += 1
            log.debug("Evaluation game {0} result: best wins".format(game_idx))
        if result[curr_idx] == 1:
            scores[1] += 1
            log.debug("Evaluation game {0} result: current wins".format(game_idx))

    log.info("Result of evaluation: best wins {0}, current wins {1}".format(scores[0], scores[1]))
    decision = scores[1] > args.eval_plays * 0.55  # is current player won more than 55% of all games
    return decision


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tensorflow logging level

    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="Root dir of project", required=True)
    # parser.add_argument("--human_play", action='store_true', help="Start server for human play")
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=100, help="Total number of iterations to run")
    parser.add_argument("--self_plays", type=int, default=120, help="Number of self play games to execute")
    parser.add_argument("--eval_plays", type=int, default=100, help="Number of evaluation play games to execute")
    parser.add_argument("--path_to_database", type=str, default=Problem.name+'.hdf', help="Path to HDF file")
    parser.add_argument("--train_epochs", type=int, default=300, help="Number of epochs for training")
    parser.add_argument("--train_sample_size", type=int, default=256, help="Number of game states to use for training")
    args = parser.parse_args()

    log = get_logger()
    add_file_logger(log, os.path.join(args.root_dir, 'data', Problem.name+'.log'))
    if args.path_to_database == Problem.name+'.hdf':
        args.path_to_database = os.path.join(args.root_dir, 'data', Problem.name+'.hdf')

    log.info("TensorFlow V: {0}, CUDA: {1}".format(tf.__version__, tf.test.is_built_with_cuda()))
    for gpu in tf.config.list_physical_devices('GPU'):
        log.info("GPU: {0}".format(gpu))
        tf.config.experimental.set_memory_growth(gpu, True)

    database = Database(log, args.path_to_database, Problem.dims_state, Problem.dims_policy)
    best_model = Predict(log, Problem.dims_state, Problem.dims_policy)
    curr_model = Predict(log, Problem.dims_state, Problem.dims_policy)

    if not os.path.isdir(os.path.join(args.root_dir, 'models', 'best_0')):
        best_model.save(os.path.join(args.root_dir, 'models', 'best_0'))
        best_model.model.summary()
        log.info("Created new weights")
    best_model.load(os.path.join(args.root_dir, 'models', 'best_{0}'.format(args.iteration)))
    curr_model.load(os.path.join(args.root_dir, 'models', 'best_{0}'.format(args.iteration)))

    # Test Code
    # curr_model.load_weight(os.path.join(args.root_dir, 'models', 'save_1', 'weights'))
    # value, policy = curr_model.model.predict(np.array(database.datafile["state"]), batch_size=512)
    # print(np.unique(value, return_counts=True))

    try:
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
