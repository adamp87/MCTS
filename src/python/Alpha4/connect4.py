"""
Implementation of Connect4, a problem to be solved by AlphaZero framework.

Author: AdamP 2020-2020
"""

import numpy as np


class Connect4Action:
    """Represents an action for Connect4"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return "X{0}Y{1}".format(self.x+1, self.y+1)


class Connect4:
    """
    Implementation of the game Connect4.
    Implements the required interfaces for MCTS and for the AlphaZero framework.
    """
    ALPHA = 1.0 / 7.0  # in average seven actions are possible, value for dirichlet distribution
    UCT_C = 1.0  # constant for ucb computation, for weighting exploration
    TAU_TIME = 15  # number of steps after stochastic games change tau value
    dnn_history = 2  # number of previous turns to add to dnn
    dims_state = (6, 7, 2 * (dnn_history + 1) + 1)  # dimension of the input state
    dims_policy = (6, 7, 1)  # dimension of the output policy
    name = "connect4"  # helper for filenames

    def __init__(self, model_p1, model_p2):
        self.time = 0  # incremented on each move
        self.history = []  # previous board positions
        self.models = [model_p1, model_p2]  # can be [white, black] or [black, white]
        self.finished = [False, False]  # flag for each player
        self.board = np.zeros((6, 7), dtype=np.int8)  # 0-empty, 1-p1, 2-p2

    def __str__(self):
        fig = np.array([' ', 'O', 'X'], dtype=str)
        board = np.flipud(self.board)  # start printing out from top
        return np.array2string(fig[board], separator='|')

    def copy(self):
        """Make deep-copy of object, TFModels are not copied"""
        copy = Connect4(self.models[0], self.models[1])
        copy.time = self.time
        copy.board = self.board.copy()
        copy.history = self.history.copy()
        copy.finished = self.finished.copy()
        return copy

    def get_time(self):
        """Return the current time(turn) number"""
        return self.time

    def get_player(self, time=-1):
        """Return idx of current player"""
        if time == -1:
            time = self.time
        return time % 2

    def is_finished(self):
        """Return True if game has finished"""
        return self.finished[0] or self.finished[1]

    def get_actions(self):
        """Return possible and valid actions to take in current state"""
        actions = []
        for x in range(7):  # from left to right
            if self.board[5, x] != 0:
                continue  # top element is filled
            for y in range(6):  # from bottom to top
                if self.board[y, x] != 0:
                    continue  # stone is placed in position
                actions.append(Connect4Action(x, y))  # first empty spot found, store action
                break  # rule of gravity
        return actions

    def update(self, action):
        """Update game state based on action, note: action validity is not checked"""
        def is_inside(_x, _y, dx, dy):
            """Helper function: is position in board"""
            xx = _x + dx
            yy = _y + dy
            return 0 <= xx < 7 and 0 <= yy < 6

        def is_own_stone(_x, _y, dx, dy):
            """Helper function: is stone current players stone"""
            return is_inside(_x, _y, dx, dy) and self.board[_y+dy, _x+dx] == self.board[_y, _x]

        def scan_line(_x, _y, ddx, ddy):
            """Helper function: check in one line if current player won"""
            count = 1
            for n in range(1, 7):
                if not is_own_stone(_x, _y, n * ddx, n * ddy):
                    break  # empty or opponent stone
                count += 1
            return count >= 4

        empty_count = 0  # count empty positions
        idx_ai = self.get_player()  # idx of current player
        self.history.append(self.board.copy())  # store last state in history
        self.board[action.y, action.x] = idx_ai + 1  # update board with action (board stores idx+1)

        # verify if player won or game even
        for y in range(6):
            for x in range(7):
                if self.board[y, x] == 0:
                    empty_count += 1  # empty position
                if self.board[y, x] != idx_ai+1:
                    continue  # opponents stone

                # check if current player won
                if scan_line(x, y, 0, 1):  # horizontal
                    self.finished[idx_ai] = True
                if scan_line(x, y, 1, 0):  # vertical
                    self.finished[idx_ai] = True
                if scan_line(x, y, +1, 1):  # right diagonal
                    self.finished[idx_ai] = True
                if scan_line(x, y, -1, 1):  # left diagonal
                    self.finished[idx_ai] = True

        # if there is no empty positions and no winner, game is even
        if empty_count == 0 and not self.finished[0] and not self.finished[1]:
            self.finished[0] = self.finished[1] = True  # even

        self.time += 1  # increment time

    def get_game_state_dnn(self, idx_ai, idx_move):
        """Return input state for DNN"""
        p1_start = 0
        p2_start = Connect4.dnn_history + 1
        fig_ai = idx_ai + 1
        fig_op = ((idx_ai + 1) % 2) + 1
        state = np.zeros(Connect4.dims_state, dtype=np.float32)
        boards = [self.board] + [self.history[-1 - i] for i in range(min(Connect4.dnn_history, len(self.history)))]
        for t in range(len(boards)):
            board = boards[t]
            state[board == fig_ai, p1_start + t] = 1  # one-hot encoding for player ai
            state[board == fig_op, p2_start + t] = 1  # one-hot encoding for player op
        state[:, :, -1] = idx_ai  # layer to encode player's color
        return state

    def compute_mcts_wp(self, idx_ai, idx_move, actions):
        """
        Evaluate current state with DNN and predict priors for actions.
        :param idx_ai: index of ai(DNN) to be executed for prediction
        :param idx_move: index of player who makes next move
        :param actions: Actions to predict prior for.
        :return: prediction of policy layer -> priors
        :return: prediction of value layer -> win
        """
        # perform state evaluation with DNN
        state = self.get_game_state_dnn(idx_ai, idx_move)
        state = np.expand_dims(state, axis=0)  # batch size 1
        value, policy = self.models[idx_ai].predict(state)
        policy.shape = Connect4.dims_policy  # policy is flattened, reshape

        policy = np.array([policy[act.y, act.x, 0] for act in actions])  # collect valid policy values
        policy = np.exp(policy) / (np.sum(np.exp(policy)) + np.finfo(np.float32).eps)  # softmax
        return policy, value

    @staticmethod
    def get_policy_train_dnn(actions, pi):
        """
        Prepare the output policy layer of DNN for training
        :param actions: valid actions to set priors for
        :param pi: computed prior values by MCTS
        :return: policy layer
        """
        policy = np.zeros(Connect4.dims_policy, dtype=np.float32)
        action_idx = [(act.y, act.x, 0) for act in actions]  # indices where valid actions present in board
        for i in range(len(action_idx)):
            policy[action_idx[i]] = pi[i]  # set positions with priors
        return policy

    def get_result(self):
        """Return a list of two elements, describing win/lose/even for both players"""
        if self.finished[0] and self.finished[1]:
            return [0, 0]  # even
        if self.finished[0]:
            return [1, -1]  # p1 wins
        if self.finished[1]:
            return [-1, 1]  # p2 wins
        raise Exception("Result requested when game is not finished")
