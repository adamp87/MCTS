import numpy as np
from os import linesep


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
    A problem to be solved by the MCTS.
    """
    ALPHA = 1.0 / 7.0  # in average seven actions are possible
    UCT_C = 1.0  # constant for ucb computation
    dims_state = (6, 7, 3)  # dimension of the input state
    dims_policy = (6, 7, 1)  # dimension of the output policy
    name = "connect4"

    def __init__(self, model_p1, model_p2):
        self.time = 0
        self.models = [model_p1, model_p2]  # can be [white, black] or [black, white]
        self.finished = [False, False]  # flag for each player
        self.board = np.zeros((6, 7), dtype=np.int8)  # 0-empty, 1-p1, 2-p2

    def __str__(self):
        ret = ""
        fig = [' ', 'O', 'X']
        board = np.flip(self.board)  # start printing out from top
        for y in range(6):
            for x in range(7):
                ret += '|'
                ret += fig[board[y, x]]
            ret += '|'+linesep
        return ret

    def copy(self):
        """Make deep-copy of object, TFModels are not copied"""
        copy = Connect4(self.models[0], self.models[1])
        copy.time = self.time
        copy.board = self.board.copy()
        copy.finished = self.finished.copy()
        return copy

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
        """Update state based on action, note: action validity is not checked"""
        # TODO doc
        def is_inside(x, y, dx, dy):
            xx = x + dx
            yy = y + dy
            return 0 <= xx < 7 and 0 <= yy < 6

        def is_own_stone(x, y, dx, dy):
            return is_inside(x, y, dx, dy) and self.board[y+dy, x+dx] == self.board[y, x]

        def scan_line(x, y, ddx, ddy):
            count = 1
            for n in range(1, 7):
                if not is_own_stone(x, y, n * ddx, n * ddy):
                    break
                count += 1
            return count >= 4

        empyt_count = 0
        idx_ai = self.get_player()
        self.board[action.y, action.x] = idx_ai + 1

        for y in range(6):
            for x in range(7):
                if self.board[y, x] == 0:
                    empyt_count += 1
                if self.board[y, x] != idx_ai+1:
                    continue

                if scan_line(x, y, 0, 1):
                    self.finished[idx_ai] = True
                if scan_line(x, y, 1, 0):
                    self.finished[idx_ai] = True
                if scan_line(x, y, 1, 1):
                    self.finished[idx_ai] = True
                if scan_line(x, y, -1, 1):
                    self.finished[idx_ai] = True

        if empyt_count == 0 and not self.finished[0] and not self.finished[1]:
            self.finished[0] = self.finished[1] = True  # even

        self.time += 1

    def get_game_state_dnn(self):
        """Return input state for DNN"""
        # TODO history
        input = np.zeros(Connect4.dims_state, dtype=np.float32)
        input[self.board == 1, 0] = 1  # one-hot encoding for player 1
        input[self.board == 2, 1] = 1  # one-hot encoding for player 2
        input[:, :, 2] = self.get_player()  # layer to encode current player
        return input

    def compute_mcts_wp(self, actions):
        """
        Evaluate current state with DNN and predict priors for actions.
        :param actions: Actions to predict prior for.
        :return: prediction of policy layer -> priors
        :return: prediction of value layer -> win
        """

        # perform state evaluation with DNN
        player_idx = self.get_player()
        input = self.get_game_state_dnn()
        input = np.expand_dims(input, axis=0)  # batch size 1
        value, policy = self.models[player_idx].predict(input)
        policy.shape = Connect4.dims_policy  # policy is flattened, reshape

        policy = np.array([policy[act.y, act.x, 0] for act in actions])  # collect valid policy values
        policy = np.exp(policy) / (np.sum(policy) + np.finfo(np.float32).eps)  # softmax
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
