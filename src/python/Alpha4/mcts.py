import numpy as np


class MCTSNode:
    """Represents one node in Monte-Carlo Tree"""
    def __init__(self, action, p):
        self.n = 0  # visit count
        self.p = p  # prior, computed by neural network
        self.w = np.float64(0)  # accumulated win value
        self.action = action  # action held by node, transparent for MCTS
        self.childs = []  # possible further actions if current node's action is selected


class MCTS:
    """
    Implements Monte Carlo Tree Search.
    The tree search does not know the exact problem it solves.
    Each problem must implement the same interface functions.
    """
    def __init__(self):
        """Construct tree and root node"""
        self.root = MCTSNode(None, 0)  # artificial root node
        self.sub_root = self.root  # keep complete tree for debugging

    def _policy(self, state, visited_nodes):
        """
        Applies the policy step of the Monte-Carlo Tree Search.
        :param state: state of the problem
        :param visited_nodes: list to store visited nodes
        :return: last visited node
        :return: win value of last evaluation
        """
        node = self.sub_root
        visited_nodes.append(node)  # store subroot in visited nodes
        dirichlet_noise = np.random.dirichlet([state.ALPHA], len(node.childs))
        while not state.is_finished():
            if len(node.childs) == 0:  # leaf node
                actions = state.get_actions()  # get possible actions
                p, w = state.compute_mcts_wp(actions)  # evaluate position, get prior and win
                node.childs = [MCTSNode(a, p) for a, p in zip(actions, p)]  # construct children for leaf node
                return node, w

            ucb = MCTS._get_ucb(node, dirichlet_noise, state.UCT_C)  # compute upper confidence bound
            idx = int(np.argmax(ucb))  # get highest ucb value
            node = node.childs[idx]  # select best child
            visited_nodes.append(node)  # store child in visited list
            state.update(node.action)  # update problem according to best move
            dirichlet_noise = None  # use only for subroot

        _, w = state.compute_mcts_wp([])  # compute value of terminating node
        return node, w

    @staticmethod
    def _get_ucb(parent, dirichlet_noise, ucb_c):
        """
        Computes upper confidence bound
        :param parent: node for which children' the ucb will be computed
        :param dirichlet_noise: noise applied for root node
        :param ucb_c: predefined constant for each problem
        :return: numpy array of ucb value for each node
        """
        sum_n = np.sqrt(parent.n)  # parent visit equals to the sum of each child's visit
        pnw = np.array([[child.p, child.n, child.w] for child in parent.childs])
        p = pnw[:, 0]  # priors
        n = pnw[:, 1]  # visit count
        w = pnw[:, 2]  # accumulated win
        if dirichlet_noise is not None:  # add noise for root's priors
            p = 0.75 * p + 0.25 * dirichlet_noise
        q = w / (n + np.finfo(np.float32).eps)  # value based on win values (exploitation)
        u = p * (sum_n/(1+n))  # value based on prior * visit count (exploration)

        val = q + u * ucb_c  # exploitation + exploration
        return val

    @staticmethod
    def _backprop(visited_nodes, w):
        """
        Backpropagation, write back win value for each visited node
        :param visited_nodes: nodes visited during policy search
        :param w: win value of policies last evaluation
        :return: None
        """
        for node in visited_nodes:
            node.n += 1  # increment visit count for node
            node.w += w  # accumulate computed win value

    @staticmethod
    def _move_deterministic(childs):
        """
        Select best move based on the visit count.
        :param childs: children of root node
        :return: index of selected child
        """
        n = np.array([child.n for child in childs])  # get visit count for children
        idx = np.argmax(n)  # select highest
        return idx

    @staticmethod
    def _move_stochastic(childs, time):
        """
        Select best move based on a distribution.
        :param childs: children of root node
        :param time: current time of problem
        :return: index of selected child
        :return: distribution probability for each children
        """
        tau = 1.0 if time < 60 else 0.05
        n = np.array([child.n for child in childs])  # get visit count for children
        pi = np.power(n, 1/tau)
        pi = pi / np.sum(pi)  # normalize distribution
        idx = np.random.choice(np.arange(len(pi), dtype=int), 1, p=pi)[0]  # select from distribution
        return idx, pi

    def execute(self, iterations, cstate, is_deterministic, log):
        """
        Execute MCTS+DNN algorithm to get next action for problem
        :param iterations: number of policy iterations
        :param cstate: state of the problem, do not modify this variable
        :param is_deterministic: move selection (deterministic or stochastic)
        :param log: logging object
        :return: decided action to take in next step
        :return: state of the current problem for DNN input, only for stochastic else None
        :return: policy of the current problem for DNN output, only for stochastic else None
        """
        for i in range(iterations):
            policy_nodes = []  # visited nodes during current iteration
            state = cstate.copy()  # make mandatory copy of problem
            node, w = self._policy(state, policy_nodes)  # perform search in tree
            self._backprop(policy_nodes, w)  # write back results to visited nodes

        state = None
        policy = None
        if is_deterministic:
            idx = MCTS._move_deterministic(self.sub_root.childs)  # get index of action to select
        else:
            time = cstate.get_time()  # time of the problem (e.g. steps made)
            state = cstate.get_game_state_dnn()  # get input layer for dnn training
            idx, pi = MCTS._move_stochastic(self.sub_root.childs, time)  # get index of action to select
            actions = [child.action for child in self.sub_root.childs]  # get all possible actions
            policy = cstate.get_policy_train_dnn(actions, pi)  # get output layer for dnn training

        for i in range(len(self.sub_root.childs)):
            child = self.sub_root.childs[i]
            log.debug("{0}; W: {1}; N: {2}; Q: {3}".format(child.action, child.w, child.n, child.w / child.n))

        new_sub_root = self.sub_root.childs[idx]  # select best action
        self.sub_root = new_sub_root  # set new subroot (keep root, so the complete tree is stored, debugging)
        return new_sub_root.action, state, policy

    def update(self, action, log):
        """
        Update tree according to opponent's move
        :param action: last action made by opponent
        :param log: logging object
        :return: None
        """
        if len(self.sub_root.childs) == 0:
            self.sub_root.childs = [MCTSNode(action, 1.0)]
            self.sub_root = self.sub_root.childs[0]
            return
        for child in self.sub_root.childs:
            if child.action == action:
                self.sub_root = child
                return
        log.critical("Action not found during update")
        exit(-1)
