#ifndef MCTS_HPP
#define MCTS_HPP

#include <mutex>
#include <atomic>
#include <vector>
#include <memory>
#include <limits>
#include <random>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <functional>

//! Base class for single-thread MCTS::Node implementation
template <typename T_Act>
struct MCTSNodeBase {
    typedef T_Act ActType;
    typedef std::uint_fast32_t CountType;

    //! Dummy LockGuard, helps to keep policy transparent
    class LockGuard {
    public:
        LockGuard(MCTSNodeBase<T_Act>&) {}
    };

    CountType   N;      //!< state visit count
    double      W;      //!< total value of state
    double      P;      //!< prior probability to select action
    T_Act       action; //!< action that takes to state, e.g. card played out

    MCTSNodeBase(const T_Act& action)
        : N(0), W(0.0), P(0.0), action(action)
    { }
};

//! Base class for multi-thread MCTS::Node implementation
template <typename T_Act>
struct MCTSNodeBaseMT {
    typedef T_Act ActType;
    typedef std::uint_fast32_t CountType;

    //! LockGuard inherited from std, constructor takes node
    class LockGuard : public std::lock_guard<std::mutex> {
    public:
        LockGuard(MCTSNodeBaseMT<T_Act>& node)
            : std::lock_guard<std::mutex>(node.lock)
        {}
    };

    //! Workaround, until C20 standard enables atomic operations for double
    class MyAtomicDouble {
        std::atomic<double> value;

    public:
        MyAtomicDouble() : value(0) {}
        operator double() const {
            return value;
        }
        void operator+=(double& val) {
            double prev = value;
            double next = prev + val;
            while (!value.compare_exchange_weak(prev, next)) {}
        }
    };

    std::atomic<CountType>  N;      //!< state visit count
    MyAtomicDouble          W;      //!< total value of state
    double                  P;      //!< prior probability to select action
    T_Act                   action; //!< action that takes to state, e.g. card played out
    std::mutex              lock;   //!< mutex for thread-safe policy

    MCTSNodeBaseMT(const T_Act& action)
        : N(0), P(0.0), action(action)
    { }
};

//! Monte Carlo tree search to apply AI
/*!
 * \details This class implements Monte Carlo tree search as described in Wikipedia.
 *          It follows the policy-rollout-backprop idea.
 *          Before policy, it walks the tree according to the history of the state of the problem, this is called catchup.
 *          History must be handled by the main program.
 *          The tree search does not know the exact problem it solves.
 *          Interfacing with the problem is done by template interfaces.
 * \author adamp87
*/
template <class TProblem, class TNodeBase>
class MCTS {

    //! One node with childs, interface
    class Node : public TNodeBase {
    private:
        friend class MCTS;
        typedef typename TNodeBase::ActType ActType;

        std::vector<std::unique_ptr<Node> > childs; //!< store childs as unique pointers

        Node(const ActType& action) : TNodeBase(action) {}
        Node(const Node&) = delete;

        //! Add child node
        Node* add(const ActType& action) {
            // init leaf node
            std::unique_ptr<Node> child(new Node(action));

            childs.push_back(std::move(child));
            return childs.back().get();
        }

        //! Get number of childs
        size_t size() const { return childs.size(); }

        //! Can be used for debug
        size_t getNodeId() const {
            return reinterpret_cast<size_t>(this);
        }
    };

    typedef typename Node* NodePtr;
    typedef typename TNodeBase::LockGuard LockGuard;
    typedef typename TNodeBase::ActType ActType;
    typedef typename TNodeBase::CountType CountType;
    typedef typename std::uint_fast32_t ActCounterType;

private:
    std::unique_ptr<Node> root; //!< root of the tree
    std::default_random_engine generator; //!< random generator

private:
    //! Walk the tree according to the history of the problem
    NodePtr catchup(const TProblem& state, const std::vector<ActType>& history) {
        NodePtr node = root.get();

        for (size_t time = 0; time < history.size(); ++time) {
            bool found = false;
            for (auto it = node->childs.begin(); it != node->childs.end(); ++it) {
                NodePtr child = it->get();
                if (child->action == history[time]) {
                    node = child;
                    found = true;
                    break;
                }
            }
            if (!found) { // no child, update tree according to history
                node = node->add(history[time]);
            }
        }
        return node;
    }

    //! Applies the policy step of the Tree Search
    NodePtr policy(const NodePtr subRoot, TProblem& state, int idxAi, std::vector<NodePtr>& visited_nodes, double& W) {
        NodePtr node = subRoot;
        visited_nodes.push_back(node); // store subroot as policy

        //dirchlet
        double ratio = 0.75; // ratio of noise for root priors
        std::vector<double> dirichlet(node->size());
        computeDirichlet(dirichlet);

        while (!state.isFinished()) {

            if (node->N == 0) { // leaf node, N will be increased by backprop
                LockGuard guard(*node); (void)guard; // thread-safe scope on current leaf node
                if (node->size() == 0) { // enter if have no child
                    double P[TProblem::MaxActions];
                    ActType actions[TProblem::MaxActions];

                    ActCounterType nActions = state.getPossibleActions(idxAi, state.getPlayer(), actions);
                    state.computeMCTS_WP(idxAi, actions, nActions, P, W);
                    for (ActCounterType i = 0; i < nActions; ++i) { // add all child nodes as leaf nodes
                        NodePtr n = node->add(actions[i]);
                        n->P = P[i];
                    }
                    return node;
                }
            }

            // node fully expanded
            // set node to best leaf
            NodePtr best = node; // init
            double best_val = -std::numeric_limits<double>::max();
            double subRootVisitSqrt = sqrt(std::max(static_cast<ActCounterType>(node->N), (ActCounterType)1));
            for (size_t i = 0; i < node->childs.size(); ++i) {
                NodePtr child = node->childs[i].get();
                const size_t idx = i % dirichlet.size(); // index not goes outofbounds for child notes
                double val = getUCB(child, subRootVisitSqrt, ratio, dirichlet[idx], TProblem::UCT_C);
                if (best_val < val) {
                    best = child;
                    best_val = val;
                }
            }
            ratio = 1.0; // child nodes do not need dirichlet noise for priors
            node = best;
            visited_nodes.push_back(node);
            state.update(node->action);
        }

        //evaluate board win value also on terminating nodes
        state.computeMCTS_WP(idxAi, NULL, 0, NULL, W);

        return node;
    }

    //! Applies the backprop step of the Tree Search
    void backprop(std::vector<NodePtr>& visited_nodes, double W) const {
        // backprop results to visited nodes
        for (auto it = visited_nodes.begin(); it != visited_nodes.end(); ++it) {
            Node& node = *(*it);
            node.N++;
            node.W += W;
        }
    }

    //! Compute Dirichlet distribution
    void computeDirichlet(std::vector<double>& dirichlet) const {
        std::gamma_distribution<double> distribution(TProblem::DirichletAlpha);
        std::generate(std::begin(dirichlet), std::end(dirichlet), std::bind(distribution, generator));
        double sum = std::accumulate(std::begin(dirichlet), std::end(dirichlet), 0.0);
        std::transform (std::begin(dirichlet), std::end(dirichlet),
                        std::begin(dirichlet), std::bind2nd(std::divides<double>(), sum));
    }

    //! Returns the value of a node for tree search evaluation
    double getUCB(const NodePtr& _node, double subRootVisitSqrt, double ratio, double dnoise, double c) const {
        const Node& node = *_node;
        double p = ratio * node.P + (1.0-ratio) * dnoise;
        double n = static_cast<double>(node.N) + std::numeric_limits<double>::epsilon();
        double q = node.W / n;
        double u = p * (subRootVisitSqrt/(1+n));

        double val = q + c*u;
        return val;
    }

    ActType selectMoveDeterministic(NodePtr node) {
        NodePtr most_ptr = node; // init
        CountType most_visit = 0;
        for (auto it = node->childs.begin(); it != node->childs.end(); ++it) {
            NodePtr child = it->get();
            if (most_visit < child->N) {
                most_ptr = child;
                most_visit = child->N;
            }
        }
        return most_ptr->action;
    }

    ActType selectMoveStochastic(NodePtr node, double tau, std::vector<std::pair<ActType, double> >& piAction) {
        std::vector<double> pi;
        std::vector<NodePtr> childs;

        // collect and compute pi
        for (auto it = node->childs.begin(); it != node->childs.end(); ++it) {
            NodePtr child = it->get();
            pi.push_back(pow(child->N, 1.0/tau));
            childs.push_back(child);
        }

        // normalize
        double sum = std::accumulate(std::begin(pi), std::end(pi), 0.0);
        std::transform (std::begin(pi), std::end(pi),
                        std::begin(pi), std::bind2nd(std::divides<double>(), sum));

        // store pi/action for dataset
        for (size_t i = 0; i < pi.size(); ++i) {
            piAction.push_back(std::make_pair(childs[i]->action, pi[i]));
        }

        // select
        std::discrete_distribution<int> distribution(pi.begin(), pi.end());
        int select = distribution(generator);
        return childs[select]->action;
    }

public:
    //! Construct tree
    MCTS(unsigned int seed = 0) : generator(seed) {
        // artifical root, doesnt hold valid action
        root.reset(new Node(ActType()));
    }

    //! Execute a search on the current state for the ai, return the action
    ActType execute(int idxAi,
                    bool isDeterministic,
                    const TProblem& cstate,
                    unsigned int policyIter,
                    const std::vector<ActType>& history)
    {
        // walk tree according to history
        NodePtr subroot = catchup(cstate, history);

        { // make sure root is expanded before multithreaded execution
            double W = 0;
            TProblem state(cstate); // NOTE: copy of state is mandatory
            std::vector<NodePtr> policyNodes;

            // selection and expansion
            NodePtr node = policy(subroot, state, idxAi, policyNodes, W);

            // backpropagation of policy node
            backprop(policyNodes, W);

            // only one choice, dont think
            if (subroot->size() == 1 && isDeterministic) {
                NodePtr child = subroot->childs[0].get();
                return child->action;
            }
        }

        #pragma omp parallel for schedule(dynamic, 6)
        for (int i = 1; i < (int)policyIter; ++i) { // signed int to support omp2 for msvc
            double W = 0;
            TProblem state(cstate); // NOTE: copy of state is mandatory
            std::vector<NodePtr> policyNodes;

            // selection and expansion
            NodePtr node = policy(subroot, state, idxAi, policyNodes, W);

            // backpropagation of policy node
            backprop(policyNodes, W);
        }

        if (isDeterministic) {
            ActType action = selectMoveDeterministic(subroot);

            for (auto it = subroot->childs.begin(); it != subroot->childs.end(); ++it) {
                NodePtr child = it->get();
                std::cout << TProblem::act2str(child->action) << "; "
                          << "W: " << child->W << "; "
                          << "N: " << child->N << "; "
                          << "Q: " << child->W/child->N
                          << std::endl;
            }

            return action;

        } else { // stochastic
            double tau = 1.0;
            if (history.size()>60)
                tau = 0.05;
            std::vector<float> stateDNN;
            std::vector<float> policyDNN;
            std::vector<std::pair<ActType, double> > piAction;

            ActType action = selectMoveStochastic(subroot, tau, piAction);
            cstate.getGameStateDNN(stateDNN, idxAi);
            cstate.getPolicyTrainDNN(policyDNN, idxAi, piAction);
            cstate.storeGamePolicyDNN(stateDNN, policyDNN);

            for (size_t i = 0; i < subroot->childs.size(); ++i) {
                double pi = piAction[i].second;
                NodePtr child = subroot->childs[i].get();
                std::cout << TProblem::act2str(child->action) << "; "
                          << "Pi: " << pi << "; "
                          << "W: " << child->W << "; "
                          << "N: " << child->N << "; "
                          << "Q: " << child->W/child->N
                          << std::endl;
            }

            return action;
        }
    }

    template <typename T>
    void writeBranchNodes(unsigned int branch,
                          NodePtr parent,
                          NodePtr next,
                          int time,
                          float maxIter,
                          int opponent,
                          T& stream) {
        stream << (int)branch;
        stream << ";" << next->getNodeId();
        stream << ";" << parent->getNodeId();
        stream << ";" << time; //TODO:fix
        stream << ";" << TProblem::act2str(next->action);
        stream << ";" << opponent; //TODO:fix
        stream << ";" << "0"; // not selected
        stream << ";" << next->N / maxIter;
        stream << ";" << next->W / (float)next->N;
        stream << std::endl;


        for (auto it = next->childs.begin(); it != next->childs.end(); ++it) {
            NodePtr child = it->get();
            writeBranchNodes(branch+1, next, child, time, maxIter, opponent, stream);
        }
    }

    template <typename T>
    void writeResults(const TProblem& state, int idxAi, float maxIter, const std::vector<ActType>& history, T& stream) {
        stream << "Branch;ID;ParentID;Time;Actions;Opponent;Select;Visit;Win";
        stream << std::endl;
        stream << "0;0;0;0;ROOT;0;0;0;0";
        stream << std::endl;

        NodePtr parent = root.get();
        NodePtr child = root.get();
        for (size_t time = 0; time < history.size() ; ++time) {
            ActType act = history[time];
            int opponent = state.getPlayer(time) != idxAi ? 1 : 0;

            for (auto it = parent->childs.begin(); it != parent->childs.end(); ++it) {
                NodePtr next = it->get();

                if (next->action == act) {
                    child = next; // set child to selected node

                    stream << "0"; // not branch node, dont filter
                    stream << ";" << next->getNodeId();
                    stream << ";" << parent->getNodeId();
                    stream << ";" << time;
                    stream << ";" << TProblem::act2str(next->action);
                    stream << ";" << opponent;
                    stream << ";" << "1"; // selected
                    stream << ";" << next->N / maxIter;
                    stream << ";" << next->W / (float)next->N;
                    stream << std::endl;

                } else { // not selected nodes
                    writeBranchNodes(0, parent, next, time, maxIter, opponent, stream);
                }
            }

            parent = child;
        }
    }
};

#endif //MCTS_HPP
