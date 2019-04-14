#ifndef MCTS_HPP
#define MCTS_HPP

#include <vector>
#include <memory>
#include <limits>
#include <numeric>
#include <algorithm>

#include "mcts.cuh"
#include "mcts_debug.hpp"

//! Monte Carlo tree search to apply AI
/*!
 * \details This class implements Monte Carlo tree search as described in Wikipedia.
 *          It follows the policy-rollout-backprop idea.
 *          Before policy, it walks the tree according to the history of the state of the problem, this is called catchup.
 *          History must be handled by the main program.
 *          The tree search does not know the exact problem it solves.
 *          Interfacing with the problem is done by template interfaces.
 *
 *          The underlying data container is transparent for this class.
 *          A container must support the template interface required by the tree search.
 *          This is not documented, but the MCTreeDynamic class is the easiest to understand the interface.
 * \author adamp87
*/
template <class TTree, class TProblem, class TPolicyDebug>
class MCTS : public TTree {
    friend TPolicyDebug;
    typedef typename TTree::Node Node;
    typedef typename TTree::NodePtr NodePtr;
    typedef typename Node::CountType CountType;
    typedef typename Node::LockGuard LockGuard;
    typedef typename TProblem::MoveType MoveType;
    typedef typename TProblem::MoveCounterType MoveCounterType;

private:
    //! Walk the tree according to the history of the problem
    NodePtr catchup(const TProblem& state, const std::vector<MoveType>& history) {
        NodePtr node = TTree::getRoot();

        for (size_t time = 0; time < history.size(); ++time) {
            bool found = false;
            auto it = TTree::getChildIterator(node);
            while (it.hasNext()) {
                NodePtr child = it.next();
                if (child->move == history[time]) {
                    node = child;
                    found = true;
                    break;
                }
            }
            if (!found) { // no child, update tree according to history
                node = TTree::addNode(node, history[time]);
            }
        }
        return node;
    }

    //! Applies the policy step of the Tree Search
    NodePtr policy(const NodePtr subRoot, TProblem& state, int idxAi, std::vector<NodePtr>& visited_nodes) {
        NodePtr node = subRoot;
        MoveType moves[TProblem::MaxMoves];
        visited_nodes.push_back(node); // store subroot as policy
        while (!state.isFinished()) {
#if 1 // compute possible moves once and store in container
            { // thread-safe scope
                LockGuard guard(*node); (void)guard;
                if (node->nexts.isUnset()) { // get possible moves
                    MoveCounterType nMoves = state.getPossibleMoves(idxAi, state.getPlayer(), moves);
                    std::random_shuffle(moves, moves+nMoves);
                    node->nexts.init(moves, nMoves);
                    //node->nexts.test(moves, nMoves);
                }
                if (!node->nexts.isEmpty()) { // node is not fully expanded
                    MoveType move;
                    node->nexts.next(&move);
                    state.update(move);
                    // create child
                    node = TTree::addNode(node, move);
                    visited_nodes.push_back(node);
                    return node;
                }
            } // end of thread-safe scope

#else // compute possibleMoves every iter, remove duplicates
            MoveCounterType nMoves = state.getPossibleMoves(idxAi, state.getPlayer(), moves);

            { // thread-safe scope
                LockGuard guard(*node); (void)guard;
                auto it = TTree::getChildIterator(node);
                while (it.hasNext()) { // remove moves that already have been expanded
                    NodePtr child = it.next();
                    auto it = std::find(moves, moves + nMoves, child->move);
                    if (it != moves + nMoves) {
                        //moves.erase(it);
                        --nMoves;
                        *it = moves[nMoves];
                    }
                }
                if (nMoves != 0) { // node is not fully expanded
                    // expand
                    // select move and update
                    MoveCounterType pick = static_cast<MoveCounterType>(rand() % nMoves);
                    MoveType move = moves[pick];
                    state.update(move);
                    // create child
                    node = TTree::addNode(node, move);
                    visited_nodes.push_back(node);
                    return node;
                }
            } // end of thread-safe scope
#endif
            // node fully expanded
            // set node to best leaf
            NodePtr best = node; // init
            double best_val = -std::numeric_limits<double>::max();
            // compute subRootVisit for each iteration, multithread
            double subRootVisitLog = static_cast<double>(subRoot->visits);
            subRootVisitLog += std::numeric_limits<double>::epsilon(); // avoid log(0)
            subRootVisitLog = log(subRootVisitLog);
            auto it = TTree::getChildIterator(node);
            while (it.hasNext()) {
                NodePtr child = it.next();
                double val = value(child, subRootVisitLog, idxAi != state.getPlayer(), TProblem::UCT_C);
                if (best_val < val) {
                    best = child;
                    best_val = val;
                }
            }
            node = best;
            visited_nodes.push_back(node);
            state.update(node->move);
        }
        return node;
    }

    //! Applies the rollout step of the Tree Search
    bool rollout(TProblem& state, int idxAi, int maxRolloutDepth) const {
        int depth = 0; //if max is zero, until finished
        MoveType moves[TProblem::MaxMoves];
        while (!state.isFinished()) {
            if (++depth == maxRolloutDepth)
                break;
            MoveCounterType nMoves = state.getPossibleMoves(idxAi, state.getPlayer(), moves);
            MoveCounterType pick = static_cast<MoveCounterType>(rand() % nMoves);
            state.update(moves[pick]);
        }
        return true;
    }

    //! Applies the backprop step of the Tree Search
    void backprop(std::vector<NodePtr>& visited_nodes, double win) const {
        // backprop results to visited nodes
        for (auto it = visited_nodes.begin(); it != visited_nodes.end(); ++it) {
            Node& node = *(*it);
            node.visits++;
            node.wins += win;
        }
    }

    //! Returns the value of a node for tree search evaluation
    double value(const NodePtr& _node, double subRootVisitLog, bool isOpponent, double c) const {
        const Node& node = *_node;
        double q = node.wins;
        double n = static_cast<double>(node.visits);
        n += std::numeric_limits<double>::epsilon(); //avoid div by 0

        if (isOpponent)
            q = n-q; // opponent is trying to minimalize win
        double val = q / n + c * sqrt(subRootVisitLog/n);
        return val;
    }

    MoveType selectBestByVisit(NodePtr node) {
        NodePtr most_ptr = node; // init
        CountType most_visit = 0;
        auto it = TTree::getChildIterator(node);
        while (it.hasNext()) {
            NodePtr child = it.next();
            if (most_visit < child->visits) {
                most_ptr = child;
                most_visit = child->visits;
            }
        }
        return most_ptr->move;
    }

    MoveType selectBestByValue(NodePtr node) {
        NodePtr best_ptr = node; // init
        double best_val = -std::numeric_limits<double>::max();
        double nodeVisitLog = log(static_cast<double>(node->visits));
        auto it = TTree::getChildIterator(node);
        while (it.hasNext()) {
            NodePtr child = it.next();
            double val = value(child, nodeVisitLog, false, 0.0);
            if (best_val < val) {
                best_ptr = child;
                best_val = val;
            }
        }
        return best_ptr->move;
        // NOTE: use lower exploration rate for value computation to
        //       avoid selection of a node, which has a low visit count
        // NOTE: other solution to use constant zero, win ratio decides
    }

public:
    MCTS() { }

    //! Execute a search on the current state for the ai, return the move
    MoveType execute(int idxAi,
                     int maxRolloutDepth,
                     const TProblem& cstate,
                     unsigned int policyIter,
                     unsigned int rolloutIter,
                     const std::vector<MoveType>& history,
                     RolloutCUDA<TProblem>* rollloutCuda,
                     TPolicyDebug& policyDebug)
    {
        // walk tree according to history
        NodePtr subroot = catchup(cstate, history);
        { // only one choice, dont think
            MoveType moves[TProblem::MaxMoves];
            MoveCounterType nMoves = cstate.getPossibleMoves(idxAi, idxAi, moves);
            if (nMoves == 1) {
                return moves[0];
            }
        }

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < (int)policyIter; ++i) { // signed int to support omp2 for msvc
            double wins = 0;
            TProblem state(cstate); // NOTE: copy of state is mandatory
            std::vector<NodePtr> policyNodes;

            // selection and expansion
            NodePtr node = policy(subroot, state, idxAi, policyNodes);
            policyDebug.push(*this, cstate, policyNodes, subroot, idxAi, i, history.size());

            // rollout and backprop
            if (rollloutCuda->hasGPU() && rolloutIter != 1 &&
                rollloutCuda->cuRollout(idxAi, maxRolloutDepth, state, rolloutIter, wins))
            { // rollout on gpu (if has gpu, is requested and gpu is free)
                // NOTE: backprop one-by-one to decrease inconsistent increment in multithreading
                wins /= rolloutIter; // cudaRollout returns the sum of wins
                for (unsigned int j = 0; j < rolloutIter; ++j)
                    backprop(policyNodes, wins);
            } else { // rollout on cpu (fallback)
                for (unsigned int j = 0; j < rolloutIter; ++j) {
                    TProblem rstate(state); // NOTE: copy of state is mandatory
                    rollout(rstate, idxAi, maxRolloutDepth);
                    // backprop
                    wins = rstate.computeMCTSWin(idxAi);
                    backprop(policyNodes, wins);
                }
            }
        }
        return selectBestByValue(subroot);
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
        stream << ";" << TTree::getNodeId(next);
        stream << ";" << TTree::getNodeId(parent);
        stream << ";" << time; //TODO:fix
        stream << ";" << TProblem::move2str(next->move);
        stream << ";" << opponent; //TODO:fix
        stream << ";" << "0"; // not selected
        stream << ";" << next->visits / maxIter;
        stream << ";" << next->wins / (float)next->visits;
        stream << std::endl;

        auto it = TTree::getChildIterator(next);
        while (it.hasNext()) {
            NodePtr child = it.next();
            writeBranchNodes(branch+1, next, child, time, maxIter, opponent, stream);
        }
    }

    template <typename T>
    void writeResults(const TProblem& state, int idxAi, float maxIter, const std::vector<MoveType>& history, T& stream) {
        stream << "Branch;ID;ParentID;Time;Move;Opponent;Select;Visit;Win";
        stream << std::endl;
        stream << "0;0;0;0;ROOT;0;0;0;0";
        stream << std::endl;

        NodePtr parent = TTree::getRoot();
        NodePtr child = TTree::getRoot();
        for (size_t time = 0; time < history.size() ; ++time) {
            MoveType move = history[time];
            int opponent = state.getPlayer(time) != idxAi ? 1 : 0;

            auto it = TTree::getChildIterator(parent);
            while (it.hasNext()) {
                NodePtr next = it.next();

                if (next->move == move) {
                    child = next; // set child to selected node

                    stream << "0"; // not branch node, dont filter
                    stream << ";" << TTree::getNodeId(next);
                    stream << ";" << TTree::getNodeId(parent);
                    stream << ";" << time;
                    stream << ";" << TProblem::move2str(next->move);
                    stream << ";" << opponent;
                    stream << ";" << "1"; // selected
                    stream << ";" << next->visits / maxIter;
                    stream << ";" << next->wins / (float)next->visits;
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
