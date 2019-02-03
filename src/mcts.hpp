#ifndef MCTS_HPP
#define MCTS_HPP

#include <vector>
#include <memory>
#include <limits>
#include <numeric>

#include "defs.hpp"
#include "mcts.cuh"
#include "hearts.hpp"

bool debug_invalidState(uint8 nCards) {
#if 0
    if (nCards != 0)
        return false;

    std::cout << "i";
    return true; // invalid state
#else
    return false;
#endif
}

//! Monte Carlo tree search to apply AI
/*!
 * \details This class implements Monte Carlo tree search as described in Wikipedia.
 *          It follows the policy-rollout-backprop idea.
 *          Before policy, it walks the tree according to the current state of the game, this is called catchup.
 *          The game Hearts does not a have a single win/lose outcome, it has a range of points that need to be avoided.
 *          To get a win value for node evaluation, the number wins/visits are normalized and summed.
 *          This mechanism is implemented in this class.
 *          To have a generic tree search class, this should de detached in the future.
 *
 *          The underlying data container is transparent for this class.
 *          A container must support the template interface required by the tree search.
 *          This is not documented, but the MCTreeDynamic class is the easiest to understand the interface.
 * \author adamp87
*/
template <class TTree>
class MCTS : public TTree {
    typedef typename TTree::Node Node;
    typedef typename TTree::NodePtr NodePtr;
    typedef typename Node::CountType CountType;
    typedef typename Node::LockGuard LockGuard;

private:
    //! Walk the tree according to the state of the game
    NodePtr catchup(const Hearts& state, std::vector<NodePtr>& visited_nodes) {
        NodePtr node = TTree::getRoot();
        visited_nodes.push_back(node);

        for (uint8 time = 0; time < 52; ++time) {
            if (!state.isCardAtTimeValid(time))
                break;

            bool found = false;
            auto it = TTree::getChildIterator(node);
            while (it.hasNext()) {
                NodePtr child = it.next();
                if (child->card == state.getCardAtTime(time)) {
                    node = child;
                    found = true;
                    break;
                }
            }
            if (!found) { // no child, update tree according to state
                node = TTree::addNode(node, state.getCardAtTime(time));
            }
            visited_nodes.push_back(node);
        }
        visited_nodes.pop_back(); // remove subroot from catchup
        return node;
    }

    //! Applies the policy step of the Tree Search
    NodePtr policy(NodePtr subRoot, Hearts& state, const Hearts::Player& player, std::vector<NodePtr>& visited_nodes) {
        uint8 cards[52];
        NodePtr node = subRoot;
        visited_nodes.push_back(node); // store subroot as policy
        while (!state.isGameOver()) {
            // get cards
            uint8 nCards = state.getPossibleCards(player, cards);
            if (debug_invalidState(nCards))
                return node; // invalid state, rollout will return false, increasing visit count

            { // thread-safe scope
                LockGuard guard(*node); (void)guard;
                auto it = TTree::getChildIterator(node);
                while (it.hasNext()) { // remove cards that already have been expanded
                    NodePtr child = it.next();
                    auto it = std::find(cards, cards + nCards, child->card);
                    if (it != cards + nCards) {
                        //cards.erase(it);
                        --nCards;
                        *it = cards[nCards];
                    }
                }
                if (nCards != 0) { // node is not fully expanded
                    // expand
                    // select card and update
                    uint8 pick = static_cast<uint8>(rand() % nCards);
                    uint8 card = cards[pick];
                    state.update(card);
                    // create child
                    node = TTree::addNode(node, card);
                    visited_nodes.push_back(node);
                    return node;
                }
            } // end of thread-safe scope

            // node fully expanded
            // set node to best leaf
            // compute subRootVisit for each iteration, multithread
            NodePtr best = node; // init
            double best_val = -std::numeric_limits<double>::max();
            double subRootVisitLog = log(static_cast<double>(subRoot->visits));
            auto it = TTree::getChildIterator(node);
            while (it.hasNext()) {
                NodePtr child = it.next();
                double val = value(child, subRootVisitLog, player.player != state.getPlayer());
                if (best_val < val) {
                    best = child;
                    best_val = val;
                }
            }
            node = best;
            visited_nodes.push_back(node);
            state.update(node->card);
        }
        return node;
    }

    //! Applies the rollout step of the Tree Search
    bool rollout(Hearts& state, const Hearts::Player& player) const {
        uint8 cards[52];
        while (!state.isGameOver()) {
            uint8 nCards = state.getPossibleCards(player, cards);
            debug_invalidState(nCards);
            uint8 pick = static_cast<uint8>(rand() % nCards);
            state.update(cards[pick]);
        }
        return true;
    }

    //! Applies the backprop step of the Tree Search
    void backprop(std::vector<NodePtr>& visited_nodes, const Hearts::Player& player, uint8* points) const {
        // get idx, where winning have to be increased
        size_t winIdx = points[player.player] + 1; // normal points (shifted with one)
        for (uint8 p = 0; p < 4; ++p) {
            if (points[p] == 26) {
                if (p == player.player)
                    winIdx = 0; // current ai shot the moon
                else
                    winIdx = 27; // other ai shot the moon
            }
        }

        // backprop results to visited nodes
        for (auto it = visited_nodes.begin(); it != visited_nodes.end(); ++it) {
            Node& node = *(*it);
            node.visits += 1;
            node.wins[winIdx] += 1;
        }
    }

    //! Returns the value of a node for tree search evaluation
    double value(const NodePtr& _node, double subRootVisitLog, bool isOpponent, double c = 2.0) const {
        //weight = (np.exp(np.linspace(1, 0, 28))-1)/(np.exp(1)-1)
        const double weight[28] = { // normalize win -> value between 1..0
          1.        , 0.94248003, 0.88705146, 0.83363825, 0.78216712,
          0.73256745, 0.68477121, 0.63871282, 0.59432909, 0.55155914,
          0.51034428, 0.47062797, 0.43235573, 0.39547506, 0.35993534,
          0.32568784, 0.29268556, 0.26088323, 0.23023722, 0.20070548,
          0.1722475 , 0.14482425, 0.11839809, 0.09293277, 0.06839336,
          0.04474619, 0.02195883, 0.        };

        double q = 0.0;
        const Node& node = *_node;
        double n = static_cast<double>(node.visits);
        for (uint8 i = 0; i < 28; ++i) {
            q += weight[i] * node.wins[i];
        }

        if (isOpponent)
            q = n-q; // opponent is trying to maximalize points
        double val = q / n + sqrt(c*subRootVisitLog/n);
        return val;
    }

    uint8 selectBestByVisit(NodePtr node) {
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
        return most_ptr->card;
    }

    uint8 selectBestByValue(NodePtr node) {
        NodePtr best_ptr = node; // init
        double best_val = -std::numeric_limits<double>::max();
        double nodeVisitLog = log(static_cast<double>(node->visits));
        auto it = TTree::getChildIterator(node);
        while (it.hasNext()) {
            NodePtr child = it.next();
            double val = value(child, nodeVisitLog, false, 0.5);
            if (best_val < val) {
                best_ptr = child;
                best_val = val;
            }
        }
        return best_ptr->card;
        // NOTE: use lower exploration rate for value computation to
        //       avoid selection of a node, which has a low visit count
    }

public:
    MCTS() { }

    //! Execute a search on the current state for the player, return the cards to play
    uint8 execute(const Hearts& cstate,
                  const Hearts::Player& player,
                  unsigned int policyIter,
                  unsigned int rolloutIter,
                  RolloutCUDA* rollloutCuda)
    {
        std::vector<NodePtr> catchup_nodes;
        catchup_nodes.reserve(52);

        // walk tree according to state
        NodePtr subroot = catchup(cstate, catchup_nodes);
        { // only one choice, dont think
            uint8 cards[52];
            uint8 nCards = cstate.getPossibleCards(player, cards);
            if (nCards == 1) {
                return cards[0];
            }
        }

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < (int)policyIter; ++i) { // signed int to support omp2 for msvc
            std::vector<NodePtr> policy_nodes;
            std::vector<uint8> points(4);
            policy_nodes.reserve(52);
            // NOTE: copy of state is mandatory
            Hearts state(cstate);
            // selection and expansion
            NodePtr node = policy(subroot, state, player, policy_nodes);
            if (rollloutCuda->hasGPU() && rolloutIter != 1 &&
                rollloutCuda->cuRollout(state, player, rolloutIter, points))
            { // rollout on gpu (if has gpu, is requested and gpu is free)
                for (unsigned int j = 0; j < rolloutIter; ++j) {
                    backprop(policy_nodes, player, points.data() + j * 4);
                }
            } else { // rollout on cpu (fallback)
                for (unsigned int j = 0; j < rolloutIter; ++j) {
                    Hearts rstate(state);
                    rollout(rstate, player);
                    // backprop
                    rstate.computePoints(points.data());
                    backprop(policy_nodes, player, points.data());
                    //backprop(catchup_nodes, player, points.data());
                }
            }
        }
        return selectBestByValue(subroot);
    }

    template <typename T>
    void writeBranchNodes(uint8 branch, NodePtr parent, NodePtr next, uint8 round, float maxIter, uint8 opponent, T& stream) {
        stream << (int)branch;
        stream << ";" << TTree::getNodeId(next);
        stream << ";" << TTree::getNodeId(parent);
        stream << ";" << (int)round;
        stream << ";" << (int)next->card;
        stream << ";" << (int)opponent;
        stream << ";" << "0"; // not selected
        stream << ";" << next->visits / maxIter;
        for (uint8 i = 0; i < 28; ++i) {
            stream << ";" << next->wins[i] / (float)next->visits;
        }
        stream << std::endl;

        auto it = TTree::getChildIterator(next);
        while (it.hasNext()) {
            NodePtr child = it.next();
            writeBranchNodes(branch+1, next, child, round, maxIter, opponent, stream);
        }
    }

    template <typename T>
    void writeResults(const Hearts& state, uint8 playerID, float maxIter, T& stream) {
        stream << "Branch;ID;ParentID;Round;Card;Opponent;Select;Conf;PM";
        for (int i = 0; i < 27; ++i)
            stream << ";P" << i;
        stream << std::endl;

        uint8 round = 0;
        NodePtr parent = TTree::getRoot();
        NodePtr child = TTree::getRoot();
        for (uint8 time = 0; time < 52; ++time) {
            round = time / 4;
            uint8 card = state.getCardAtTime(time);
            uint8 opponent = state.getPlayer(time) != playerID ? 1 : 0;

            auto it = TTree::getChildIterator(parent);
            while (it.hasNext()) {
                NodePtr next = it.next();

                if (next->card == card) {
                    child = next; // set child to selected node

                    stream << "0"; // not branch node, dont filter
                    stream << ";" << TTree::getNodeId(next);
                    stream << ";" << TTree::getNodeId(parent);
                    stream << ";" << (int)round;
                    stream << ";" << (int)next->card;
                    stream << ";" << (int)opponent;
                    stream << ";" << "1"; // selected
                    stream << ";" << next->visits / maxIter;
                    for (uint8 i = 0; i < 28; ++i) {
                        stream << ";" << next->wins[i] / (float)next->visits;
                    }
                    stream << std::endl;

                } else { // not selected nodes
                    writeBranchNodes(0, parent, next, round, maxIter, opponent, stream);
                }
            }

            parent = child;
        }
    }
};

#endif //MCTS_HPP
