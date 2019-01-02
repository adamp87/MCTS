#ifndef MCTS_HPP
#define MCTS_HPP

#include <vector>
#include <memory>
#include <limits>
#include <numeric>

#include "defs.hpp"
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
 *          To get a win value for node evaluation, the number wins/points are weighted by a distribution and summed.
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
    NodePtr policy(NodePtr node, Hearts& state, const Hearts::Player& player, std::vector<NodePtr>& visited_nodes) {
        uint8 cards[52];
        visited_nodes.push_back(node); // store subroot as policy
        double subRootVisitLog = log(static_cast<double>(node->visits));
        while (!state.isGameOver()) {
            // get cards
            uint8 nCards = state.getPossibleCards(player, cards);
            if (debug_invalidState(nCards))
                return node; // invalid state, rollout will return false, increasing visit count
            // remove cards that already have been expanded
            auto it = TTree::getChildIterator(node);
            while (it.hasNext()) {
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
            else {
                // set node to best leaf
                NodePtr best = node; // init
                double best_val = -std::numeric_limits<double>::max();
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
        }
        return node;
    }

    //! Applies the rollout step of the Tree Search
    bool rollout(Hearts& state, const Hearts::Player& player) {
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
    void backprop(std::vector<NodePtr>& visited_nodes, const Hearts::Player& player, uint8* points) {
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
    double value(const NodePtr& _node, double subRootVisitLog, bool isOpponent) const {
        const double c = 1.4142135623730950488016887242097; //sqrt(2)
        const double distribution[28] = { //exponential distibution from -1 to +1
            0.112289418368,0.103975630952,0.0962773873887,0.0891491134753,
            0.0825486092736,0.0764367992834,0.0707775011124,0.0655372112737,
            0.0606849068422,0.0561918617969,0.0520314769604,0.0481791225296,
            0.0446119922655,0.0413089684786,-0.0413089684786,-0.0446119922655,
            -0.0481791225296,-0.0520314769604,-0.0561918617969,-0.0606849068422,
            -0.0655372112737,-0.0707775011124,-0.0764367992834,-0.0825486092736,
            -0.0891491134753,-0.0962773873887,-0.103975630952,-0.112289418368 };

        double q = 0.0;
        const Node& node = *_node;
        for (uint8 i = 0; i < 28; ++i) {
            q += distribution[i] * node.wins[i];
        }

        if (isOpponent)
            q = -q; // opponent is trying to maximalize points
        double n = static_cast<double>(node.visits);
        double val = q / n + c*sqrt(subRootVisitLog / n);
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
        auto it = TTree::getChildIterator(node);
        while (it.hasNext()) {
            NodePtr child = it.next();
            double val = value(child, 0.0, false);
            if (best_val < val) {
                best_ptr = child;
                best_val = val;
            }
        }
        return best_ptr->card;
    }

public:
    MCTS() { }

    //! Execute a search on the current state for the player, return the cards to play
    uint8 execute(const Hearts& cstate, const Hearts::Player& player, unsigned int policyIter, unsigned int rolloutIter) {
        std::vector<NodePtr> policy_nodes;
        std::vector<NodePtr> catchup_nodes;
        policy_nodes.reserve(52);
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

        for (unsigned int i = 0; i < policyIter; ++i) {
            policy_nodes.clear();
            // NOTE: copy of state is mandatory
            Hearts state(cstate);
            // selection and expansion
            NodePtr node = policy(subroot, state, player, policy_nodes);
            // rollout
            for (unsigned int j = 0; j < rolloutIter; ++j) {
                Hearts rstate(state);
                rollout(rstate, player);
                // backprop
                uint8 points[4];
                rstate.computePoints(points);
                backprop(policy_nodes, player, points);
                //backprop(catchup_nodes, player, points);
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
