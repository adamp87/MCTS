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

template <class TTree>
class MCTS : public TTree {
    typedef typename TTree::Node Node;
    typedef typename TTree::NodePtr NodePtr;
    typedef typename Node::CountType CountType;

private:

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

    NodePtr policy(NodePtr node, Hearts& state, const Hearts::Player& player, std::vector<NodePtr>& visited_nodes) {
        uint8 cards[52];
        visited_nodes.push_back(node); // store subroot as policy
        while (!state.isTerminal()) {
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
                double logParentVisit = 2.0*log(static_cast<double>(node->visits));
                auto it = TTree::getChildIterator(node);
                while (it.hasNext()) {
                    NodePtr child = it.next();
                    double val = value(child, logParentVisit, player.player != state.getPlayer());
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

    bool rollout(Hearts& state, const Hearts::Player& player) {
        uint8 cards[52];
        while (!state.isTerminal()) {
            uint8 nCards = state.getPossibleCards(player, cards);
            debug_invalidState(nCards);
            uint8 pick = static_cast<uint8>(rand() % nCards);
            state.update(cards[pick]);
        }
        return true;
    }

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

    double value(const NodePtr& _node, double logParentVisits, bool isOpponent) const {
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
            q *= -1; // opponent is trying to maximalize points
        double n = static_cast<double>(node.visits);
        double val = q / n + c*sqrt(logParentVisits / n);
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

    uint8 execute(const Hearts& cstate, const Hearts::Player& player, unsigned int policyIter, unsigned int rolloutIter) {
        std::vector<NodePtr> policy_nodes;
        std::vector<NodePtr> catchup_nodes;
        policy_nodes.reserve(52);
        catchup_nodes.reserve(52);

        // starting card
        if (cstate.isFirstRoundTurn() && player.hand[0] == player.player)
            return 0;

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
    void printNodeWithChilds(NodePtr _node, uint8 shift, T& stream) {
        Node& node = *_node;
        for (uint8 i = 0; i < shift; ++i) {
            stream << " ";
        }
        auto idx = TTree::getNodeId(_node);
        stream << "<" << idx << ", " << int(node.card / 13) << ":" << int(node.card % 13) << ", Visits: " << int(node.visits) << ", Wins: ";
        for (uint8 i = 0; i < 28; ++i) {
            stream << int(node.wins[i]) << ", ";
        }
        stream << ">" << std::endl;

        auto it = TTree::getChildIterator(_node);
        while (it.hasNext()) {
            NodePtr child = it.next();
            printNodeWithChilds(child, shift+1, stream);
        }

        for (uint8 i = 0; i < shift; ++i) {
            stream << " ";
        }
        stream << "</" << idx << ">" << std::endl;
    }

    template <typename T>
    void writeResults(const Hearts& state, const Hearts::Player& player, float maxIter, T& stream) {

        stream << "Round;Card;Next;Conf;P-26";
        for (int i = 0; i < 27; ++i)
            stream << ";P" << i;
        stream << std::endl;

        uint8 round = 0;
        NodePtr parent = TTree::getRoot();
        NodePtr child = TTree::getRoot();
        for (uint8 time = 0; time < 52; ++time) {
            uint8 card = state.getCardAtTime(time);

            auto it = TTree::getChildIterator(parent);
            while (it.hasNext()) {
                NodePtr next = it.next();
                if (next->card == card) {
                    child = next;
                    break;
                }
            }

            if (player.hand[card] == player.player) {
                auto itt = TTree::getChildIterator(parent);
                while (itt.hasNext()) {
                    NodePtr next = itt.next();

                    stream << (int)round;
                    stream << ";" << (int)next->card;
                    stream << ";" << (next->card == child->card) ? int(1) : int(0);
                    stream << ";" << next->visits / maxIter;
                    for (uint8 i = 0; i < 28; ++i) {
                        stream << ";" << next->wins[i] / (float)next->visits;
                    }
                    stream << std::endl;
                }
                ++round;
            }

            parent = child;
        }
    }
};

#endif //MCTS_HPP
