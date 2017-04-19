#ifndef MCTS_HPP
#define MCTS_HPP

#include <vector>
#include <memory>
#include <limits>
#include <numeric>

#include "defs.hpp"
#include "hearts.hpp"

template <class TTree>
class MCTS : public TTree {
    typedef typename TTree::Node Node;
    typedef typename TTree::NodePtr NodePtr;
    typedef typename Node::CountType CountType;

private:

    NodePtr catchup(const Hearts::State& state, std::vector<NodePtr>& visited_nodes) {
        NodePtr node = TTree::getRoot();
        visited_nodes.push_back(node);

        for (uint8 time = 0; time < 52; ++time) {
            if (state.getCardAtTime(time) == Hearts::State::order_unset)
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
        return node;
    }

    NodePtr policy(NodePtr node, Hearts::State& state, const Hearts::Player& player, std::vector<NodePtr>& visited_nodes) {
        uint8 cards[52];
        while (!state.isTerminal()) {
            // get cards
            uint8 nCards = Hearts::getPossibleCards(state, player, cards);
            if (nCards == 0) // invalid state, rollout will return false, increasing visit count
                return node;
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
                Hearts::update(state, card);
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
                    double val = value(child, logParentVisit, player.player != state.getCurrentPlayer());
                    if (best_val < val) {
                        best = child;
                        best_val = val;
                    }
                }
                node = best;
                visited_nodes.push_back(node);
                Hearts::update(state, node->card);
            }
        }
        return node;
    }

    bool rollout(Hearts::State& state, const Hearts::Player& player, std::array<uint8, 4>& points) {
        uint8 cards[52];
        points.fill(0);
        while (!state.isTerminal()) {
            uint8 nCards = Hearts::getPossibleCards(state, player, cards);
            if (nCards == 0) {
                std::cout << "i";
                return false; // invalid state
            }
            uint8 pick = static_cast<uint8>(rand() % nCards);
            Hearts::update(state, cards[pick]);
        }
        Hearts::computePoints(state, points);
        return true; // valid state
    }

    void backprop(std::vector<NodePtr>& visited_nodes, const Hearts::Player& player, std::array<uint8, 4>& points) {
        for (auto it = visited_nodes.begin(); it != visited_nodes.end(); ++it) {
            Node& node = *(*it);
            node.visits += 1;
            bool shotTheMoon = false;
            for (uint8 p = 0; p < 4; ++p) {
                if (points[p] == 26)
                    shotTheMoon = true;
            }
            if (shotTheMoon == true) {
                if (points[player.player] == 26) { // current ai shot the moon
                    node.wins[0] += 1;
                }
                else { // someone shot the moon
                    node.wins[27] += 1;
                }
            }
            else { // normal points (shifted with one)
                node.wins[points[player.player] + 1] += 1;
            }
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

    uint8 execute(const Hearts::State& cstate, const Hearts::Player& player, unsigned int policyIter, unsigned int rolloutIter) {
        std::array<uint8, 4> points;
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
            uint8 nCards = Hearts::getPossibleCards(cstate, player, cards);
            if (nCards == 1) {
                return cards[0];
            }
        }

        for (unsigned int i = 0; i < policyIter; ++i) {
            policy_nodes.clear();
            // NOTE: copy of state is mandatory
            Hearts::State state(cstate);
            // selection and expansion
            NodePtr node = policy(subroot, state, player, policy_nodes);
            // rollout
            for (unsigned int j = 0; j < rolloutIter; ++j) {
                Hearts::State rstate(state);
                rollout(rstate, player, points);
                // backprop
                backprop(policy_nodes, player, points);
                backprop(catchup_nodes, player, points);
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
};

#endif //MCTS_HPP
