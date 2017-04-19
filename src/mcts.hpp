#ifndef MCTS_HPP
#define MCTS_HPP

#include <vector>
#include <memory>
#include <limits>
#include <numeric>

#include "defs.hpp"
#include "mcts.cuh"
#include "hearts.hpp"

template <class TTree>
class MCTS : public TTree {
    typedef typename TTree::Node Node;
    typedef typename TTree::NodePtr NodePtr;
    typedef typename Node::CountType CountType;

private:

    struct Traverser {
        Hearts::State state;
        std::vector<NodePtr> childBuffer;
        std::vector<NodePtr> visitedNodes;

        Traverser() {
            childBuffer.reserve(52);
            visitedNodes.reserve(52);
        }

        void reset(const Hearts::State& _state) {
            state = _state; // NOTE: copy of state is mandatory
            visitedNodes.clear();
        }
    };

    std::vector<NodePtr> catchupNodes;
    std::vector<Traverser> traversers;

    NodePtr catchup(const Hearts::State& state) {
        catchupNodes.clear();
        NodePtr node = TTree::getRoot();
        catchupNodes.push_back(node);

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
            catchupNodes.push_back(node);
        }
        return node;
    }

    void expand(NodePtr& node,
                const Hearts::State& state,
                const Hearts::Player& player) {
        // create childs for node
        uint8 cards[52];
        node->expanded = 1;
        uint8 nCard = Hearts::getPossibleCards(state, player, cards);
        for (uint8 i = 0; i < nCard; ++i) {
            TTree::addNode(node, cards[i]);
        }
    }

    NodePtr policy(const NodePtr _node,
                   const Hearts::Player& player,
                   Traverser& t) {
        NodePtr node = _node;
        while (!t.state.isTerminal()) {
            // get childs that are not yet expanded
            t.childBuffer.clear();
            auto it = TTree::getChildIterator(node);
            while (it.hasNext()) {
                NodePtr child = it.next();
                if (child->expanded == 0) {
                    t.childBuffer.push_back(child);
                }
            }
            if (!t.childBuffer.empty()) {
                uint8 pick = static_cast<uint8>(rand() % t.childBuffer.size());
                node = t.childBuffer[pick];
                t.visitedNodes.push_back(node);
                return node;
            }
            else {
                // set node to best leaf
                double parentLogVisit = 2.0*log(static_cast<double>(node->visits));
                node = selectBestByValue(node, parentLogVisit, player.player != t.state.getCurrentPlayer());
                t.visitedNodes.push_back(node);
                Hearts::update(t.state, node->card);
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
        uint8 win = Hearts::mapPoints2Wins(player, points.data());
        for (auto it = visited_nodes.begin(); it != visited_nodes.end(); ++it) {
            Node& node = *(*it);
            node.visits += 1;
            node.wins[win] += 1;
        }
    }

    void backprop(std::vector<NodePtr>& visited_nodes, const unsigned int* wins, CountType count) {
        for (auto it = visited_nodes.begin(); it != visited_nodes.end(); ++it) {
            Node& node = *(*it);
            node.visits += count;
            for (uint8 i = 0; i < 28; ++i)
                node.wins[i] += wins[i];
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

    NodePtr selectBestByVisit(NodePtr node) {
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
        return most_ptr;
    }

    NodePtr selectBestByValue(NodePtr node, double parentLogVisit = 0.0, bool isOpponent = false) {
        NodePtr best_ptr = node; // init
        double best_val = -std::numeric_limits<double>::max();
        auto it = TTree::getChildIterator(node);
        while (it.hasNext()) {
            NodePtr child = it.next();
            double val = value(child, parentLogVisit, isOpponent);
            if (best_val < val) {
                best_ptr = child;
                best_val = val;
            }
        }
        return best_ptr;
    }

public:
    MCTS(unsigned int traverserCount = 1) {
        catchupNodes.reserve(52);
        traversers.resize(traverserCount);
    }

    uint8 execute(const Hearts::State& cstate,
                  const Hearts::Player& player,
                  unsigned int policyIter,
                  unsigned int rolloutIter,
                  RolloutContainerCPP* cuda_data) {

        // starting card
        if (cstate.isFirstRoundTurn() && player.hand[0] == player.player)
            return 0;

        // walk tree according to state
        NodePtr subroot = catchup(cstate);

        // expand subroot if needed
        if (subroot->expanded == 0) {
            expand(subroot, cstate, player);
        }
        { // check if only one choice is possible
            Traverser& t = traversers[0];
            t.childBuffer.clear();
            auto it = TTree::getChildIterator(subroot);
            while (it.hasNext()) {
                NodePtr child = it.next();
                t.childBuffer.push_back(child);
            }
            if (t.childBuffer.size() == 1) {
                return t.childBuffer[0]->card;
            }
        }

        if (cuda_data->hasGPU()) { // gpu
            std::vector<NodePtr> nodes;
            std::vector<Hearts::State*> states;
            nodes.reserve(traversers.size());
            states.reserve(traversers.size());
            for (unsigned int i = 0; i < policyIter; ++i) {
                for (size_t j = 0; j < traversers.size(); ++j) {
                    Traverser& t = traversers[j];
                    t.reset(cstate);
                    NodePtr node = policy(subroot, player, t);
                    if (!t.state.isTerminal())
                        Hearts::update(t.state, node->card);
                    nodes.push_back(node);
                    states.push_back(&t.state);
                }
                // rollout
                const unsigned int* wins = 0;//cuda_data->rollout(states, player);
                for (size_t j = 0; j < traversers.size(); ++j) {
                    NodePtr node = nodes[j];
                    Traverser& t = traversers[j];

                    const unsigned int* nodeWins = wins + j * 28;
                    expand(node, t.state, player);
                    backprop(catchupNodes, nodeWins, rolloutIter);
                    backprop(t.visitedNodes, nodeWins, rolloutIter);
                }
            }
        }
        else { // cpu
            std::array<uint8, 4> points;
            Traverser& t = traversers[0];
            for (unsigned int i = 0; i < policyIter; ++i) {
                t.reset(cstate);
                t.visitedNodes.push_back(subroot);
                // selection and expansion
                NodePtr node = policy(subroot, player, t);
                if (t.state.isTerminal())
                    break;
                Hearts::update(t.state, node->card);
                expand(node, t.state, player);
                // rollout
                for (unsigned int j = 0; j < rolloutIter; ++j) {
                    Hearts::State rstate(t.state);
                    rollout(rstate, player, points);
                    // backprop
                    //backprop(catchupNodes, player, points);
                    backprop(t.visitedNodes, player, points);
                }
            }
        }
        return selectBestByValue(subroot)->card;
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
