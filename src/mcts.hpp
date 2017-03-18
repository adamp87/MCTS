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

    NodePtr pick(const std::vector<NodePtr>& expandables) {
        // pick next node
        uint8 pick = static_cast<uint8>(rand() % expandables.size());
        NodePtr node = expandables[pick];
        return node;
    }

    NodePtr expand(NodePtr& node,
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

    bool policy(NodePtr node,
                   Hearts::State& state,
                   const Hearts::Player& player,
                   std::vector<NodePtr>& visited_nodes,
                   std::vector<NodePtr>& expandables) {
        while (!state.isTerminal()) {
            // get childs that are not yet expanded
            expandables.clear();
            auto it = TTree::getChildIterator(node);
            while (it.hasNext()) {
                NodePtr child = it.next();
                if (child->expanded == 0) {
                    expandables.push_back(child);
                }
            }
            if (!expandables.empty()) {
                return true;
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
        return false;
    }

    bool rollout(Hearts::State& state, const Hearts::Player& player, std::array<uint8, 4>& points, std::vector<uint8>& cards) {
        points.fill(0);
        while (!state.isTerminal()) {
            Hearts::getPossibleCards(state, player, cards);
            if (cards.empty()) {
                std::cout << "i";
                return false; // invalid state
            }
            uint8 pick = static_cast<uint8>(rand() % cards.size());
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

    uint8 execute(const Hearts::State& cstate,
                  const Hearts::Player& player,
                  unsigned int policyIter,
                  unsigned int rolloutIter,
                  RolloutContainerCPP* cuda_data) {
        std::array<uint8, 4> points;
        std::vector<uint8> cards_buffer;
        std::vector<NodePtr> policy_nodes;
        std::vector<NodePtr> catchup_nodes;
        std::vector<NodePtr> expandable_nodes;
        cards_buffer.reserve(52);
        policy_nodes.reserve(52);
        catchup_nodes.reserve(52);
        expandable_nodes.reserve(52);
        // walk tree according to state
        NodePtr subroot = catchup(cstate, catchup_nodes);
        Hearts::getPossibleCards(cstate, player, cards_buffer);
        if (cards_buffer.size() == 1) {
            return cards_buffer[0]; // only one choice, dont think
        }
        if (subroot->expanded == 0) {
            subroot->expanded = 1;
            for (uint8 card : cards_buffer) {
                TTree::addNode(subroot, card);
            }
        }

        if (cuda_data->hasGPU()) { // gpu
            for (unsigned int i = 0; i < policyIter; ++i) {
                policy_nodes.clear();
                // NOTE: copy of state is mandatory
                Hearts::State state(cstate);
                // selection and expansion
                if(!policy(subroot, state, player, policy_nodes, expandable_nodes))
                    break;
                // rollout
                const unsigned int* wins = cuda_data->rollout(state, player, expandable_nodes);
                //write back results and expand all nodes
                for (uint8 j = 0; j < expandable_nodes.size(); ++j) {
                    NodePtr& node = expandable_nodes[j];
                    const unsigned int* nodeWins = wins + j * 28;
                    policy_nodes.push_back(node);
                    backprop(policy_nodes, nodeWins, rolloutIter);
                    backprop(catchup_nodes, nodeWins, rolloutIter);
                    policy_nodes.pop_back();
                    Hearts::State estate(state);
                    Hearts::update(estate, node->card);
                    expand(node, estate, player);
                }
            }
        }
        else { // cpu
            for (unsigned int i = 0; i < policyIter; ++i) {
                policy_nodes.clear();
                // NOTE: copy of state is mandatory
                Hearts::State state(cstate);
                // selection and expansion
                if(!policy(subroot, state, player, policy_nodes, expandable_nodes))
                    break;
                NodePtr node = pick(expandable_nodes);
                Hearts::update(state, node->card);
                expand(node, state, player);
                policy_nodes.push_back(node);
                // rollout
                for (unsigned int j = 0; j < rolloutIter; ++j) {
                    Hearts::State rstate(state);
                    rollout(rstate, player, points, cards_buffer);
                    // backprop
                    backprop(policy_nodes, player, points);
                    backprop(catchup_nodes, player, points);
                }
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
