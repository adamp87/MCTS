#ifndef MCTS_HPP
#define MCTS_HPP

#include <vector>
#include <memory>
#include <limits>
#include <numeric>

#include "defs.hpp"
#include "hearts.hpp"

class MCTS {
    typedef std::uint32_t IdxType;
    typedef std::uint32_t CountType;
private:
    struct MCTSNodeDisk {
        CountType visits; // node visit count
        CountType wins[28]; // number of results for each points

        IdxType childs[39]; // child indices

        std::uint8_t card; // card played out
    };

    std::vector<MCTSNodeDisk> nodes;

    IdxType catchup(const Hearts::State& state, const Hearts::Player& player, std::vector<IdxType>& visited_nodes) {
        uint8 order = 0;
        IdxType idx = 0;

        while (order != 52 && state.getCardAtTime(order) != Hearts::State::order_unset) {
            MCTSNodeDisk& node = nodes[idx];
            IdxType child_idx = 0;

            // find next child
            for (IdxType next_idx : node.childs) {
                if (next_idx == 0)
                    continue;
                MCTSNodeDisk& next = nodes[next_idx];
                if (next.card == state.getCardAtTime(order)) {
                    child_idx = next_idx;
                    break;
                }
            }

            // no child, update tree according to state
            if (child_idx == 0) {
                child_idx = addNode(node, state.getCardAtTime(order));
            }

            visited_nodes.push_back(child_idx);
            idx = child_idx;
            ++order;
        }
        return idx;
    }

    IdxType policy(IdxType node_idx, Hearts::State& state, const Hearts::Player& player, std::vector<IdxType>& visited_nodes, std::vector<uint8>& cards) {
        //while (!isNodeTerminal(nodes[node_idx])) {
        while (!state.isTerminal()) {
            MCTSNodeDisk& node = nodes[node_idx];
            // get cards
            Hearts::getPossibleCards(state, player, cards);
            if (cards.empty()) // invalid state, rollout will return false, increasing visit count
                return node_idx;
            // remove cards that already have been expanded
            for (IdxType child_idx : node.childs) {
                if (child_idx == 0)
                    continue;
                MCTSNodeDisk& child = nodes[child_idx];
                auto it = std::find(cards.begin(), cards.end(), child.card);
                if (it != cards.end()) {
                    //cards.erase(it);
                    *it = cards.back();
                    cards.pop_back();
                }
            }
            if (!cards.empty()) { // node is not fully expanded
                // expand
                // select card and update
                uint8 pick = static_cast<uint8>(rand() % cards.size());
                uint8 card = cards[pick];
                Hearts::update(state, card);
                // create child
                IdxType child_idx = addNode(node, card);
                visited_nodes.push_back(child_idx);
                return child_idx;
            }
            else {
                // set node to best leaf
                double best_val = -std::numeric_limits<double>::max();
                IdxType best_idx = 0;
                double logParentVisit = 2.0*log(static_cast<double>(node.visits));
                for (IdxType child_idx : node.childs) {
                    if (child_idx == 0)
                        continue;
                    MCTSNodeDisk& child = nodes[child_idx];
                    double val = value(child, logParentVisit, player.player != state.getCurrentPlayer());
                    if (best_val < val) {
                        best_idx = child_idx;
                        best_val = val;
                    }
                }
                node_idx = best_idx;
                visited_nodes.push_back(node_idx);
                MCTSNodeDisk& child = nodes[node_idx];
                Hearts::update(state, child.card);
            }
        }
        return node_idx;
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

    void backprop(std::vector<IdxType>& node_indices, const Hearts::Player& player, std::array<uint8, 4>& points) {
        for (auto it = node_indices.begin(); it != node_indices.end(); ++it) {
            IdxType node_idx = *it;
            MCTSNodeDisk& node = nodes[node_idx];
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

    double value(const MCTSNodeDisk& node, double logParentVisits, bool isOpponent) const {
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
        for (uint8 i = 0; i < 28; ++i) {
            q += distribution[i] * node.wins[i];
        }

        if (isOpponent)
            q *= -1; // opponent is trying to maximalize points
        double n = static_cast<double>(node.visits);
        double val = q / n + c*sqrt(logParentVisits / n);
        return val;
    }

    IdxType addNode(MCTSNodeDisk& parent, uint8 card) {
        // init leaf node
        MCTSNodeDisk child;
        child.visits = 1;
        child.card = card;
        for (uint8 i = 0; i < 39; ++i) {
            child.childs[i] = 0;
        }
        for (uint8 i = 0; i < 28; ++i) {
            child.wins[i] = 0;
        }

        // find next free idx of parent to put child
        uint8 idx = 0;
        for (; idx < 39; ++idx) {
            if (parent.childs[idx] == 0) {
                break;
            }
        }
        if (idx == 39)
            std::cerr << "Error in child saving" << std::endl;

        // add leaf to vector and init space for childs
        IdxType child_idx = static_cast<IdxType>(nodes.size());
        parent.childs[idx] = child_idx;
        nodes.push_back(child);
        return child_idx;
    }

    uint8 selectBestByVisit(IdxType node_idx) {
        MCTSNodeDisk& node = nodes[node_idx];

        IdxType most_idx = 0;
        CountType most_visit = 0;
        for (IdxType child_idx : node.childs) {
            if (child_idx == 0)
                continue;
            MCTSNodeDisk& child = nodes[child_idx];
            if (most_visit < child.visits) {
                most_idx = child_idx;
                most_visit = child.visits;
            }
        }
        MCTSNodeDisk& child = nodes[most_idx];
        return child.card;
    }

    uint8 selectBest(IdxType node_idx) {
        MCTSNodeDisk& node = nodes[node_idx];

        IdxType best_idx = 0;
        double best_val = -std::numeric_limits<double>::max();
        for (IdxType child_idx : node.childs) {
            if (child_idx == 0)
                continue;
            MCTSNodeDisk& child = nodes[child_idx];
            double val = value(child, 0.0, false);
            if (best_val < val) {
                best_idx = child_idx;
                best_val = val;
            }
        }
        MCTSNodeDisk& child = nodes[best_idx];
        return child.card;
    }

public:
    MCTS() {
        MCTSNodeDisk root;
        root.card = 255;
        root.visits = 1;
        for (uint8 i = 0; i < 28; ++i)
            root.wins[i] = 0;
        for (uint8 i = 0; i < 39; ++i) {
            root.childs[i] = 0;
        }
        nodes.push_back(root);
    }

    uint8 execute(const Hearts::State& cstate, const Hearts::Player& player, unsigned int policyIter, unsigned int rolloutIter) {
        IdxType node;
        IdxType subroot;
        std::array<uint8, 4> points;
        std::vector<uint8> cards_buffer;
        std::vector<IdxType> policy_indices;
        std::vector<IdxType> catchup_indices;
        cards_buffer.reserve(52);
        policy_indices.reserve(52);
        catchup_indices.reserve(52);
        // walk tree according to state
        catchup_indices.push_back(0);
        subroot = catchup(cstate, player, catchup_indices);
        Hearts::getPossibleCards(cstate, player, cards_buffer);
        if (cards_buffer.size() == 1) {
            return cards_buffer[0]; // only one choice, dont think
        }

        for (unsigned int i = 0; i < policyIter; ++i) {
            policy_indices.clear();
            // NOTE: copy of state is mandatory
            Hearts::State state(cstate);
            // selection and expansion
            node = policy(subroot, state, player, policy_indices, cards_buffer);
            // rollout
            for (unsigned int j = 0; j < rolloutIter; ++j) {
                Hearts::State rstate(state);
                rollout(rstate, player, points, cards_buffer);
                // backprop
                backprop(policy_indices, player, points);
                backprop(catchup_indices, player, points);
            }
        }
        return selectBest(subroot);
    }

    template <typename T>
    void printNodeWithChilds(IdxType idx, uint8 shift, T& stream) {
        MCTSNodeDisk& node = nodes[idx];
        for (uint8 i = 0; i < shift; ++i) {
            stream << " ";
        }
        stream << "<" << idx << ", " << int(node.card / 13) << ":" << int(node.card % 13) << ", Visits: " << int(node.visits) << ", Wins: ";
        for (uint8 i = 0; i < 28; ++i) {
            stream << int(node.wins[i]) << ", ";
        }
        stream << ">" << std::endl;

        for (IdxType child_idx : node.childs) {
            if (child_idx == 0)
                continue;
            printNodeWithChilds(child_idx, shift+1, stream);
        }

        for (uint8 i = 0; i < shift; ++i) {
            stream << " ";
        }
        stream << "</" << idx << ">" << std::endl;
    }
};

#endif //MCTS_HPP
