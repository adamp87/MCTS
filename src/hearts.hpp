#ifndef HEARTS_HPP
#define HEARTS_HPP

#include <array>
#include <vector>

#include <random>
#include <numeric>
#include <algorithm>

#include "defs.hpp"

class Hearts {
    static constexpr uint8 player_map[4][4] = { { 0, 1, 2, 3 },
    { 1, 2, 3, 0 },
    { 2, 3, 0, 1 },
    { 3, 0, 1, 2 } };

    static constexpr uint8 value_map[52] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    static uint8 getHighestPlayerMapID(const std::array<uint8, 52>& order, uint8 round, const uint8* playerMap) {
        uint8 color_first = order[round * 4] / 13;
        uint8 highest_value = order[round * 4] % 13;
        uint8 highest_player = 0;
        for (uint8 i = 1; i < 4; ++i) {
            uint8 color_next = order[round * 4 + i] / 13;
            uint8 value_next = order[round * 4 + i] % 13;
            if (color_first == color_next && highest_value < value_next) {
                highest_value = value_next;
                highest_player = i;
            }
        }
        return playerMap[highest_player];
    }

public:
    struct Player {
        uint8 player; // fixed player id
        std::array<uint8, 52> hand; // 4-unknown, can store info after swap
    };

    class State {
        uint8 round; // 13
        uint8 turn; // 4
        uint8 startPlayer;
        bool heartsBroken;
        const uint8* player_map;
        std::array<uint8, 52> orderInTime; //index time-value card
        std::array<uint8, 52> orderAtCard; //index card-value time
        std::array<bool, 4 * 4> hasNoColor; //player showed that he has no color(known by other players)

        friend class Hearts;
        friend class FlowNetwork;

    public:
        static constexpr uint8 order_unset = 255;

        uint8 getPlayer(uint8 i) const {
            return player_map[i];
        }

        uint8 getCurrentPlayer() const {
            return player_map[turn];
        }

        bool isTerminal() const {
            return round == 13;
        }

        uint8 getCardAtTime(uint8 order) const {
            return orderInTime[order];
        }
    };

    class FlowNetwork {
        static constexpr uint8 node_s = 0;
        static constexpr uint8 node_t = 9;
        static constexpr uint8 node_c[4] = { 1, 2, 3, 4 };
        static constexpr uint8 node_p[4] = { 5, 6, 7, 8 };
        static constexpr uint8 nodeCount = 10;

        static uint8 getEdge(uint8 i, uint8 j) {
            return j*nodeCount + i; //NOTE: findPath() speed is critical: use j*n+i
        }

        static uint8 findPath(const uint8* graph, uint8* path) {
            uint8 v = 0;
            uint8 nStack = 1;
            uint8 stack[nodeCount];
            uint8 parent[nodeCount] = { 0 };
            bool discovered[nodeCount] = { false };

            stack[0] = node_s;
            while (nStack != 0 && v != node_t) {
                v = stack[--nStack]; // pop
                if (discovered[v] == true)
                    continue; // already discovered
                discovered[v] = true; // marked as discovered
                for (uint8 i = 1; i < nodeCount; ++i) { //start from 1, as source is always discovered
                    if (discovered[i] == true)
                        continue; // already discovered
                    if (graph[getEdge(v, i)] == 0)
                        continue; // no edge
                    parent[i] = v; // set predecessor
                    stack[nStack++] = i; // push
                }
            }

            if (parent[node_t] == 0) {
                return 0; // no path found
            }
            //use stack to get reverse path
            nStack = 0;
            stack[0] = parent[node_t];
            while (stack[nStack] != 0)
            { // get reverse path
                stack[nStack + 1] = parent[stack[nStack]];
                ++nStack;
            }
            //return path in normal order
            for (uint8 i = 0; i <= nStack; ++i) {
                path[i] = stack[nStack - i];
            }
            path[++nStack] = node_t; // add sink to path
            return nStack;
        }

        static bool _verify(uint8* graph, uint8 player) {
            uint8 path[nodeCount];

            while (true) {
                uint8 pathLength = findPath(graph, path);
                if (pathLength == 0)
                    break;
                // get capacity of path
                uint8 cap = 52; //set to max
                uint8 v1 = node_s;
                uint8 v2 = node_s;
                for (uint8 i = 1; i <= pathLength; ++i) {
                    v1 = v2;
                    v2 = path[i];
                    uint8 edgeCap = graph[getEdge(v1, v2)];
                    if (edgeCap < cap)
                        cap = edgeCap;
                }

                //send flow
                v1 = node_s;
                v2 = node_s;
                for (uint8 i = 1; i <= pathLength; ++i) {
                    v1 = v2;
                    v2 = path[i];
                    uint8 e1 = getEdge(v1, v2);
                    uint8 e2 = getEdge(v2, v1);
                    graph[e1] -= cap;
                    graph[e2] += cap;
                }
            }

            // if not all source edges are saturaded, game would be invalid
            for (uint8 color = 0; color < 4; ++color) {
                if (graph[getEdge(node_s, node_c[color])] != 0) {
                    return false;
                }
            }
            return true;
        }

    public:
        typedef uint8 Graph[nodeCount * nodeCount];

        static void init(const Hearts::State& state, const Hearts::Player& ai, Graph& graph) {
            for (uint8 i = 0; i < nodeCount * nodeCount; ++i) {
                graph[i] = 0;
            }

            // set edges of players to sink
            for (uint8 p = 0; p < 4; ++p) {
                graph[getEdge(node_p[p], node_t)] = 13 - state.round;
            }
            for (uint8 p = 0; p < 4; ++p) {
                if (p < state.turn) { // player already played card within this round
                    uint8 player = state.player_map[p];
                    graph[getEdge(node_p[player], node_t)] -= 1;
                }
            }
            graph[getEdge(node_p[ai.player], node_t)] = 0; // current player know his cards

            // set edges of source to color, modify player to sink
            for (uint8 color = 0; color < 4; ++color) {
                for (uint8 value = 0; value < 13; ++value) {
                    uint8 card = color * 13 + value;
                    if (state.orderAtCard[card] != Hearts::State::order_unset)
                        continue; //card has been played
                    if (ai.hand[card] == 4) {
                        graph[getEdge(node_s, node_c[color])] += 1; // unknown by color
                    }
                    else if (ai.hand[card] != ai.player) { // card is known because of swap(or open cards)
                        graph[getEdge(node_p[ai.hand[card]], node_t)] -= 1;
                    }
                }

                //set edges between color and players
                for (uint8 player = 0; player < 4; ++player) {
                    if (player == ai.player)
                        continue; // player know his own cards
                    if (state.hasNoColor[player * 4 + color] == false) {
                        graph[getEdge(node_c[color], node_p[player])] = 52;
                    }
                }
            }
        }

        static bool verifyOneColor(const uint8* _graph, uint8 player, uint8 color) {
            Graph graph;
            for (uint8 i = 0; i < nodeCount * nodeCount; ++i) {
                graph[i] = _graph[i];
            }
            graph[getEdge(node_c[color], node_p[player])] = 0; // deactivate edge

            return _verify(graph, player);
        }

        static bool verifyOneCard(const uint8* _graph, uint8 player, uint8 color) {
            Graph graph;
            for (uint8 i = 0; i < nodeCount * nodeCount; ++i) {
                graph[i] = _graph[i];
            }
            if (graph[getEdge(node_p[player], node_t)] == 0) //TODO:should not happen (keep this?)
                return false;
            if (graph[getEdge(node_s, node_c[color])] == 0)
                return false; // all unknown cards of this color have been played
            graph[getEdge(node_s, node_c[color])] -= 1; // decrement unknow cards for color
            graph[getEdge(node_p[player], node_t)] -= 1; // decrement unknow cards for player

            return _verify(graph, player);
        }

        static bool verifyHeart(const uint8* _graph, uint8 player) {
            Graph graph;
            for (uint8 i = 0; i < nodeCount * nodeCount; ++i) {
                graph[i] = _graph[i];
            }
            for (uint8 i = 0; i < 3; ++i)
                graph[getEdge(node_c[i], node_p[player])] = 0; // deactivate all edges, except heart

            return _verify(graph, player);
        }

        static void printGraph(const uint8* graph) {
            for (uint8 i = 0; i < nodeCount; ++i) {
                for (uint8 j = 0; j < nodeCount; ++j) {
                    if (graph[getEdge(i, j)] == 0)
                        continue;
                    //std::cout << "(" << i << "," << j << "):" << graph[getEdge(i, j)] << std::endl;
                }
            }
        }
    };

    static void init(State& state, std::array<Player, 4>& players) {
        state.turn = 0;
        state.round = 0;
        state.startPlayer = 0;
        state.heartsBroken = false;
        state.hasNoColor.fill(false);
        state.orderInTime.fill(State::order_unset);
        state.orderAtCard.fill(State::order_unset);

        // shuffle deck
        std::vector<uint8> cards(52);
        std::iota(cards.begin(), cards.end(), 0);
        std::random_shuffle(cards.begin(), cards.end());

        // set cards
        for (uint8 i = 0; i < 4; ++i) {
            players[i].player = i;
            std::fill(players[i].hand.begin(), players[i].hand.end(), 4);//set to unknown
            for (uint8 j = 0; j < 13; ++j) {
                uint8 card = cards[i * 13 + j];
                players[i].hand[card] = i; //set card to own id
            }
            if (players[i].hand[0] == i) { // set first player
                state.startPlayer = i;
            }
        }
        state.player_map = Hearts::player_map[state.startPlayer];
    }

    static void getPossibleCards(const State& state, const Player& ai, std::vector<uint8>& cards) {
        cards.clear();
        uint8 player = state.player_map[state.turn]; // mapped player id
        if (state.round == 0 && state.turn == 0 && player == ai.player) { //ai start with first card
            cards.push_back(0);
            return;
        }

        uint8 color_first = state.orderInTime[state.round * 4] / 13; //note, must check if not first run

        if (player == ai.player) { // own
            // check if has color
            bool aiHasNoColor[4];
            for (uint8 color = 0; color < 4; ++color) {
                aiHasNoColor[color] = true;
                for (uint8 value = 0; value < 13; ++value) {
                    uint8 card = color * 13 + value;
                    if (ai.hand[card] == ai.player && state.orderAtCard[card] == Hearts::State::order_unset) {
                        aiHasNoColor[color] = false;
                        break;
                    }
                }
            }

            // select cards
            for (uint8 color = 0; color < 4; ++color) {
                if (aiHasNoColor[color] == true)
                    continue; // ai has no card of this color
                if (state.turn != 0 && color != color_first && aiHasNoColor[color_first] == false)
                    continue; // must play same color
                if (state.round == 0 && color == 3 && !(aiHasNoColor[0] && aiHasNoColor[1] && aiHasNoColor[2]))
                    continue; // in first round no hearts (only if he has only hearts, quite impossible:))
                if (state.turn == 0 && color == 3 && !(state.heartsBroken || (aiHasNoColor[0] && aiHasNoColor[1] && aiHasNoColor[2])))
                    continue; // if hearts not broken or has other color, no hearts as first card
                for (uint8 value = 0; value < 13; ++value) {
                    if (state.round == 0 && color == 2 && value == 10)
                        continue; // in first round no queen

                    uint8 card = color * 13 + value;
                    if (state.orderAtCard[card] != State::order_unset)
                        continue; //card has been played

                    if (ai.hand[card] == ai.player) {
                        cards.push_back(card); // select card
                    }
                }
            }
        }
        else { // opponent

            // check if it is know if opponent has color (swap, open cards)
            bool hasColor[4];
            for (uint8 color = 0; color < 4; ++color) {
                hasColor[color] = false;
                for (uint8 value = 0; value < 13; ++value) {
                    uint8 card = color * 13 + value;
                    if (state.orderAtCard[card] != State::order_unset)
                        continue; //card has been played
                    if (ai.hand[card] == player) {
                        hasColor[color] = true;
                        break;
                    }
                }
            }

            FlowNetwork::Graph graph;
            FlowNetwork::init(state, ai, graph);

            for (uint8 color = 0; color < 4; ++color) {
                if (state.hasNoColor[player * 4 + color] == true)
                    continue; // ai has no card of this color
                if (state.turn != 0 && color != color_first && hasColor[color_first] == true)
                    continue; // it is known that ai has card of the first color
                if (state.turn == 0 && color == 3 && !state.heartsBroken && (hasColor[0] || hasColor[1] || hasColor[2]))
                    continue; // player has other color therefore cant start with hearth
                // theoretically opponent can play any color (he could play hearts even in first round(if he has no other color))

                // verify if flow network breaks by playing a card from the given color
                if (hasColor[color] == false) {
                    if (FlowNetwork::verifyOneCard(graph, player, color) == false)
                        continue;
                }

                // verify if hearts can be played for the first time as first card
                if (state.turn == 0 && color == 3 && !state.heartsBroken && hasColor[3] == false) {
                    if (FlowNetwork::verifyHeart(graph, player) == false)
                        continue;
                }

                // verify if other color than first can be played
                if (state.turn != 0 && color != color_first && hasColor[color] == false) {
                    if (FlowNetwork::verifyOneColor(graph, player, color_first) == false)
                        continue;
                }

                for (uint8 value = 0; value < 13; ++value) {
                    if (state.round == 0 && color == 2 && value == 10)
                        continue; // in first round no queen

                    uint8 card = color * 13 + value;
                    if (state.orderAtCard[card] != State::order_unset)
                        continue; //card has been played

                    if (ai.hand[card] == player || ai.hand[card] == 4) {
                        cards.push_back(card); // select from known or unknown opponent cards
                    }
                }
            }
        }

        //TODO: filter cards (values next to each other), dont forget result is the same for all, dont discard here?
    }

    static void update(State& state, uint8 card) {
        //set card order
        uint8 player = state.player_map[state.turn]; // mapped player id
        uint8 order = state.round * 4 + state.turn;
        state.orderInTime[order] = card;
        state.orderAtCard[card] = order;

        //get first and next card color of round
        uint8 color_first = state.orderInTime[state.round * 4] / 13;
        uint8 color_next = card / 13;

        if (state.turn == 0 && color_next == 3) { //player starts with hearts
            if (state.heartsBroken == false) { // player has only hearts
                for (uint8 i = 0; i < 3; ++i) {
                    state.hasNoColor[player * 4 + i] = true;
                }
            }
            state.heartsBroken = true;
        }

        if (color_first != color_next) { // card color is not the same
            state.hasNoColor[player * 4 + color_first] = true;
            if (color_next == 3) { // hearts broken
                state.heartsBroken = true;
            }
        }

        //next player
        state.turn += 1;
        if (state.turn == 4) { //end of round
            uint8 highest_player = getHighestPlayerMapID(state.orderInTime, state.round, state.player_map);
            state.player_map = Hearts::player_map[highest_player]; // set player map
            state.round += 1;
            state.turn = 0;
        }
    }

    static void computePoints(const State& state, std::array<uint8, 4>& points) {
        uint8 highestPlayer = state.startPlayer;
        for (uint8 i = 0; i < 4; ++i) {
            points[i] = 0; // clear points
        }

        for (uint8 round = 0; round < 13; ++round) {
            uint8 point = 0;
            const uint8* map = Hearts::player_map[highestPlayer];
            highestPlayer = getHighestPlayerMapID(state.orderInTime, round, map);
            for (uint8 player = 0; player < 4; ++player) {
                uint8 card = state.orderInTime[round * 4 + player];
                point += value_map[card]; // add points
            }
            points[highestPlayer] += point;
        }
    }
};

#endif //HEARTS_HPP
