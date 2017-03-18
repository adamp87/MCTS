#ifndef HEARTS_HPP
#define HEARTS_HPP

#include <array>
#include <vector>

#include <random>
#include <numeric>
#include <algorithm>

#include "defs.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Hearts {
    CUDA_CALLABLE_MEMBER static uint8 getHighestPlayerMapID(const uint8* order, uint8 round, const uint8* playerMap) {
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
        uint8 hand[52]; // 4-unknown, can store info after swap
    };

    class State {
        uint8 round; // 13
        uint8 turn; // 4
        uint8 startPlayer;
        bool heartsBroken;
        uint8 player_map[4];
        uint8 orderInTime[52]; //index time-value card
        uint8 orderAtCard[52]; //index card-value time
        bool hasNoColor[4 * 4]; //player showed that he has no color(known by other players)

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

        uint8 getCardAtTime(uint8 order) const {
            return orderInTime[order];
        }

        bool isFirstRoundTurn() const {
            return round == 0 && turn == 0;
        }

        CUDA_CALLABLE_MEMBER bool isTerminal() const {
            return round == 13;
        }

        CUDA_CALLABLE_MEMBER static void setPlayerMap(uint8 player, uint8* map) {
            const uint8 cplayer_map[4][4] = { { 0, 1, 2, 3 },
                                              { 1, 2, 3, 0 },
                                              { 2, 3, 0, 1 },
                                              { 3, 0, 1, 2 } };
            for (uint8 i = 0; i < 4; ++i) {
                map[i] = cplayer_map[player][i];
            }
        }
    };

    class FlowNetwork {
        static constexpr uint8 node_s = 0;
        static constexpr uint8 node_t = 9;
        static constexpr uint8 node_cA[4] = { 1, 2, 3, 4 };
        static constexpr uint8 node_pA[4] = { 5, 6, 7, 8 };
        static constexpr uint8 nodeCount = 10;

        constexpr static CUDA_CALLABLE_MEMBER uint8 get_node_c(uint8 idx) { return node_cA[idx]; }
        constexpr static CUDA_CALLABLE_MEMBER uint8 get_node_p(uint8 idx) { return node_pA[idx]; }

        static CUDA_CALLABLE_MEMBER void get_node(uint8* c, uint8* p) {
            c[0] = get_node_c(0); c[1] = get_node_c(1); c[2] = get_node_c(2); c[3] = get_node_c(3);
            p[0] = get_node_p(0); p[1] = get_node_p(1); p[2] = get_node_p(2); p[3] = get_node_p(3);
        }

        CUDA_CALLABLE_MEMBER static uint8 getEdge(uint8 i, uint8 j) {
            return j*nodeCount + i; //NOTE: findPath() speed is critical: use j*n+i
        }

        CUDA_CALLABLE_MEMBER static uint8 findPath(const uint8* graph, uint8* path) {
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

        CUDA_CALLABLE_MEMBER static bool _verify(uint8* graph, uint8 player) {
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
            return graph[getEdge(node_s, get_node_c(0))] == 0
                && graph[getEdge(node_s, get_node_c(1))] == 0
                && graph[getEdge(node_s, get_node_c(2))] == 0
                && graph[getEdge(node_s, get_node_c(3))] == 0;
        }

    public:
        typedef uint8 Graph[nodeCount * nodeCount];

        CUDA_CALLABLE_MEMBER static void init(const Hearts::State& state, const Hearts::Player& ai, Graph& graph) {
            uint8 node_c[4];
            uint8 node_p[4];
            get_node(node_c, node_p);
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

        CUDA_CALLABLE_MEMBER static bool verifyOneColor(const uint8* _graph, uint8 player, uint8 color) {
            Graph graph;
            uint8 node_c[4];
            uint8 node_p[4];
            get_node(node_c, node_p);
            for (uint8 i = 0; i < nodeCount * nodeCount; ++i) {
                graph[i] = _graph[i];
            }
            graph[getEdge(node_c[color], node_p[player])] = 0; // deactivate edge

            return _verify(graph, player);
        }

        CUDA_CALLABLE_MEMBER static bool verifyOneCard(const uint8* _graph, uint8 player, uint8 color) {
            Graph graph;
            uint8 node_c[4];
            uint8 node_p[4];
            get_node(node_c, node_p);
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

        CUDA_CALLABLE_MEMBER static bool verifyHeart(const uint8* _graph, uint8 player) {
            Graph graph;
            uint8 node_c[4];
            uint8 node_p[4];
            get_node(node_c, node_p);
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
        std::fill(state.hasNoColor, state.hasNoColor + 4 * 4, false);
        std::fill(state.orderInTime, state.orderInTime + 52, State::order_unset);
        std::fill(state.orderAtCard, state.orderAtCard + 52, State::order_unset);

        // shuffle deck
        std::vector<uint8> cards(52);
        std::iota(cards.begin(), cards.end(), 0);
        std::random_shuffle(cards.begin(), cards.end());

        // set cards
        for (uint8 i = 0; i < 4; ++i) {
            players[i].player = i;
            std::fill(players[i].hand, players[i].hand + 52, 4);//set to unknown
            for (uint8 j = 0; j < 13; ++j) {
                uint8 card = cards[i * 13 + j];
                players[i].hand[card] = i; //set card to own id
            }
            if (players[i].hand[0] == i) { // set first player
                state.startPlayer = i;
            }
        }
        Hearts::State::setPlayerMap(state.startPlayer, state.player_map);
    }

    CUDA_CALLABLE_MEMBER static uint8 getPossibleCards(const State& state, const Player& ai, uint8* cards) {
        uint8 count = 0;
        uint8 player = state.player_map[state.turn]; // mapped player id

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
                        cards[count++] = card; // select card
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
                        cards[count++] = card; // select from known or unknown opponent cards
                    }
                }
            }
        }
        return count;

        //TODO: filter cards (values next to each other), dont forget result is the same for all, dont discard here?
    }

    CUDA_CALLABLE_MEMBER static void update(State& state, uint8 card) {
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
            Hearts::State::setPlayerMap(highest_player, state.player_map); // set player map
            state.round += 1;
            state.turn = 0;
        }
    }

    static void computePoints(const State& state, std::array<uint8, 4>& points) {
        computePoints(state, points.data());
    }

    CUDA_CALLABLE_MEMBER static void computePoints(const State& state, uint8* points) {
        const uint8 value_map[52] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

        uint8 map[4];
        uint8 highestPlayer = state.startPlayer;
        for (uint8 i = 0; i < 4; ++i) {
            points[i] = 0; // clear points
        }

        for (uint8 round = 0; round < 13; ++round) {
            uint8 point = 0;
            Hearts::State::setPlayerMap(highestPlayer, map);
            highestPlayer = getHighestPlayerMapID(state.orderInTime, round, map);
            for (uint8 player = 0; player < 4; ++player) {
                uint8 card = state.orderInTime[round * 4 + player];
                point += value_map[card]; // add points
            }
            points[highestPlayer] += point;
        }
    }

    CUDA_CALLABLE_MEMBER static uint8 mapPoints2Wins(const Player& player, uint8* points) {
        bool shotTheMoon = false;
        for (uint8 p = 0; p < 4; ++p) {
            if (points[p] == 26)
                shotTheMoon = true;
        }
        if (shotTheMoon == true) {
            if (points[player.player] == 26) { // current ai shot the moon
                return 0;
            }
            else { // someone shot the moon
                return 27;
            }
        }
        else { // normal points (shifted with one)
            return points[player.player] + 1;
        }
    }
};

#endif //HEARTS_HPP
