#include "flownetwork.hpp"
#include "hearts.hpp"

constexpr uint8 FlowNetwork::node_cA[4];
constexpr uint8 FlowNetwork::node_pA[4];

CUDA_CALLABLE_MEMBER FlowNetwork::FlowNetwork(const Hearts& state, uint8 ai, const uint8* ai_hand, const bool* hasNoColor) {
    for (uint8 i = 0; i < nodeCount * nodeCount; ++i) {
        graph[i] = 0;
    }

    // set edges of players to sink
    for (uint8 p = 0; p < 4; ++p) {
        graph[getEdge(node_pA[p], node_t)] = 13 - state.round;
    }
    for (uint8 p = 0; p < 4; ++p) {
        if (p < state.turn) { // player already played card within this round
            uint8 player = state.orderPlayer[state.round * 4 + p];
            graph[getEdge(node_pA[player], node_t)] -= 1;
        }
    }
    graph[getEdge(node_pA[ai], node_t)] = 0; // current player know his cards

    // set edges of source to color, modify player to sink
    for (uint8 color = 0; color < 4; ++color) {
        for (uint8 value = 0; value < 13; ++value) {
            uint8 card = color * 13 + value;
            if (state.orderAtCard[card] != Hearts::order_unset)
                continue; //card has been played
            if (ai_hand[card] == Hearts::order_unset) {
                graph[getEdge(node_s, node_cA[color])] += 1; // unknown by color
            }
            else if (ai_hand[card] != ai) { // card is known because of swap(or open cards)
                graph[getEdge(node_pA[ai_hand[card]], node_t)] -= 1;
            }
        }

        //set edges between color and players
        for (uint8 player = 0; player < 4; ++player) {
            if (player == ai)
                continue; // player know his own cards
            if (hasNoColor[player * 4 + color] == false) {
                graph[getEdge(node_cA[color], node_pA[player])] = 52;
            }
        }
    }
}
