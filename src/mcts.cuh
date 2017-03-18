#ifndef MCTS_CUH
#define MCTS_CUH

#include <vector>

#include "defs.hpp"
#include "hearts.hpp"

struct RolloutContainer; // forward declaration

class RolloutContainerCPP {
    RolloutContainer* data;

    unsigned int* _cuRollout(const Hearts::State& state,
                             const Hearts::Player& ai,
                             const uint8* cards,
                             uint8 nCards);
public:
    RolloutContainerCPP(uint32 iterations);
    ~RolloutContainerCPP();

    bool hasGPU() const {
        return data != 0;
    }

    template <class TNode>
    unsigned int* rollout(const Hearts::State& state,
                          const Hearts::Player& ai,
                          const std::vector<TNode>& expandables) {
        uint8 cards[52];
        uint8 nCards = expandables.size();
        for (uint8 i = 0; i < nCards; ++i) {
            cards[i] = expandables[i]->card;
        }
        return _cuRollout(state, ai, cards, nCards);
    }
};

#endif // MCTS_CUH
