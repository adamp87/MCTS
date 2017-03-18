#ifndef MCTS_CUH
#define MCTS_CUH

#include <vector>

#include "defs.hpp"
#include "hearts.hpp"

struct RolloutContainer; // forward declaration

#ifdef __CUDACC__

__host__ RolloutContainer* init(uint32 iterations);
__host__ void free(RolloutContainer* data);
__host__ unsigned int* cuRollout(const Hearts::State& state,
                                 const Hearts::Player& ai,
                                 const uint8* cards,
                                 uint8 nCards,
                                 RolloutContainer* data);

class RolloutContainerCPP {
    RolloutContainer* data;

public:
    RolloutContainerCPP(uint32 iterations) {
        data = init(iterations);
    }

    ~RolloutContainerCPP() {
        free(data);
        data = 0;
    }

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
        return cuRollout(state, ai, cards, nCards, data);
    }
};

#else

class RolloutContainerCPP {
    RolloutContainer* data;

public:
    RolloutContainerCPP(uint32) {
        data = 0;
    }

    ~RolloutContainerCPP() {
    }

    bool hasGPU() const {
        return false;
    }

    template <class TNode>
    unsigned int* rollout(const Hearts::State&,
                          const Hearts::Player&,
                          const std::vector<TNode>&) {
        return 0; // should not be called
    }
};

#endif

#endif // MCTS_CUH
