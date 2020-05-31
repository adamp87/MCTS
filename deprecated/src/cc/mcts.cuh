#ifndef MCTS_CUH
#define MCTS_CUH

//! Execute multiple rollout on CUDA
/*!
 * \details TODO
 * \author adamp87
*/
template <class TProblem>
class RolloutCUDA {
    struct impl;
    impl* pimpl;

public:
    RolloutCUDA(unsigned int* iterations, unsigned int seed);
    ~RolloutCUDA();

    bool hasGPU() const {
        return pimpl != 0;
    }

    //! TODO
    bool cuRollout(int idxAi,
                   int maxRolloutDepth,
                   const TProblem& state,
                   unsigned int iteration,
                   double& W) const;
};

#endif // MCTS_CUH
