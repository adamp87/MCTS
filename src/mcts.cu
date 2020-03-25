#include <cuda.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>

#include <mutex>
#include <numeric>
#include <iostream>

#include "mcts.cuh"
#include "hearts.hpp"

// NOTE: this is a dirty solution, compile flow.cpp for cuda here
#include "flownetwork.cpp"

class Hearts;
template class RolloutCUDA<Hearts>;

template <class TProblem>
struct RolloutCUDA<TProblem>::impl {
    TProblem* u_state;
    double* u_result; //size=maxrollout
    curandState* d_rnd;

    std::mutex lock;
    constexpr static int nThread = 32;
};

template <typename TProblem>
__global__ void rollout(int idxAi,
                        int maxRolloutDepth,
                        const TProblem* u_state,
                        curandState* d_rnd,
                        double* u_result) {
    typedef typename TProblem::ActType ActType;
    typedef typename TProblem::ActCounterType ActCounterType;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    TProblem state(*u_state);
    ActType actions[TProblem::MaxActions];

    // Copy state to local memory for efficiency
    curandState localRnd = d_rnd[idx];

    int depth = 0; //if max is zero, until finished
    while (!state.isFinished()) {
        if (++depth == maxRolloutDepth)
            break;
        ActCounterType count = state.getPossibleActions(idxAi, state.getPlayer(), actions);
        ActCounterType pick = curand(&localRnd) % count;
        state.update(actions[pick]);
    }

    // Copy state back to global memory
    d_rnd[idx] = localRnd;

    u_result[idx] = state.computeMCTSWin(idxAi);
}

__global__ void setup_random(curandState* state, unsigned int seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(seed, id, 0, &state[id]);
}

template <class TProblem>
RolloutCUDA<TProblem>::RolloutCUDA(unsigned int* iterations, unsigned int seed) {
    pimpl = 0;
    int deviceCount = 0;
    if(cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        std::cout << "Failed to get device count" << std::endl;
        return;
    }
    if (deviceCount == 0) {
        std::cout << "No CUDA device found" << std::endl;
        return;
    }
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if (i == 0)
            std::cout << "Selected ";
        std::cout << "Device number: " << i << " Device name: " << prop.name << std::endl;
    }

    unsigned int maxIterations = iterations[0];
    for (int i = 0; i < 4; ++i) {
        if (iterations[i] == 1)
            continue; // cuda rollout disabled for player
        if (iterations[i] % RolloutCUDA::impl::nThread != 0) {
            std::cout << "Rollout " << i << " is not dividable with " << RolloutCUDA::impl::nThread << std::endl;
            return;
        }
        if (maxIterations < iterations[i]) {
            maxIterations = iterations[i];
        }
    }
    if (maxIterations == 1) {
        std::cout << "CUDA was not requested, enable with e.g. (r3 2048)" << std::endl;
        return;
    }

    std::unique_ptr<RolloutCUDA::impl> ptr(new RolloutCUDA::impl());
    dim3 threads(RolloutCUDA::impl::nThread);
    dim3 blocks(maxIterations / threads.x);
    if(cudaMallocManaged(&ptr->u_state, sizeof(TProblem)) != cudaSuccess) {
        std::cout << "Failed to allocate state" << std::endl;
        return;
    }
    if(cudaMalloc(&ptr->d_rnd, sizeof(curandState) * maxIterations) != cudaSuccess) {
        std::cout << "Failed to allocate random" << std::endl;
        return;
    }
    if(cudaMallocManaged(&ptr->u_result, sizeof(double) * maxIterations) != cudaSuccess) {
        std::cout << "Failed to allocate results" << std::endl;
        return;
    }

    setup_random<<<blocks, threads>>>(ptr->d_rnd, seed);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cout << "Failed to sync CUDA call" << std::endl;
        return;
    }

    pimpl = ptr.release();
}

template <class TProblem>
RolloutCUDA<TProblem>::~RolloutCUDA() {
    if (pimpl == 0)
        return;
    cudaFree(pimpl->d_rnd);
    cudaFree(pimpl->u_state);
    cudaFree(pimpl->u_result);
    delete pimpl;
    pimpl = 0;
}

template <typename TProblem>
__host__ bool RolloutCUDA<TProblem>::cuRollout(int idxAi,
                                               int maxRolloutDepth,
                                               const TProblem& state,
                                               unsigned int iterations,
                                               double& winSum) const {
    std::unique_lock<std::mutex> lock(pimpl->lock, std::defer_lock);
    if (lock.try_lock() == false)
        return false;
    dim3 threads(RolloutCUDA::impl::nThread);
    dim3 blocks(iterations / threads.x);
    *pimpl->u_state = state;
    rollout<<<blocks, threads>>>(idxAi,
                                 maxRolloutDepth,
                                 pimpl->u_state,
                                 pimpl->d_rnd,
                                 pimpl->u_result);
    if(cudaDeviceSynchronize() != cudaSuccess) {
        std::cout << "Failed to sync CUDA call" << std::endl;
        return false;
    }

    winSum = std::accumulate(pimpl->u_result, pimpl->u_result + iterations, 0.0);
    return true;
}
