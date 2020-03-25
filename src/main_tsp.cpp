#include <array>
#include <random>

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include "mcts.hpp"
#include "mcts.cuh"
#include "mctree.hpp"
#include "tsp.hpp"
#include "mcts_debug.hpp"

typedef TSP_Vertex TSP;

#ifdef _OPENMP
typedef MCTreeDynamic<MCTSNodeBaseMT<TSP::ActType> > TreeContainer;

// keep static_assert, might be usefull later
static_assert(std::is_same<TreeContainer, MCTreeDynamic<MCTSNodeBaseMT<TSP::ActType> > >::value,
              "Only MCTreeDynamic<MCTSNodeBaseMT<> > can be compiled with OpenMP");
#else
typedef MCTreeDynamic<MCTSNodeBase<TSP::ActType> > TreeContainer;
#endif

//typedef MCTS_Policy_Debug PolicyDebug;
typedef MCTS_Policy_Debug_Dummy PolicyDebug;

MovesOOC::TmpFile MovesOOC::tmp;
std::vector<double> TSP_Base::edgeWeights;

#ifdef __linux__
#include <ctime>
#include <unistd.h>
int getSeed() {
    return ((time(NULL) & 0xFFFF) | (getpid() << 16));
}
#else
#include <ctime>
#include <windows.h>
int getSeed() {
    return ((time(NULL) & 0xFFFF) | (GetCurrentProcessId() << 16));
}
#endif

int main(int argc, char** argv) {
    int writeTree = 0;
    int maxRolloutDepth = 1;
    std::string input = "";
    std::string workDir = "";
    auto timestamp = std::time(0);
    unsigned int seed = getSeed();
    unsigned int policyIter[1] = {25000};
    unsigned int rolloutIter[1] = {1};

    if (argc == 2 && (argv[1] == std::string("-h") || argv[1] == std::string("--help"))) {
        std::cout << "Paramaters:" << std::endl;
        std::cout << "input path" << std::endl;
        std::cout << "writeTree 0" << std::endl;
        std::cout << "timestamp 0" << std::endl;
        std::cout << "workDir path/" << std::endl;
        std::cout << "seed 123" << std::endl;
        std::cout << "p0 100 (policy iteration" << std::endl;
        std::cout << "r0 100 (rollout iteration" << std::endl;
        return 0;
    }

    if (argc % 2 == 0) {
        std::cout << "Invalid input, exe key1 value1 key2 value2" << std::endl;
        return -1;
    }

    for (int i = 1; i < argc; i+=2) {
        std::string key(argv[i+0]);
        std::string val(argv[i+1]);
        if (key == "input") {
            input = val;
        } else if (key == "timestamp") {
            timestamp = std::stoi(val);
        } else if (key == "writeTree") {
            writeTree = (val != "0");
        } else if (key == "seed") {
            seed = std::stoi(val);
        } else if (key == "workDir") {
            workDir = val;
        } else if (key == "p0") {
            policyIter[0] = std::stoi(val);
        } else if (key == "r0") {
            rolloutIter[0] = std::stoi(val);
        } else {
            std::cout << "Unknown Key: " << key << std::endl;
            return -1;
        }
    }

    std::cout << "Seed " << seed << std::endl;
    std::cout << "Results at: " << (writeTree == 0 ? "Disabled" : workDir) << std::endl;
    for (int i = 0; i < 1; ++i) {
        std::cout << "P" << i << " PIter: " << policyIter[i] << " Riter: " << rolloutIter[i] << std::endl;
    }

    // init program
    std::srand(seed);
    TSP state(input);
    std::vector<TSP::ActType> history;
    PolicyDebug policyDebug(writeTree, workDir, "tsp", timestamp);
    std::array<MCTS<TreeContainer, TSP, PolicyDebug>, 1> ai;

    // int cuda rollout
    std::unique_ptr<RolloutCUDA<TSP> > rolloutCuda(new RolloutCUDA<TSP>(rolloutIter, seed));
    if (rolloutCuda->hasGPU()) {
        std::cout << "GPU Mode" << std::endl;
    } else {
        std::cout << "CPU Mode" << std::endl;
    }

    // execute game
    for (int time = 0; !state.isFinished(); ++time) {
        int player = state.getPlayer(time);
        TSP::ActType act = ai[player].execute(player,
                                              maxRolloutDepth,
                                              state,
                                              policyIter[player],
                                              rolloutIter[player],
                                              history,
                                              rolloutCuda.get(),
                                              policyDebug);
        state.update(act);
        history.push_back(act);

        std::cout << "T" << time << " ";
        std::cout << TSP::act2str(act) << " ";
        std::cout << state.getTourLength() << " ";
        std::cout << std::endl;
    }

    // save tree
    for (int p = 0; p < 1; ++p) {
        if (writeTree == 0)
            continue;

        std::ofstream file;
        std::stringstream filename;
        filename << workDir << "tsp_" << timestamp << ".csv";

        file.open(filename.str());
        float maxIter = float(policyIter[p] * rolloutIter[p]);
        ai[p].writeResults(state, p, maxIter, history, file);
        file.close();

        { // filter results, write only first level branch nodes
            std::ifstream src;
            std::ofstream dst;
            std::string line;

            src.open(filename.str());
            filename << "_filtered.csv";
            dst.open(filename.str());
            std::getline(src, line); // get header
            dst << line << std::endl;
            while (std::getline(src, line)) {
                if (line[0] != '0')
                    continue; // skip lines that not first level branch
                dst << line << std::endl;
            }
            dst.close();
            src.close();
        }
    }

    return 0;
}
