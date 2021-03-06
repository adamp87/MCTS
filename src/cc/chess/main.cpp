#include <array>
#include <random>
#include <chrono>

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include "mcts.hpp"
#include "chess.hpp"

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

#ifdef _OPENMP
typedef MCTS<Chess, MCTSNodeBaseMT<Chess::ActType> > MCTSDef;
#else
typedef MCTS<Chess, MCTSNodeBase<Chess::ActType> > MCTSDef;
#endif

Chess::ActType getCmdInput(const Chess& state, int player) {
    bool valid = false;
    Chess::ActType act;
    Chess::ActType actions[Chess::MaxActions];
    Chess::ActCounterType nActions = state.getPossibleActions(player, player, actions);
    while (!valid) {
        std::string actStr;
        std::cout << "Player" << player << ": ";
        std::cin >> actStr;
        std::transform(actStr.begin(), actStr.end(), actStr.begin(), [](unsigned char c){ return ::toupper(c); });
        if (actStr.size() < 4)
            continue;
        actStr.push_back('N'); // make sure has 4th element
        Chess::ActType::Type actT = Chess::ActType::Normal;
        switch (actStr[4]) {
        case 'C':
            actT = Chess::ActType::Castling;
            break;
        case 'E':
            actT = Chess::ActType::EnPassant;
            break;
        case 'Q':
            actT = Chess::ActType::PromoteQ;
            break;
        case 'R':
            actT = Chess::ActType::PromoteR;
            break;
        case 'B':
            actT = Chess::ActType::PromoteB;
            break;
        case 'K':
            actT = Chess::ActType::PromoteK;
            break;
        case 'M': // game must be finished explicitly
            actT = Chess::ActType::CheckMate;
            break;
        case 'D': // game must be finished explicitly
            actT = Chess::ActType::Even;
            break;
        default:
            break;
        }
        act = Chess::ActType(actStr[0]-'A', actStr[1]-'1', actStr[2]-'A', actStr[3]-'1', actT);
        for (Chess::ActCounterType i = 0; i < nActions; ++i) {
            if (act == actions[i]) {
                valid = true;
            }
        }
    }
    return act;
}

int main(int argc, char** argv) {
    int writeTree = 0;
    std::string workDir = "";
    bool isDeterministic = true;
    std::string portWhite = "tcp://localhost:5555";
    std::string portBlack = "tcp://localhost:5555";
    auto timestamp = std::time(0);
    unsigned int seed = getSeed();
    unsigned int policyIter[2] = {1600, 1600};

    if (argc == 2 && (argv[1] == std::string("-h") || argv[1] == std::string("--help"))) {
        std::cout << "Paramaters:" << std::endl;
        std::cout << "deterministic 1 (deterministic, or 0 for stochastic)" << std::endl;
        std::cout << "portW tcp://localhost:5555 (port for DNN decisions)" << std::endl;
        std::cout << "portB tcp://localhost:5555 (port for DNN decisions)" << std::endl;
        std::cout << "writeTree 0" << std::endl;
        std::cout << "workDir path/" << std::endl;
        std::cout << "seed 123" << std::endl;
        std::cout << "p0 100 (policy iteration for player0, zero for human player)" << std::endl;
        std::cout << "p[1] 100" << std::endl;
        return 0;
    }

    if (argc % 2 == 0) {
        std::cout << "Invalid input, exe key1 value1 key2 value2" << std::endl;
        return -1;
    }

    for (int i = 1; i < argc; i+=2) {
        std::string key(argv[i+0]);
        std::string val(argv[i+1]);
        if (key == "writeTree") {
            writeTree = (val != "0");
        } else if (key == "deterministic") {
            isDeterministic = (val != "0");
        } else if (key == "seed") {
            seed = std::stoi(val);
        } else if (key == "workDir") {
            workDir = val;
        } else if (key == "portW") {
            portWhite = val;
        } else if (key == "portB") {
            portBlack = val;
        } else if (key == "p0") {
            policyIter[0] = std::stoi(val);
        } else if (key == "p1") {
            policyIter[1] = std::stoi(val);
        } else {
            std::cout << "Unknown Key: " << key << std::endl;
            return -1;
        }
    }

    std::cout << "Seed " << seed << std::endl;
    std::cout << "Port White: " << portWhite << std::endl;
    std::cout << "Port Black: " << portBlack << std::endl;
    std::cout << "Deterministic: " << isDeterministic << std::endl;
    std::cout << "Results at: " << (writeTree == 0 ? "Disabled" : workDir) << std::endl;
    for (int i = 0; i < 2; ++i) {
        std::cout << "P" << i << " PIter: " << policyIter[i] << std::endl;
    }

    // init program
    zmq::context_t zmq_context(16);
    std::vector<Chess::ActType> history;
    Chess state(zmq_context, portWhite, portBlack);
    std::array<MCTSDef, 2> ai = {seed, seed};
    if (!state.test_actions()) {
        std::cout << "Error in logic" << std::endl;
        return -1;
    }
    state.setDebugBoard(0); // zero means no change

    // execute game
    for (int time = 0; !state.isFinished(); ++time) {
        int player = state.getPlayer(time);
        auto t0 = std::chrono::high_resolution_clock::now();
        Chess::ActType act =   (policyIter[player] == 0)  ?
                               getCmdInput(state, player) :
                               ai[player].execute(player,
                                                  isDeterministic,
                                                  state,
                                                  policyIter[player],
                                                  history);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::string actDesc = state.getActionDescription(act);
        state.update(act);
        history.push_back(act);

        std::cout << "T" << time << " ";
        std::cout << "P" << int(player) << " ";
        std::cout << Chess::act2str(act) << " ";
        std::cout << actDesc << " ";
        std::cout << state.getBoardDescription() << " ";
        std::cout << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count() << " sec";
        std::cout << std::endl;
    }
    std::cout << state.getEndOfGameString() << std::endl;

    // save tree
    for (int p = 0; p < 2; ++p) {
        if (writeTree == 0)
            continue;

        std::ofstream file;
        std::stringstream filename;
        filename << workDir << "chess_" << timestamp << "_player_" << p << ".csv";

        file.open(filename.str());
        float maxIter = float(policyIter[p]);
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
