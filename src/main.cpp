#include <array>
#include <random>

#include <sstream>
#include <fstream>
#include <iostream>

#include "defs.hpp"
#include "mcts.hpp"
#include "mctree.hpp"
#include "hearts.hpp"

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

int main() {
    int seed = getSeed();
    std::srand(seed);

    Hearts::State state;
    std::array<MCTS<MCTreeStaticArray>, 4> ai;
    std::array<Hearts::Player, 4> players;
    Hearts::init(state, players);

    unsigned int policyIter = 2000;// 100000;
    unsigned int rolloutIter = 10;
    std::stringstream gameStream;

    //cheat, play with open cards
    for (uint8 p = 0; p < 4; ++p) {
        for (uint8 color = 0; color < 4; ++color) {
            for (uint8 value = 0; value < 13; ++value) {
                uint8 card = 13 * color + value;
                for (uint8 p2 = 0; p2 < 4; ++p2) {
                    if (players[p2].hand[card] == p2) {
                        //players[p].hand[card] = p2; // player see cards of other players
                    }
                }
            }
        }
    }

    std::cout << "Start game with seed:" << seed << std::endl;
    gameStream << "Seed:" << seed << std::endl;
    //gameStream << "Cheet On" << std::endl;
    gameStream << "Cheet Off" << std::endl;

    //print player cards
    for (uint8 p = 0; p < 4; ++p) {
        std::cout << "Player" << int(p) << ":";
        gameStream << "Player" << int(p) << ";";
        for (uint8 color = 0; color < 4; ++color) {
            for (uint8 value = 0; value < 13; ++value) {
                uint8 card = 13 * color + value;
                if (players[p].hand[card] == p) {
                    std::cout << int(color) << ":" << int(value) << " ";
                    gameStream << int(color) << ":" << int(value) << "(" << int(card) << ")" << ";";
                }
            }
        }
        std::cout << std::endl;
        gameStream << std::endl;
    }

    // execute game
    for (uint8 round = 0; round < 13; ++round) {
        std::cout << "Round " << round + 1 << ":";
        gameStream << "Round " << round + 1 << ";";
        for (uint8 turn = 0; turn < 4; ++turn) {
            uint8 player = state.getPlayer(round * 4 + turn);
            uint8 card = ai[player].execute(state, players[player], policyIter, rolloutIter);
            Hearts::update(state, card);

            std::cout << "P" << int(player) << ": ";
            gameStream << "P" << int(player) << ";";
            std::cout << card / 13 << "," << card % 13 << " ";
            gameStream << card / 13 << ":" << card % 13 << "(" << int(card) << ")" ";";
        }
        std::cout << std::endl;
        gameStream << std::endl;
    }

    //print points
    std::array<uint8, 4> points;
    Hearts::computePoints(state, points);
    for (int p = 0; p < 4; ++p) {
        std::cout << "P" << p << ": " << int(points[p]) << std::endl;
        gameStream << "P" << p << ":" << int(points[p]) << std::endl;
    }

    std::string workDir = "/";

    //save tree
    for (int p = 0; p < 4; ++p) {
        std::ofstream file;
        std::ofstream resFile;
        std::stringstream sstream;
        //sstream << "log/tree_a" << p << "_p" << policyIter << "_r" << rolloutIter << ".txt";
        sstream << workDir << "results/seed_" << seed << ".csv";
        std::string filename(sstream.str());
        sstream.str(std::string());
        sstream << workDir << "results/seed_" << seed << "_player_" << p << ".csv";
        std::string resFileName(sstream.str());
        sstream.str(std::string());

        float maxIter = float(policyIter * rolloutIter);
        resFile.open(resFileName);
        ai[p].writeResults(state, players[p], maxIter, resFile);
        
        file.open(filename);
        //ai[p].printNodeWithChilds(ai[p].getRoot(), 0, sstream);
        file << gameStream.str();
        //file << sstream.str();
    }

    //std::cout << "Bye" << std::endl;

    return 0;
}
