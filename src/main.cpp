#include <array>
#include <random>

#include <sstream>
#include <fstream>
#include <iostream>

#include "defs.hpp"
#include "mcts.hpp"
#include "hearts.hpp"

int main() {
    std::srand(0); // fixed for debugging purpose

    Hearts::State state;
    std::array<MCTS, 4> ai;
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

    //print player cards
    for (uint8 p = 0; p < 4; ++p) {
        std::cout << "Player" << int(p) << ":";
        gameStream << "Player" << int(p) << ":";
        for (uint8 color = 0; color < 4; ++color) {
            for (uint8 value = 0; value < 13; ++value) {
                uint8 card = 13 * color + value;
                if (players[p].hand[card] == p) {
                    std::cout << int(color) << ":" << int(value) << " ";
                    gameStream << int(color) << ":" << int(value) << " ";
                }
            }
        }
        std::cout << std::endl;
        gameStream << std::endl;
    }

    // execute game
    for (uint8 round = 0; round < 13; ++round) {
        std::cout << "Round " << round + 1 << ":";
        gameStream << "Round " << round + 1 << ":";
        for (uint8 p = 0; p < 4; ++p) {
            uint8 player = state.getPlayer(p);
            uint8 card = ai[player].execute(state, players[player], policyIter, rolloutIter);
            Hearts::update(state, card);

            std::cout << "P" << int(player) << ": ";
            gameStream << "P" << int(player) << ": ";
            std::cout << card / 13 << "," << card % 13 << " ";
            gameStream << card / 13 << "," << card % 13 << " ";
        }
        std::cout << std::endl;
        gameStream << std::endl;
    }

    //print points
    std::array<uint8, 4> points;
    Hearts::computePoints(state, points);
    for (int p = 0; p < 4; ++p) {
        std::cout << "P" << p << ": " << int(points[p]) << std::endl;
        gameStream << "P" << p << ": " << int(points[p]) << std::endl;
    }

    //save tree
    for (int p = 0; p < 4; ++p) {
        std::ofstream file;
        std::stringstream sstream;
        sstream << "log/tree_a" << p << "_p" << policyIter << "_r" << rolloutIter << ".txt";
        std::string filename(sstream.str());
        sstream.str(std::string());

        ai[p].printNodeWithChilds(0, 0, sstream);
        file.open(filename);
        file << gameStream.str();
        file << sstream.str();
    }
    std::cout << "Bye" << std::endl;

    return 0;
}
