#ifndef CONNECT4_HPP
#define CONNECT4_HPP

#include <string>
#include <vector>
#include <sstream>

#include <zmq.hpp>

class Connect4 {
public:
    struct ActType {
        std::uint_fast8_t x;
        std::uint_fast8_t y;

        ActType() : x(0), y(0) {}
        ActType(int x, int y) : x(x), y(y) {}

        operator std::string() const {
            std::string str("X-Y-");
            str[1] = char(x)+'1';
            str[3] = char(y)+'1';
            return str;
        }

        bool operator==(const ActType& other) const {
            return x==other.x && y==other.y;
        }
    };


    typedef std::uint_fast8_t ActCounterType; //!< interface
    constexpr static double UCT_C = 1.0; //!< interface, constant for exploration in uct formula
    constexpr static double DirichletAlpha = 1.0 / 7.0; //!< interface
    constexpr static unsigned int MaxActions = 6 * 7; //!< interface
    constexpr static unsigned int MaxChildPerNode = MaxActions; //!< interface

private:
    enum Figure { White=0, Black=1, Unset=2 };
    struct Board {
        Figure board[6*7];

        Figure& operator[](int i) {
            return board[i];
        }

        const Figure& operator[](int i) const {
            return board[i];
        }
    };


    int time;
    Board board;
    bool finished[2];
    std::vector<Board> history;

    std::string ports[2]; //!< ports for zeromq socket connections: white, black
    zmq::context_t& zmq_context; //!< zeromq context for socket connections

    int getXY(int y, int x) const {
        return y*7+x;
    }

public:
    //! Set initial state
    Connect4(zmq::context_t& zmq_context, const std::string& portW, const std::string& portB)
        : zmq_context(zmq_context)
    {
        ports[0] = portW;
        ports[1] = portB;

        time = 0;
        finished[0] = finished[1] = false;
        for (int i = 0; i < 6*7; ++i)
            board[i] = Figure::Unset;
    }

    //! Interface
    int getPlayer(int time=-1) const {
        if (time == -1)
            time = this->time;
        return time%2;
    }

    //! Interface
    bool isFinished() const {
        return finished[0] || finished[1];
    }

    //! Interface, Implements game logic, return the possible actions that idxAi can play
    /*!
    * \param idxMe ID of player who executes function
    * \param idxAi ID of player to get possible actions for
    * \param actions Allocated array to store possible actions
    * \return Number of possible actions
    */
    ActCounterType getPossibleActions(int idxMe, int idxAi, ActType* actions) const {
        ActCounterType count = 0;
        for(int x = 0; x < 7; ++x) { // left to right
            if (board[getXY(5, x)] != int(Figure::Unset))
                continue; // top element not free
            for (int y = 0; y < 6; ++y) { // from bottom to top
                if (board[getXY(y, x)] != Figure::Unset)
                    continue; // not free
                actions[count].x = x;
                actions[count].y = y;
                ++count;
                break; // first free element stop
            }
        }
        return count;
    }

    //! Interface, Update the game state according to move
    void update(ActType& act) {
        int idxAi = getPlayer();

        auto getPos = [&] (int x, int y, int dx, int dy) -> int { return getXY(y+dy,x+dx); };
        auto isInsideBoard = [&] (int x, int y, int dx, int dy) -> bool { int xx=x+dx; int yy=y+dy; return 0<=xx && xx<7 && 0<=yy && yy<6; };
        auto isOwnStone = [&] (int x, int y, int dx, int dy) -> bool { return isInsideBoard(x,y,dx,dy) && board[getPos(x,y,dx,dy)] == Figure(idxAi); };

        auto opNul = [&] (int    ) -> int { return    0; };
        auto opPos = [&] (int val) -> int { return  val; };
        auto opNeg = [&] (int val) -> int { return -val; };
        auto scanLine = [&] (int x, int y, std::function<int (int)> opX, std::function<int (int)> opY) -> bool {
            int count = 1;
            for(int n=1; n < 7; ++n) {
                if (!isOwnStone(x,y,opX(n),opY(n)))
                    break;
                ++count;
            }
            return count >= 4;
        };

        int emptyCount = 0;
        history.push_back(board);
        board[getXY(act.y, act.x)] = Figure(idxAi);

        for (int y = 0; y < 6; ++y) {
            for (int x = 0; x < 7; ++x) {
                if (board[getXY(y, x)] == Figure::Unset)
                    ++emptyCount;
                if (board[getXY(y, x)] != idxAi)
                    continue;

                if (scanLine(x,y,opNul,opPos))
                    finished[idxAi] = true;
                if (scanLine(x,y,opPos,opNul))
                    finished[idxAi] = true;
                if (scanLine(x,y,opPos,opPos))
                    finished[idxAi] = true;
                if (scanLine(x,y,opNeg,opPos))
                    finished[idxAi] = true;
            }
        }

        if (emptyCount == 0 && finished[0] == false && finished[1] == false)
            finished[0] = finished[1] = true; // even

        ++time;
    }

    void getGameStateDNN(std::vector<float>& data, int idxMe) const {
        const int T = 4;
        const int p1_piece_start = 0;
        const int p1_piece_count = T * 6 * 7;
        const int p2_piece_start = p1_piece_start + p1_piece_count;
        const int p2_piece_count = T * 6 * 7;
        const int color_start = p2_piece_start + p2_piece_count;
        const int color_count = 6 * 7;

        data.resize(color_start+color_count, 0.0);

        int t = 0;
        int idxOp = (idxMe + 1) % 2;
        const Board* game = &board;
        while (t != T && time-t>=0) {
            for (int y = 0; y < 6; ++y) {
                for (int x = 0; x < 7; ++x) {
                    if (game->board[getXY(y, x)] == idxMe) {
                        data[0*6*7+t*6*7+getXY(y, x)] = 1.0;
                    }
                    if (game->board[getXY(y, x)] == idxOp) {
                        data[T*6*7+t*6*7+getXY(y, x)] = 1.0;
                    }
                }
            }

            ++t;
            if (time-t>=0)
                game = &history[history.size()-t];
        }
        std::fill(data.data()+color_start, data.data()+color_start+color_count, getPlayer(time));
    }

    void getPolicyTrainDNN(std::vector<float>& data, int idxMe, std::vector<std::pair<ActType, double> >& piAction) const {
        data.resize(6*7, 0.0);
        for (size_t i = 0; i < piAction.size(); ++i) {
            double pi = piAction[i].second;
            ActType& act = piAction[i].first;
            data[getXY(act.y, act.x)] = pi;
        }
    }

    void storeGamePolicyDNN(std::vector<float>& game, std::vector<float>& policy) const {
        //connect socket
        zmq::socket_t socket(zmq_context, ZMQ_REQ);
        socket.connect("tcp://localhost:5557");

        //send game
        zmq::message_t request1(game.size()*sizeof(float));
        memcpy(request1.data(), game.data(), game.size()*sizeof(float));
        socket.send(request1);

        //get the reply
        char ok[2];
        zmq::message_t reply;
        socket.recv(&reply);
        memcpy(ok, reply.data(), 2);
        if (ok[0] != 4 || ok[1] != 2)
            throw std::runtime_error("Could not store gamestate");

        //send game
        zmq::message_t request2(policy.size()*sizeof(float));
        memcpy(request2.data(), policy.data(), policy.size()*sizeof(float));
        socket.send(request2);

        //get the reply
        socket.recv(&reply);
        memcpy(ok, reply.data(), 2);
        if (ok[0] != 4 || ok[1] != 2)
            throw std::runtime_error("Could not store gamepolicy");
    }

    //! Interface, Compute W and P values for MCTS
    //! * \param idxMe ID of player who executes function
    void computeMCTS_WP(int idxMe, ActType* actions, ActCounterType nActions, double* P, double& W) const {
        std::vector<float> state_dnn;
        getGameStateDNN(state_dnn, idxMe);

        //connect socket
        zmq::socket_t socket(zmq_context, ZMQ_REQ);
        socket.connect(ports[idxMe]);

        //send request
        zmq::message_t request(state_dnn.size()*sizeof(float));
        memcpy(request.data(), state_dnn.data(), state_dnn.size()*sizeof(float));
        socket.send(request);

        //get the reply
        zmq::message_t reply;
        socket.recv(&reply);
        std::vector<float> result(reply.size() / sizeof(float));
        memcpy(result.data(), reply.data(), reply.size());
        if (result.size() != 6*7+1)
            throw std::runtime_error("Bad Reply");
        socket.close();

        W = result[6*7];
        double pi_sum = 0;
        for (ActCounterType i = 0; i < nActions; ++i) {
            ActType& act = actions[i];
            P[i] = exp(result[getXY(act.y, act.x)]); //softmax
            pi_sum += P[i];
        }
        // apply softmax on valid actions
        for (ActCounterType i = 0; i < nActions; ++i) {
            P[i] /= pi_sum;
        }
    }

    double computeMCTS_W(int) const {
        throw std::runtime_error("Unimplemented");
    }

    std::string getEndOfGameString() const {
        if (finished[0] && finished[1])
            return std::string("Even!");
        if (finished[0])
            return std::string("White Wins!");
        if (finished[1])
            return std::string("Black Wins!");
        return std::string("Error");
    }

    std::string getBoardDescription() const {
        const char figstr[3] = {'O', 'X', ' '};
        std::stringstream str;
        for (int y = 5; y >= 0; --y) { // from top to bottom
            for(int x = 0; x < 7; ++x) { // left to right
                str << "| " << figstr[board[getXY(y, x)]] << " ";
            }
            str << "|" << std::endl;
        }
        return str.str();
    }

    ///! Interface, convert act to string
    static std::string act2str(ActType& act) {
        return static_cast<std::string>(act);
    }
};

#endif // CONNECT4_HPP
