#ifndef CHESS_HPP
#define CHESS_HPP

#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

#include <zmq.hpp>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

//! Stores the state of the figures on the board
/*!
 * \details This class implements the function getPossibleActions for applying AI.
 *          This function returns those actions, which can be played in the next turn without breaking the rules.
 *          Other functionalities are helping to simulate a game and to get the an evaluation of the current state.
 * \author adamp87
*/
class Chess {
public:
    //!< interface
    struct ActType {
        enum Type { Normal, Castling, EnPassant, PromoteQ, PromoteR, PromoteB, PromoteK, CheckMate, Even };
        Type type;
        std::int_fast8_t toX;
        std::int_fast8_t toY;
        std::int_fast8_t fromX;
        std::int_fast8_t fromY;

        CUDA_CALLABLE_MEMBER ActType():type(Normal),toX(0),toY(0),fromX(0),fromY(0) {}
        CUDA_CALLABLE_MEMBER ActType(int fromX,int fromY,int toX,int toY,Type type=Normal):type(type),toX(toX),toY(toY),fromX(fromX),fromY(fromY) {}

        bool operator==(const ActType& other) const {
            return type == other.type &&
                   toX == other.toX &&
                   toY == other.toY &&
                   fromX == other.fromX &&
                   fromY == other.fromY;
        }
        operator std::string() const {
            std::string str("XXXX");
            str[0] = char(fromX)+'A';
            str[1] = char(fromY)+'1';
            str[2] = char(toX)+'A';
            str[3] = char(toY)+'1';
            return str;
        }
    };

    typedef std::uint_fast8_t ActCounterType; //!< interface
    constexpr static double UCT_C = 1.0; //!< interface, constant for exploration in uct formula
    constexpr static double DirichletAlpha = 0.3; //!< interface
    constexpr static unsigned int MaxActions = 218; //!< interface: https://chess.stackexchange.com/questions/4490/maximum-possible-movement-in-a-turn
    constexpr static unsigned int MaxChildPerNode = MaxActions; //!< interface

private:
    struct Figure {
        enum Type { Unset=0, Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6 };
        Type type;
        int playerIdx;
        std::int_fast16_t firstMoved;

        Figure() : type(Unset) { }

        bool operator==(const Figure& other) const {
            return type == other.type && playerIdx == other.playerIdx && firstMoved == other.firstMoved;
        }
    };
    struct FigureSparse : public Figure {
        std::int_fast8_t posX;
        std::int_fast8_t posY;

        bool operator==(const FigureSparse& other) const {
            return posX == other.posX && posY == other.posY &&
                    static_cast<Figure>(*this) == static_cast<Figure>(other);
        }
    };
    struct StateSparse {
        FigureSparse figures[2*16];

        FigureSparse& operator[](int idx) {
            return figures[idx];
        }
        const FigureSparse& operator[](int idx) const {
            return figures[idx];
        }
        bool operator==(const StateSparse& other) const {
            for (int i = 0; i < 2*16; ++i) {
                if (!(figures[i] == other[i]))
                    return false;
            }
            return true;
        }
    };

    StateSparse figures; //!< Sparse representation of chess figures
    std::int_fast16_t time; //!< current turn of the game
    std::int_fast16_t timeLastProgress; //!< last time when figure was taken or pawn moved
    std::vector<StateSparse> history; //!< previous relevant board states, current state should not be part of it

    std::string ports[2]; //!< ports for zeromq socket connections: white, black
    zmq::context_t& zmq_context; //!< zeromq context for socket connections

    int repetitions(const StateSparse& figs, int t_skip=0) const {
        // note: moves are not checked, only if board has the same state
        int count = 0;
        for (auto it = history.rbegin()+t_skip; it != history.rend(); ++it) {
            const StateSparse& hist = *it;
            if (hist == figs)
                count += 1;
        }
        return count;
    }

public:
    //! Set initial state
    Chess(zmq::context_t& zmq_context, const std::string& portW, const std::string& portB)
        : zmq_context(zmq_context)
    {
        ports[0] = portW;
        ports[1] = portB;
        time = 0;
        timeLastProgress = 0;
        for (int idxAi = 0; idxAi < 2; ++idxAi) {
            figures[0+16*idxAi].posX = 4;
            figures[0+16*idxAi].type = Figure::King;
            figures[1+16*idxAi].posX = 3;
            figures[1+16*idxAi].type = Figure::Queen;
            figures[2+16*idxAi].posX = 0;
            figures[2+16*idxAi].type = Figure::Rook;
            figures[3+16*idxAi].posX = 7;
            figures[3+16*idxAi].type = Figure::Rook;
            figures[4+16*idxAi].posX = 1;
            figures[4+16*idxAi].type = Figure::Knight;
            figures[5+16*idxAi].posX = 6;
            figures[5+16*idxAi].type = Figure::Knight;
            figures[6+16*idxAi].posX = 2;
            figures[6+16*idxAi].type = Figure::Bishop;
            figures[7+16*idxAi].posX = 5;
            figures[7+16*idxAi].type = Figure::Bishop;
            for (int i = 0; i < 8; ++i) {
                figures[i+16*idxAi].firstMoved = 0;
                figures[i+16*idxAi].playerIdx = idxAi;
                figures[i+16*idxAi].posY = idxAi==0?0:7;
            }
            for(int i = 0; i < 8; ++i) {
                figures[i+8+16*idxAi].firstMoved = 0;
                figures[i+8+16*idxAi].playerIdx = idxAi;
                figures[i+8+16*idxAi].posX = i;
                figures[i+8+16*idxAi].posY = idxAi==0?1:6;
                figures[i+8+16*idxAi].type = Figure::Pawn;
            }
        }
    }

    //! Interface
    CUDA_CALLABLE_MEMBER int getPlayer(int time=-1) const {
        if (time == -1)
            time = this->time;
        return time%2;
    }

    //! Interface
    CUDA_CALLABLE_MEMBER bool isFinished() const {
        bool kingw = figures[ 0].type == Figure::Unset;
        bool kingb = figures[16].type == Figure::Unset;
        return kingb || kingw;
    }

    //! Interface, Implements game logic, return the possible actions that idxAi can play
    /*!
    * \param idxMe ID of player who executes function
    * \param idxAi ID of player to get possible actions for
    * \param actions Allocated array to store possible actions
    * \return Number of possible actions
    */
    CUDA_CALLABLE_MEMBER ActCounterType getPossibleActions(int idxMe, int idxAi, ActType* actions, bool checkKing=true) const {
        // inspired by: https://codereview.stackexchange.com/questions/173656/chess-game-in-c
        //NOTE: cuda does not like lambda functions, should be refactored to build for cuda
        Figure board[8*8];
        ActCounterType nActions = 0;

        auto getPos = [&] (int x, int y, int dx, int dy) -> int { return (y+dy)*8+x+dx; };
        auto isInsideBoard = [&] (int x, int y, int dx, int dy) -> bool { int xx=x+dx; int yy=y+dy; return 0<=xx && xx<8 && 0<=yy && yy<8; };
        auto isFree = [&] (int x, int y, int dx, int dy) -> bool { return isInsideBoard(x,y,dx,dy) && board[getPos(x,y,dx,dy)].type == Figure::Unset; };
        auto isOpponent = [&] (int x, int y, int dx, int dy) -> bool { return isInsideBoard(x,y,dx,dy) && board[getPos(x,y,dx,dy)].type != Figure::Unset && board[getPos(x,y,dx,dy)].playerIdx != idxAi; };
        auto isEnPassant = [&] (int x, int y, int dx, int dy) -> bool { return isOpponent(x,y,dx,dy) && board[getPos(x,y,dx,dy)].type == Figure::Pawn && board[getPos(x,y,dx,dy)].firstMoved == time; };

        auto isKingInCheck = [&] (int x, int y, int dx, int dy, ActType::Type type=ActType::Normal) -> bool {
            if (!checkKing)
                return false; // block recursion
            Chess copy(*this);
            ActType testAction(x,y,x+dx,y+dy,type);
            copy.update(testAction);
            ActType copyactions[MaxActions];
            ActCounterType copynActions = copy.getPossibleActions(copy.getPlayer(), copy.getPlayer(), copyactions, false);
            for (int i = 0; i < copynActions; ++i) {
                int king = idxAi * 16;
                if (copyactions[i].toX == copy.figures[king].posX && copyactions[i].toY == copy.figures[king].posY) {
                    // opponent can hit king
                    return true;
                }
            }
            return false;
        };

        auto addMove = [&] (int x, int y, int dx, int dy, ActType::Type type=ActType::Normal) -> void {
            if(isFree(x,y,dx,dy) || isOpponent(x,y,dx,dy)) {
                if (isKingInCheck(x, y, dx, dy, type))
                    return;

                actions[nActions++] = ActType(x,y,x+dx,y+dy,type);
            }
        };
        auto addPromote = [&] (int x, int y, int dx, int dy) -> void {
            addMove(x,y,dx,dy,ActType::PromoteK);
            addMove(x,y,dx,dy,ActType::PromoteB);
            addMove(x,y,dx,dy,ActType::PromoteR);
            addMove(x,y,dx,dy,ActType::PromoteQ);
        };

        auto opNul = [&] (int    ) -> int { return    0; };
        auto opPos = [&] (int val) -> int { return  val; };
        auto opNeg = [&] (int val) -> int { return -val; };
        auto scanLine = [&] (int x, int y, std::function<int (int)> opX, std::function<int (int)> opY) -> void {
            for(int n=1; n < 8; ++n) {
                if (isFree(x,y,opX(n),opY(n))) {
                    addMove(x,y,opX(n),opY(n));
                    continue;
                }
                if (isOpponent(x,y,opX(n),opY(n))) {
                    addMove(x,y,opX(n),opY(n));
                    break;
                }
                break;
            }
        };

        auto castlingL = [&] (int x, int y) -> bool {
            bool isKingMoved = figures[idxAi*16  ].type != Figure::King || figures[idxAi*16  ].firstMoved != 0;
            bool isRookMoved = figures[idxAi*16+2].type != Figure::Rook || figures[idxAi*16+2].firstMoved != 0;
            return (!isKingMoved && !isRookMoved &&
                    isFree(x, y, -1, 0) && isFree(x, y, -2, 0) && isFree(x, y, -3, 0) &&
                    !isKingInCheck(x, y, 0, 0) && !isKingInCheck(x, y, -1, 0) && !isKingInCheck(x, y, -2, 0));
        };

        auto castlingR = [&] (int x, int y) -> bool {
            bool isKingMoved = figures[idxAi*16  ].type != Figure::King || figures[idxAi*16  ].firstMoved != 0;
            bool isRookMoved = figures[idxAi*16+3].type != Figure::Rook || figures[idxAi*16+3].firstMoved != 0;
            return (!isKingMoved && !isRookMoved &&
                    isFree(x, y, +1, 0) && isFree(x, y, +2, 0) &&
                    !isKingInCheck(x, y, 0, 0) && !isKingInCheck(x, y, +1, 0) && !isKingInCheck(x, y, +2, 0));
        };

        // check repetitions count
        if (repetitions(figures) == 3) {
            int king = idxAi * 16;
            int x = figures[king].posX;
            int y = figures[king].posY;
            actions[0] = ActType(x,y,x,y,ActType::Even);
            return 1;
        }

        // no progress have been made (50 turn rule)
        if (time-timeLastProgress >= 100) {
            int king = idxAi * 16;
            int x = figures[king].posX;
            int y = figures[king].posY;
            actions[0] = ActType(x,y,x,y,ActType::Even);
            return 1;
        }

        // set up dense board
        for(int i = 0; i < 16*2; ++i) {
            if (figures[i].type == Figure::Unset)
                continue;
            int x = figures[i].posX;
            int y = figures[i].posY;
            board[getPos(x,y,0,0)] = figures[i];
        }

        // for each figure of current ai
        for(int i = idxAi*16; i < (idxAi+1)*16; ++i) {
            int x = figures[i].posX;
            int y = figures[i].posY;
            switch(figures[i].type){
                case Figure::Type::Pawn:
                    // black pawn
                    if(idxAi==1 && y==6 && isFree(x,y,0,-1) && isFree(x,y,0,-2)) addMove(x,y,0,-2);
                    if(idxAi==1 && isFree(x,y,0,-1)) {if (y!=1) addMove(x,y,0,-1); else addPromote(x,y,0,-1);}
                    if(idxAi==1 && isOpponent(x,y,-1,-1)) {if (y!=1) addMove(x,y,-1,-1); else addPromote(x,y,-1,-1);}
                    if(idxAi==1 && isOpponent(x,y,1,-1)) {if (y!=1) addMove(x,y,1,-1); else addPromote(x,y,1,-1);}
                    if(idxAi==1 && y==3 && isEnPassant(x,y,1,0)) addMove(x,y,1,-1,ActType::EnPassant);
                    if(idxAi==1 && y==3 && isEnPassant(x,y,-1,0)) addMove(x,y,-1,-1,ActType::EnPassant);
                    // white pawn
                    if(idxAi==0 && y==1 && isFree(x,y,0,1) && isFree(x,y,0,2)) addMove(x,y,0,2);
                    if(idxAi==0 && isFree(x,y,0,1)) {if (y!=6) addMove(x,y,0,1); else addPromote(x,y,0,1);}
                    if(idxAi==0 && isOpponent(x,y,-1,1)) {if (y!=6) addMove(x,y,-1,1); else addPromote(x,y,-1,1);}
                    if(idxAi==0 && isOpponent(x,y,1,1)) {if (y!=6) addMove(x,y,1,1); else addPromote(x,y,1,1);}
                    if(idxAi==0 && y==4 && isEnPassant(x,y,1,0)) addMove(x,y,1,1,ActType::EnPassant);
                    if(idxAi==0 && y==4 && isEnPassant(x,y,-1,0)) addMove(x,y,-1,1,ActType::EnPassant);
                    break;

                case Figure::Type::Knight:
                    addMove(x,y,-2,-1); addMove(x,y,-2,1); addMove(x,y,2,-1); addMove(x,y,2,1);
                    addMove(x,y,-1,-2); addMove(x,y,-1,2); addMove(x,y,1,-2); addMove(x,y,1,2);
                    break;

                case Figure::Type::King:
                    for(auto dy : {-1,0,1})
                        for(auto dx : {-1,0,1})
                            addMove(x,y,dy,dx);
                    if (castlingL(x, y)) addMove(x,y,-2,0,ActType::Castling);
                    if (castlingR(x, y)) addMove(x,y,+2,0,ActType::Castling);
                    break;

                case Figure::Type::Rook:
                    scanLine(x,y,opNul,opPos);
                    scanLine(x,y,opNul,opNeg);
                    scanLine(x,y,opPos,opNul);
                    scanLine(x,y,opNeg,opNul);
                    break;

                case Figure::Type::Bishop:
                    scanLine(x,y,opPos,opPos);
                    scanLine(x,y,opPos,opNeg);
                    scanLine(x,y,opNeg,opPos);
                    scanLine(x,y,opNeg,opNeg);
                    break;

                case Figure::Type::Queen:
                    scanLine(x,y,opNul,opPos);
                    scanLine(x,y,opNul,opNeg);
                    scanLine(x,y,opPos,opNul);
                    scanLine(x,y,opNeg,opNul);
                    scanLine(x,y,opPos,opPos);
                    scanLine(x,y,opPos,opNeg);
                    scanLine(x,y,opNeg,opPos);
                    scanLine(x,y,opNeg,opNeg);
                    break;

                case Figure::Type::Unset:
                    break;
            }
        }

        if (nActions == 0 && checkKing) {
            //checkmate or even
            int king = idxAi * 16;
            bool checkmate = false;
            nActions = getPossibleActions(idxMe, getPlayer(time+1), actions, false);
            for (int i = 0; i < nActions; ++i) {
                if (actions[i].toX == figures[king].posX && actions[i].toY == figures[king].posY) {
                    // opponent can hit king
                    checkmate = true;
                }
            }
            int x = figures[king].posX;
            int y = figures[king].posY;
            if (checkmate) {
                actions[0] = ActType(x,y,x,y,ActType::CheckMate);
            } else {
                actions[0] = ActType(x,y,x,y,ActType::Even);
            }
            return 1;
        }

        return nActions;
    }

    //! Interface, Update the game state according to move
    CUDA_CALLABLE_MEMBER void update(ActType& act) {
        int idxAi = getPlayer(time);
        int idxOp = getPlayer(time+1);
        history.push_back(figures);
        for(int i = idxAi*16; i < 16*(idxAi+1); ++i) {
            if (figures[i].type != Figure::Unset && figures[i].posX == act.fromX && figures[i].posY == act.fromY) {
                figures[i].posX = act.toX;
                figures[i].posY = act.toY;
                if (figures[i].type == Figure::Pawn)
                    timeLastProgress = time;
                if (figures[i].firstMoved == 0)
                    figures[i].firstMoved = time+1;
                for(int i = idxOp*16; i < 16*(idxOp+1); ++i) { // remove opponent figure
                    if (figures[i].type != Figure::Unset && figures[i].posX == act.toX && figures[i].posY == act.toY) {
                        figures[i].type = Figure::Unset;
                        timeLastProgress = time;
                    }
                }
                switch (act.type) {
                case ActType::Normal:
                    break;
                case ActType::Castling:
                    // move rook
                    figures[idxAi*16+((act.toX<act.fromX)?2:3)].posX = act.toX+((act.toX<act.fromX)?1:-1);
                    figures[idxAi*16+((act.toX<act.fromX)?2:3)].firstMoved = time+1;
                    break;
                case ActType::EnPassant:
                    for(int i = idxOp*16+8; i < 16*(idxOp+1); ++i) { // remove opponent pawn
                        if (figures[i].posX == act.toX && figures[i].posY == act.fromY) {
                            figures[i].type = Figure::Unset;
                        }
                    }
                    break;
                case ActType::PromoteK:
                    figures[i].type = Figure::Knight;
                    break;
                case ActType::PromoteB:
                    figures[i].type = Figure::Bishop;
                    break;
                case ActType::PromoteR:
                    figures[i].type = Figure::Rook;
                    break;
                case ActType::PromoteQ:
                    figures[i].type = Figure::Queen;
                    break;
                case ActType::CheckMate:
                    // checkmate is updated at opponents turn
                    figures[idxAi*16].type = Figure::Unset;
                    break;
                case ActType::Even:
                    figures[idxAi*16].type = Figure::Unset;
                    figures[idxOp*16].type = Figure::Unset;
                    break;
                }
            }
        }
        ++time;
    }

    void getGameStateDNN(std::vector<float>& data, int idxMe) const {
        const int T = 8;
        const int p1_piece_start = 0;
        const int p1_piece_count = 6 * T * 8 * 8;
        const int p2_piece_start = p1_piece_start + p1_piece_count;
        const int p2_piece_count = 6 * T * 8 * 8;
        const int repetition_start = p2_piece_start + p2_piece_count;
        const int repetition_count = 2 * T * 8 * 8;
        const int color_start = repetition_start + repetition_count;
        const int color_count = 8 * 8;
        const int movecount_start = color_start + color_count;
        const int movecount_count = 8 * 8;
        const int p1_castlingL_start = movecount_start + movecount_count;
        const int p1_castlingL_count = 8 * 8;
        const int p1_castlingR_start = p1_castlingL_start + p1_castlingL_count;
        const int p1_castlingR_count = 8 * 8;
        const int p2_castlingL_start = p1_castlingR_start + p1_castlingR_count;
        const int p2_castlingL_count = 8 * 8;
        const int p2_castlingR_start = p2_castlingL_start + p2_castlingL_count;
        const int p2_castlingR_count = 8 * 8;
        const int noactioncount_start = p2_castlingR_start + p2_castlingR_count;
        const int noactioncount_count = 8 * 8;

        auto fill_player_piece = [&] (const StateSparse& figs, int idxPP, int p_piece_start, int t) -> void {
            for (int i = idxPP*16; i < (idxPP+1)*16; ++i) {
                const FigureSparse& fig = figs[i];
                if (fig.type == Figure::Type::Unset)
                    continue;

                // flip board
                int posY = fig.posY;
                if (idxMe == 1 && idxPP == 1)
                    posY = 7-posY;
                if (idxMe == 1 && idxPP == 0)
                    posY = -posY+7;

                data[p_piece_start + t*8*8*6 + (int(fig.type)-1)*8*8 + posY*8+fig.posX] = 1.0f;
            }
        };

        const int data_size = noactioncount_start + noactioncount_count;
        data.resize(data_size, 0.0f);

        int t = 0;
        int idxOp = (idxMe + 1) % 2;
        const StateSparse* figs = &figures;
        while (t != T && time-t>=0) {
            fill_player_piece(*figs, idxMe, p1_piece_start, t);
            fill_player_piece(*figs, idxOp, p2_piece_start, t);

            int count = repetitions(*figs, t);
            std::fill(data.data()+repetition_start+ t*8*8,
                      data.data()+repetition_start+ t*8*8 + std::min<int>(count,2)*8*8,
                      1.0f);

            ++t;
            if (time-t>=0)
                figs = &history[history.size()-t];
        }

        float p1castleL = 0.0f;
        float p1castleR = 0.0f;
        float p2castleL = 0.0f;
        float p2castleR = 0.0f;
        {
            ActCounterType nActions;
            ActType actions[MaxActions];
            nActions = getPossibleActions(idxMe, idxMe, actions);
            for (ActCounterType i = 0; i < nActions; ++i) {
                ActType& action = actions[i];
                if (action.type != ActType::Castling)
                    continue;
                if (action.fromX < action.toX)
                    p1castleR = 1.0f;
                else
                    p1castleL = 1.0f;
            }
            nActions = getPossibleActions(idxMe, idxOp, actions);
            for (ActCounterType i = 0; i < nActions; ++i) {
                ActType& action = actions[i];
                if (action.type != ActType::Castling)
                    continue;
                if (action.fromX < action.toX)
                    p2castleR = 1.0f;
                else
                    p2castleL = 1.0f;
            }
        }

        std::fill(data.data()+color_start, data.data() + color_start+color_count, getPlayer(time));
        std::fill(data.data()+movecount_start, data.data() + movecount_start+movecount_count, time/200.0f);
        std::fill(data.data()+p1_castlingL_start, data.data() + p1_castlingL_start+p1_castlingL_count, p1castleL);
        std::fill(data.data()+p1_castlingR_start, data.data() + p1_castlingR_start+p1_castlingR_count, p1castleR);
        std::fill(data.data()+p2_castlingL_start, data.data() + p2_castlingL_start+p2_castlingL_count, p2castleL);
        std::fill(data.data()+p2_castlingR_start, data.data() + p2_castlingR_start+p2_castlingR_count, p2castleR);
        std::fill(data.data()+noactioncount_start, data.data() + noactioncount_start+noactioncount_count, (time-timeLastProgress)/100.0f);
    }

    void getPolicyTrainDNN(std::vector<float>& data, int idxMe, std::vector<std::pair<ActType, double> >& piAction) const {
        data.resize(8*8, 0.0);
        for (size_t i = 0; i < piAction.size(); ++i) {
            const ActType& action = piAction[i].first;
            float pi = static_cast<float>(piAction[i].second);

            int fromY = action.fromY;
            if (idxMe == 1)
                fromY = 7-fromY;// flip board
            data[fromY*8+action.fromX] += pi;
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
    CUDA_CALLABLE_MEMBER void computeMCTS_WP(int idxMe, ActType* actions, ActCounterType nActions, double* P, double& W) const {

        std::vector<float> state_dnn;
        getGameStateDNN(state_dnn, idxMe);

        if (ports[idxMe] == "0") {
            // compute W based on figure count, no dnn
            W = computeMCTS_W(idxMe);
            for (ActCounterType i = 0; i < nActions; ++i) {
                int fromY = actions[i].fromY;
                if (idxMe == 1)
                    fromY = 7-fromY;// flip board
                P[i] = 1.0;
            }
            return;
        }

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
        if (result.size() != 65)
            throw std::runtime_error("Bad Reply");
        socket.close();

        W = result[64];
        double pi_sum = 0;
        for (ActCounterType i = 0; i < nActions; ++i) {
            int fromY = actions[i].fromY;
            if (idxMe == 1)
                fromY = 7-fromY;// flip board
            P[i] = exp(result[fromY*8+actions[i].fromX]); //softmax
            pi_sum += P[i];
        }
        // apply softmax on valid actions
        for (ActCounterType i = 0; i < nActions; ++i) {
            P[i] /= pi_sum;
        }
    }

    //! Interface, Compute win value for MCTreeSearch, between 0-1
    //! * \param idxAi ID of player who executes function
    CUDA_CALLABLE_MEMBER double computeMCTS_W(int idxAi) const {
        const double figureValue[] = {0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 4.0};
        double aiWin = 0.0;
        double opWin = 0.0;
        int idxOp = (idxAi+1)%2;
        for (int i = 0; i < 16; ++i)
        { // compute weighted sum of figure-points for each player
            aiWin+=figureValue[figures[idxAi*16+i].type];
            opWin+=figureValue[figures[idxOp*16+i].type];
        }
        // win is the ratio between current players points
        // divided with the total number of points on the board
        // NOTE: if player has more points in tends to swap figures of same value
        // NOTE: if player has less points in tends not to change figures of same value
        double win = aiWin/(aiWin+opWin);
        if (figures[idxAi*16].type == Figure::Unset && figures[idxOp*16].type == Figure::Unset)
            return 0.5; // even
        if (figures[idxAi*16].type == Figure::Unset)
            return 0.0; // lost
        if (figures[idxOp*16].type == Figure::Unset)
            return 1.0; // won
        return win;

        //Discussion: during tree search, the algorithm always tries to maximise the win value.
        //When actions of opponents are traversed, the algorithm will rather visit actions,
        //that are good for him and not what good for the opponent is.
        //This means, if opponent can hit a figure,
        //it will rather visit moves which does not hit the figure.
        //For this reason, during tree search it was alternating the UCB value,
        //because opponent is trying to minimize current players win.
        //DNN based W computation will not require this alternation.
        //It is not good to alternate directly here the win value,
        //when computing for opponent it will increase the win value
        //for current player when it loses a figure.
        //if (idxAi == getPlayer(time+1))
        //    return win;
        //else
        //    return 1.0-win; // opponent is trying to minimalize win rate of current player
    }

    std::string getEndOfGameString() const {
        if (figures[0].type == Figure::Unset && figures[16].type == Figure::Unset)
            return std::string("Even!");
        if (figures[0].type == Figure::Unset)
            return std::string("Black Wins!");
        if (figures[16].type == Figure::Unset)
            return std::string("White Wins!");
        return std::string("Error");
    }

    std::string getActionDescription(const ActType& act) const {
        const char figs[] = {'U', 'P', 'k', 'B', 'R', 'Q', 'K'};
        Figure::Type t1 = Figure::Unset;
        Figure::Type t2 = Figure::Unset;
        for (int i = 0; i < 16*2; ++i) {
            if (figures[i].type == Figure::Unset)
                continue;
            if (act.fromX == figures[i].posX && act.fromY == figures[i].posY)
                t1 = figures[i].type;
            if (act.toX == figures[i].posX && act.toY == figures[i].posY)
                t2 = figures[i].type;
        }
        std::string str("X2X");
        str[0] = figs[t1];
        str[2] = figs[t2];
        return str;
    }

    std::string getBoardDescription() const {
        const char figs[] = {'U', 'P', 'k', 'B', 'R', 'Q', 'K'};
        std::string str("________________________________");
        for (int i = 0; i < 32; ++i) {
            if (figures[i].type == Figure::Unset)
                continue;
            str[i] = figs[figures[i].type];
        }
        return str;
    }

    ///! Interface, convert act to string
    static std::string act2str(ActType& act) {
        return static_cast<std::string>(act);
    }

    ///! Test function
    static bool test_actions() {
        zmq::context_t dummy(1);
        Chess chess(dummy, "", "");
        for(int i = 0; i < 32; ++i) {
            chess.figures[i].type = Figure::Unset;
        }
        chess.time = 2;
        chess.figures[0].type = Figure::King;
        chess.figures[2].type = Figure::Rook;
        chess.figures[3].type = Figure::Rook;
        chess.figures[3].firstMoved = 1;
        chess.figures[8].type = Figure::Pawn;
        chess.figures[8].posY = 4;
        chess.figures[8].firstMoved = 1;
        chess.figures[8+7].type = Figure::Pawn;
        chess.figures[8+7].posX = 6;
        chess.figures[8+7].posY = 6;
        chess.figures[8+7].firstMoved = 1;
        chess.figures[16].type = Figure::King;
        chess.figures[16+3].type = Figure::Rook;
        chess.figures[16+3].firstMoved = 1;
        chess.figures[16+8+4].type = Figure::Pawn;
        chess.figures[16+8+4].posX = 1;
        chess.figures[16+8+4].posY = 4;
        chess.figures[16+8+4].firstMoved = 2;
        ActType actions[Chess::MaxActions];
        ActCounterType nActions = chess.getPossibleActions(0, 0, actions);

        ActType movePromote;
        ActType moveCastling;
        ActType moveEnPassant;
        bool canPromote = false;
        bool canCastling = false;
        bool canEnPassant = false;
        for (int i = 0; i < nActions; ++i) {
            if (actions[i].type == ActType::PromoteQ) {
                canPromote = true;
                movePromote = actions[i];
            }
            if (actions[i].type == ActType::Castling) {
                canCastling = true;
                moveCastling = actions[i];
            }
            if (actions[i].type == ActType::EnPassant) {
                canEnPassant = true;
                moveEnPassant = actions[i];
            }
        }
        if (!canCastling || !canEnPassant || !canPromote)
            return false;

        Chess chessPromote(chess);
        Chess chessCastling(chess);
        Chess chessEnPassant(chess);
        chessPromote.update(movePromote);
        chessCastling.update(moveCastling);
        chessEnPassant.update(moveEnPassant);
        bool donePromote  = chessPromote.figures[8+7].posX == 7 &&
                            chessPromote.figures[8+7].posY == 7 &&
                            chessPromote.figures[8+7].type == Figure::Queen &&
                            chessPromote.figures[16+3].posX == 7 &&
                            chessPromote.figures[16+3].posY == 7 &&
                            chessPromote.figures[16+3].type == Figure::Unset;
        bool doneCastling = chessCastling.figures[0].posX == 2 &&
                            chessCastling.figures[0].posY == 0 &&
                            chessCastling.figures[0].type == Figure::King &&
                            chessCastling.figures[2].posX == 3 &&
                            chessCastling.figures[2].posY == 0 &&
                            chessCastling.figures[2].type == Figure::Rook;
        bool doneEnPassant =chessEnPassant.figures[8].posX == 1 &&
                            chessEnPassant.figures[8].posY == 5 &&
                            chessEnPassant.figures[8].type == Figure::Pawn &&
                            chessEnPassant.figures[16+8+4].posX == 1 &&
                            chessEnPassant.figures[16+8+4].posY == 4 &&
                            chessEnPassant.figures[16+8+4].type == Figure::Unset;
        if (!doneCastling || !doneEnPassant || !donePromote)
            return false;
        return true;
    }

    ///! Test function
    void setDebugBoard(int m) {
        if (m == 0)
            return;
        for(int i = 0; i < 32; ++i) {
            figures[i].type = Figure::Unset;
        }
        if (m == 1) {
            figures[0].type = Figure::King;
            figures[2].type = Figure::Rook;
            figures[3].type = Figure::Rook;
            figures[16+0].type = Figure::King;
            ActType move;
            move = ActType(0,0,5,1);
            update(move);
            move = ActType(4,7,6,7);
            update(move);
            move = ActType(7,0,6,0);
            update(move);
            move = ActType(6,7,7,7);
            update(move);
            //WKing unmoved
            //WRook G1
            //WRook F2
            //BKing H8
            //Checkmate F2H2

#if 0
            move = MoveType(5,1,7,1);
            update(move);

            MoveType moves[Chess::MaxMoves];
            MoveCounterType nMoves = getPossibleMoves(1, 1, moves);
            update(moves[0]);
            bool isGameOver = isFinished();
            nMoves = getPossibleMoves(0, 0, moves);
            return;
#endif
        }

        if (m == 2) {
            // https://index.hu/sport/sakk/2019/03/02/sakkfeladvany_otodik_resz
            figures[0].type = Figure::King;
            figures[4].type = Figure::Knight;
            figures[16+0].type = Figure::King;
            figures[16+8].type = Figure::Pawn;
            ActType move;
            move = ActType(4,0,5,0);
            update(move);
            move = ActType(4,7,7,0);
            update(move);
            move = ActType(1,0,4,4);
            update(move);
            move = ActType(0,6,7,2);
            update(move);
        }

        if (m == 3) {
            // https://index.hu/sport/sakk/2019/02/23/sakkfeladvany_negyedik_resz/
            figures[0].type = Figure::King;
            figures[4].type = Figure::Knight;
            figures[16+0].type = Figure::King;
            figures[16+4].type = Figure::Knight;
            figures[16+8+7].type = Figure::Pawn;
            ActType move;
            move = ActType(4,0,5,7);
            update(move);
            move = ActType(4,7,7,7);
            update(move);
            move = ActType(1,0,5,4);
            update(move);
            move = ActType(1,7,7,5);
            update(move);
        }

        if (m == 4) {
            // https://index.hu/sport/sakk/2019/02/16/sakkfeladvany_3._resz/
            figures[0].type = Figure::King;
            figures[0].posX = 5;
            figures[0].posY = 4;
            figures[0].firstMoved = 1;
            figures[2].type = Figure::Rook;
            figures[2].posX = 0;
            figures[2].posY = 3;
            figures[2].firstMoved = 1;
            figures[8].type = Figure::Pawn;
            figures[8].posX = 6;
            figures[8].posY = 2;
            figures[8].firstMoved = 1;

            figures[16+0].type = Figure::King;
            figures[16+0].posX = 7;
            figures[16+0].posY = 4;
            figures[16+0].firstMoved = 2;
            figures[16+1].type = Figure::Queen;
            figures[16+1].posX = 4;
            figures[16+1].posY = 6;
            figures[16+1].firstMoved = 2;
            figures[16+2].type = Figure::Rook;
            figures[16+2].posX = 4;
            figures[16+2].posY = 0;
            figures[16+2].firstMoved = 2;
            figures[16+3].type = Figure::Rook;
            figures[16+3].posX = 7;
            figures[16+3].posY = 0;
            figures[16+3].firstMoved = 2;
            figures[16+4].type = Figure::Knight;
            figures[16+4].posX = 7;
            figures[16+4].posY = 2;
            figures[16+4].firstMoved = 2;
            figures[16+8+6].type = Figure::Pawn;
            figures[16+8+6].posX = 6;
            figures[16+8+6].posY = 4;
            figures[16+8+6].firstMoved = 2;
            figures[16+8+7].type = Figure::Pawn;
            figures[16+8+7].posX = 7;
            figures[16+8+7].posY = 5;
            figures[16+8+7].firstMoved = 2;

            time = 2;
        }

        if (m == 5) {
            // https://index.hu/sport/sakk/2019/03/09/carlsen_lepese_vb-donto/
            figures[0].type = Figure::King;
            figures[0].posX = 7;
            figures[0].posY = 0;
            figures[0].firstMoved = 1;
            figures[1].type = Figure::Queen;
            figures[1].posX = 5;
            figures[1].posY = 3;
            figures[1].firstMoved = 1;
            figures[2].type = Figure::Rook;
            figures[2].posX = 2;
            figures[2].posY = 7;
            figures[2].firstMoved = 1;
            figures[3].type = Figure::Rook;
            figures[3].posX = 5;
            figures[3].posY = 4;
            figures[3].firstMoved = 1;
            figures[8+4].type = Figure::Pawn;
            figures[8+4].posX = 4;
            figures[8+4].posY = 3;
            figures[8+4].firstMoved = 1;
            figures[8+5].type = Figure::Pawn;
            figures[8+5].posX = 5;
            figures[8+5].posY = 2;
            figures[8+5].firstMoved = 1;
            figures[8+6].type = Figure::Pawn;
            figures[8+6].posX = 7;
            figures[8+6].posY = 4;
            figures[8+6].firstMoved = 1;
            figures[8+7].type = Figure::Pawn;
            figures[8+7].posX = 7;
            figures[8+7].posY = 1;
            figures[8+7].firstMoved = 0;

            figures[16+0].type = Figure::King;
            figures[16+0].posX = 7;
            figures[16+0].posY = 6;
            figures[16+0].firstMoved = 2;
            figures[16+1].type = Figure::Queen;
            figures[16+1].posX = 5;
            figures[16+1].posY = 1;
            figures[16+1].firstMoved = 2;
            figures[16+2].type = Figure::Rook;
            figures[16+2].posX = 0;
            figures[16+2].posY = 1;
            figures[16+2].firstMoved = 2;
            figures[16+6].type = Figure::Bishop;
            figures[16+6].posX = 4;
            figures[16+6].posY = 6;
            figures[16+6].firstMoved = 2;
            figures[16+8+1].type = Figure::Pawn;
            figures[16+8+1].posX = 1;
            figures[16+8+1].posY = 5;
            figures[16+8+1].firstMoved = 2;
            figures[16+8+3].type = Figure::Pawn;
            figures[16+8+3].posX = 3;
            figures[16+8+3].posY = 5;
            figures[16+8+3].firstMoved = 2;
            figures[16+8+5].type = Figure::Pawn;
            figures[16+8+5].posX = 5;
            figures[16+8+5].posY = 6;
            figures[16+8+5].firstMoved = 0;
            figures[16+8+6].type = Figure::Pawn;
            figures[16+8+6].posX = 6;
            figures[16+8+6].posY = 6;
            figures[16+8+6].firstMoved = 0;

            time = 2;
        }
    }
};

#endif //CHESS_HPP
