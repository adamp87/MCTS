#ifndef CHESS_HPP
#define CHESS_HPP

#include <string>
#include <vector>

#include <numeric>
#include <algorithm>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

//! Stores the state of the figures on the board
/*!
 * \details This class implements the function getPossibleMoves for applying AI.
 *          This function returns those moves, which can be played in the next turn without breaking the rules.
 *          Other functionalities are helping to simulate a game and to get the an evaluation of the current state.
 * \author adamp87
*/
class Chess {
public:
    //!< interface
    struct MoveType {
        enum Type { Normal, Castling, EnPassant, PromoteQ, PromoteR, PromoteB, PromoteK, CheckMate, Even };
        Type type;
        std::int_fast8_t toX;
        std::int_fast8_t toY;
        std::int_fast8_t fromX;
        std::int_fast8_t fromY;

        CUDA_CALLABLE_MEMBER MoveType():type(Normal),toX(0),toY(0),fromX(0),fromY(0) {}
        CUDA_CALLABLE_MEMBER MoveType(int fromX,int fromY,int toX,int toY,Type type=Normal):type(type),toX(toX),toY(toY),fromX(fromX),fromY(fromY) {}

        bool operator==(const MoveType& other) const {
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

    typedef std::uint_fast8_t MoveCounterType; //!< interface
    constexpr static unsigned int MaxMoves = 218; //!< interface: https://chess.stackexchange.com/questions/4490/maximum-possible-movement-in-a-turn
    constexpr static unsigned int MaxChildPerNode = MaxMoves; //!< interface

private:
    struct Figure {
        enum Type { Unset=0, Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6 };
        Type type;
        int playerIdx;
        std::int_fast16_t movedCount;

        Figure() : type(Unset) { }
    };
    struct FigureSparse : public Figure {
        std::int_fast8_t posX;
        std::int_fast8_t posY;
    };

    std::int_fast16_t time; //!< current turn of the game
    FigureSparse figures[2*16]; //!< Sparse representation of chess figures

public:
    //! Set initial state
    Chess() {
        time = 0;
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
                figures[i+16*idxAi].movedCount = 0;
                figures[i+16*idxAi].playerIdx = idxAi;
                figures[i+16*idxAi].posY = idxAi==0?0:7;
            }
            for(int i = 0; i < 8; ++i) {
                figures[i+8+16*idxAi].movedCount = 0;
                figures[i+8+16*idxAi].playerIdx = idxAi;
                figures[i+8+16*idxAi].posX = i;
                figures[i+8+16*idxAi].posY = idxAi==0?1:6;
                figures[i+8+16*idxAi].type = Figure::Pawn;
            }
        }
    }

    //! Interface
    CUDA_CALLABLE_MEMBER int getPlayer(int time = 255) const {
        if (time == 255)
            time = this->time;
        return time%2;
    }

    //! Interface
    CUDA_CALLABLE_MEMBER bool isFinished() const {
        bool kingw = figures[ 0].type == Figure::Unset;
        bool kingb = figures[16].type == Figure::Unset;
        return kingb || kingw;
    }

    //! Interface, Implements game logic, return the possible moves that idxAi can play
    /*!
    * \param idxMe ID of player who executes function
    * \param idxAi ID of player to get possible moves for
    * \param moves Allocated array to store possible moves
    * \return Number of possible moves
    * \note Using hands[idxAi] is hard-coded cheating
    */
    CUDA_CALLABLE_MEMBER MoveCounterType getPossibleMoves(int idxMe, int idxAi, MoveType* moves, bool checkKing=true) const {
        // inspired by: https://codereview.stackexchange.com/questions/173656/chess-game-in-c
        //TODO: cuda does not like lambda functions, should be refactored to build for cuda
        MoveCounterType nMoves = 0;
        Figure board[8*8];

        auto getPos = [&] (int x, int y, int dx, int dy) -> int { return (y+dy)*8+x+dx; };
        auto isInsideBoard = [&] (int x, int y, int dx, int dy) -> bool { int xx=x+dx; int yy=y+dy; return 0<=xx && xx<8 && 0<=yy && yy<8; };
        auto isFree = [&] (int x, int y, int dx, int dy) -> bool { return isInsideBoard(x,y,dx,dy) && board[getPos(x,y,dx,dy)].type == Figure::Unset; };
        auto isOpponent = [&] (int x, int y, int dx, int dy) -> bool { return isInsideBoard(x,y,dx,dy) && board[getPos(x,y,dx,dy)].type != Figure::Unset && board[getPos(x,y,dx,dy)].playerIdx != idxAi; };
        auto isEnPassant = [&] (int x, int y, int dx, int dy) -> bool { return isOpponent(x,y,dx,dy) && board[getPos(x,y,dx,dy)].type == Figure::Pawn && board[getPos(x,y,dx,dy)].movedCount == 1; };

        auto isKingInCheck = [&] (int x, int y, int dx, int dy, MoveType::Type type=MoveType::Normal) -> bool {
            if (!checkKing)
                return false; // block recursion
            Chess copy(*this);
            MoveType testMove(x,y,x+dx,y+dy,type);
            copy.update(testMove);
            MoveType copymoves[MaxMoves];
            MoveCounterType copynMoves = copy.getPossibleMoves(copy.getPlayer(), copy.getPlayer(), copymoves, false);
            for (int i = 0; i < copynMoves; ++i) {
                int king = idxAi * 16;
                if (copymoves[i].toX == copy.figures[king].posX && copymoves[i].toY == copy.figures[king].posY) {
                    // opponent can hit king
                    return true;
                }
            }
            return false;
        };

        auto addMove = [&] (int x, int y, int dx, int dy, MoveType::Type type=MoveType::Normal) -> void {
            if(isFree(x,y,dx,dy) || isOpponent(x,y,dx,dy)) {
                if (isKingInCheck(x, y, dx, dy, type))
                    return;

                moves[nMoves++] = MoveType(x,y,x+dx,y+dy,type);
            }
        };
        auto addPromote = [&] (int x, int y, int dx, int dy) -> void {
            addMove(x,y,dx,dy,MoveType::PromoteK);
            addMove(x,y,dx,dy,MoveType::PromoteB);
            addMove(x,y,dx,dy,MoveType::PromoteR);
            addMove(x,y,dx,dy,MoveType::PromoteQ);
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
            bool isKingMoved = figures[idxAi*16  ].type != Figure::King || figures[idxAi*16  ].movedCount != 0;
            bool isRookMoved = figures[idxAi*16+2].type != Figure::Rook || figures[idxAi*16+2].movedCount != 0;
            return (!isKingMoved && !isRookMoved &&
                    isFree(x, y, -1, 0) && isFree(x, y, -2, 0) && isFree(x, y, -3, 0) &&
                    !isKingInCheck(x, y, 0, 0) && !isKingInCheck(x, y, -1, 0) && !isKingInCheck(x, y, -2, 0));
        };

        auto castlingR = [&] (int x, int y) -> bool {
            bool isKingMoved = figures[idxAi*16  ].type != Figure::King || figures[idxAi*16  ].movedCount != 0;
            bool isRookMoved = figures[idxAi*16+3].type != Figure::Rook || figures[idxAi*16+3].movedCount != 0;
            return (!isKingMoved && !isRookMoved &&
                    isFree(x, y, +1, 0) && isFree(x, y, +2, 0) &&
                    !isKingInCheck(x, y, 0, 0) && !isKingInCheck(x, y, +1, 0) && !isKingInCheck(x, y, +2, 0));
        };

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
                    if(idxAi==1 && y==3 && isEnPassant(x,y,1,0)) addMove(x,y,1,-1,MoveType::EnPassant);
                    if(idxAi==1 && y==3 && isEnPassant(x,y,-1,0)) addMove(x,y,-1,-1,MoveType::EnPassant);
                    // white pawn
                    if(idxAi==0 && y==1 && isFree(x,y,0,1) && isFree(x,y,0,2)) addMove(x,y,0,2);
                    if(idxAi==0 && isFree(x,y,0,1)) {if (y!=6) addMove(x,y,0,1); else addPromote(x,y,0,1);}
                    if(idxAi==0 && isOpponent(x,y,-1,1)) {if (y!=6) addMove(x,y,-1,1); else addPromote(x,y,-1,1);}
                    if(idxAi==0 && isOpponent(x,y,1,1)) {if (y!=6) addMove(x,y,1,1); else addPromote(x,y,1,1);}
                    if(idxAi==0 && y==4 && isEnPassant(x,y,1,0)) addMove(x,y,1,1,MoveType::EnPassant);
                    if(idxAi==0 && y==4 && isEnPassant(x,y,-1,0)) addMove(x,y,-1,1,MoveType::EnPassant);
                    break;

                case Figure::Type::Knight:
                    addMove(x,y,-2,-1); addMove(x,y,-2,1); addMove(x,y,2,-1); addMove(x,y,2,1);
                    addMove(x,y,-1,-2); addMove(x,y,-1,2); addMove(x,y,1,-2); addMove(x,y,1,2);
                    break;

                case Figure::Type::King:
                    for(auto dy : {-1,0,1})
                        for(auto dx : {-1,0,1})
                            addMove(x,y,dy,dx);
                    if (castlingL(x, y)) addMove(x,y,-2,0,MoveType::Castling);
                    if (castlingR(x, y)) addMove(x,y,+2,0,MoveType::Castling);
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

        if (nMoves == 0 && checkKing) {
            //checkmate or even
            int king = idxAi * 16;
            bool checkmate = false;
            nMoves = getPossibleMoves(idxMe, getPlayer(time+1), moves, false);
            for (int i = 0; i < nMoves; ++i) {
                if (moves[i].toX == figures[king].posX && moves[i].toY == figures[king].posY) {
                    // opponent can hit king
                    checkmate = true;
                }
            }
            int x = figures[king].posX;
            int y = figures[king].posY;
            if (checkmate) {
                moves[0] = MoveType(x,y,x,y,MoveType::CheckMate);
            } else {
                moves[0] = MoveType(x,y,x,y,MoveType::Even);
            }
            return 1;
        }

        return nMoves;
    }

    //! Interface, Update the game state according to move
    CUDA_CALLABLE_MEMBER void update(MoveType& move) {
        int idxAi = getPlayer(time);
        int idxOp = getPlayer(time+1);
        for(int i = idxAi*16; i < 16*(idxAi+1); ++i) {
            if (figures[i].type != Figure::Unset && figures[i].posX == move.fromX && figures[i].posY == move.fromY) {
                figures[i].posX = move.toX;
                figures[i].posY = move.toY;
                figures[i].movedCount += 1;
                for(int i = idxOp*16; i < 16*(idxOp+1); ++i) { // remove opponent figure
                    if (figures[i].posX == move.toX && figures[i].posY == move.toY) {
                        figures[i].type = Figure::Unset;
                    }
                }
                switch (move.type) {
                case MoveType::Normal:
                    break;
                case MoveType::Castling:
                    // move rook
                    figures[idxAi*16+((move.toX<move.fromX)?2:3)].posX = move.toX+((move.toX<move.fromX)?1:-1);
                    figures[idxAi*16+((move.toX<move.fromX)?2:3)].movedCount += 1;
                    break;
                case MoveType::EnPassant:
                    for(int i = idxOp*16+8; i < 16*(idxOp+1); ++i) { // remove opponent pawn
                        if (figures[i].posX == move.toX && figures[i].posY == move.fromY) {
                            figures[i].type = Figure::Unset;
                        }
                    }
                    break;
                case MoveType::PromoteK:
                    figures[i].type = Figure::Knight;
                    break;
                case MoveType::PromoteB:
                    figures[i].type = Figure::Bishop;
                    break;
                case MoveType::PromoteR:
                    figures[i].type = Figure::Rook;
                    break;
                case MoveType::PromoteQ:
                    figures[i].type = Figure::Queen;
                    break;
                case MoveType::CheckMate:
                    // checkmate is updated at opponents turn
                    figures[idxAi*16].type = Figure::Unset;
                    break;
                case MoveType::Even:
                    figures[idxAi*16].type = Figure::Unset;
                    figures[idxOp*16].type = Figure::Unset;
                    break;
                }
            }
        }
        ++time;
    }

    //! Interface, Compute win value for MCTreeSearch, between 0-1
    CUDA_CALLABLE_MEMBER double computeMCTSWin(int idxAi) const {
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
    }

    std::string getMoveDescription(const MoveType& move) const {
        const char figs[] = {'U', 'P', 'k', 'B', 'R', 'Q', 'K'};
        Figure::Type t1 = Figure::Unset;
        Figure::Type t2 = Figure::Unset;
        for (int i = 0; i < 16*2; ++i) {
            if (figures[i].type == Figure::Unset)
                continue;
            if (move.fromX == figures[i].posX && move.fromY == figures[i].posY)
                t1 = figures[i].type;
            if (move.toX == figures[i].posX && move.toY == figures[i].posY)
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

    ///! Interface, convert move to string
    static std::string move2str(MoveType& card) {
        return static_cast<std::string>(card);
    }

    ///! Test function
    static bool test_moves() {
        Chess chess;
        for(int i = 0; i < 32; ++i) {
            chess.figures[i].type = Figure::Unset;
        }
        chess.figures[0].type = Figure::King;
        chess.figures[2].type = Figure::Rook;
        chess.figures[3].type = Figure::Rook;
        chess.figures[3].movedCount = 1;
        chess.figures[8].type = Figure::Pawn;
        chess.figures[8].posY = 4;
        chess.figures[8].movedCount = 1;
        chess.figures[8+7].type = Figure::Pawn;
        chess.figures[8+7].posX = 6;
        chess.figures[8+7].posY = 6;
        chess.figures[8+7].movedCount = 1;
        chess.figures[16].type = Figure::King;
        chess.figures[16+3].type = Figure::Rook;
        chess.figures[16+3].movedCount = 1;
        chess.figures[16+8+4].type = Figure::Pawn;
        chess.figures[16+8+4].posX = 1;
        chess.figures[16+8+4].posY = 4;
        chess.figures[16+8+4].movedCount = 1;
        MoveType moves[Chess::MaxMoves];
        MoveCounterType nMoves = chess.getPossibleMoves(0, 0, moves);

        MoveType movePromote;
        MoveType moveCastling;
        MoveType moveEnPassant;
        bool canPromote = false;
        bool canCastling = false;
        bool canEnPassant = false;
        for (int i = 0; i < nMoves; ++i) {
            if (moves[i].type == MoveType::PromoteQ) {
                canPromote = true;
                movePromote = moves[i];
            }
            if (moves[i].type == MoveType::Castling) {
                canCastling = true;
                moveCastling = moves[i];
            }
            if (moves[i].type == MoveType::EnPassant) {
                canEnPassant = true;
                moveEnPassant = moves[i];
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
            MoveType move;
            move = MoveType(0,0,5,1);
            update(move);
            move = MoveType(4,7,6,7);
            update(move);
            move = MoveType(7,0,6,0);
            update(move);
            move = MoveType(6,7,7,7);
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
    }
};

#endif //CHESS_HPP
