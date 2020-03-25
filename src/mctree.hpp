#ifndef MCTREE_HPP
#define MCTREE_HPP

#include <vector>
#include <memory>

#include <mutex>
#include <atomic>

#include <cstdlib>

template <class TMove>
struct MovesMeM {
    int state;
    std::vector<TMove> moves;

    MovesMeM() : state(0) {}

    bool isUnset() const { return state == 0;}
    bool isEmpty() const { return state == 2;}

    template <typename TMoveCount>
    void init(TMove* _moves, TMoveCount nMoves) {
        moves.insert(moves.end(), _moves, _moves+nMoves);
        state = 1;
    }

    void next(TMove* move) {
        *move = moves.back();
        moves.pop_back();
        if (moves.empty()) {
            state = 2;
        }
    }
};

struct MovesOOC {
    struct TmpFile {
        int bytes;
        FILE* tmpfile;
        std::mutex lock;

        TmpFile() {
            bytes = 0;
            tmpfile = std::tmpfile();
        }

        template <class TMove>
        void read(TMove* move, int& posNext) {
            std::lock_guard<std::mutex> guard(lock); (void)guard;
            std::fseek(tmpfile, posNext, SEEK_SET);
            int read = std::fread(move, sizeof(TMove), 1, tmpfile);
            posNext += read * sizeof(TMove);
        }

        template <class TMove, typename TMoveCount>
        void write(TMove* moves, TMoveCount nMoves, int& posNext, int& posEnd) {
            std::lock_guard<std::mutex> guard(lock); (void)guard;
            posNext = bytes;
            std::fseek(tmpfile, bytes, SEEK_SET);
            int count = std::fwrite(moves, sizeof(TMove), nMoves, tmpfile);
            bytes += count * sizeof(TMove);
            posEnd = bytes;
        }
    };

    int state;
    int posEnd;
    int posNext;
    static TmpFile tmp;

    MovesOOC() : state(0), posEnd(0), posNext(0) {}

    bool isUnset() const { return state == 0;}
    bool isEmpty() const { return state == 2;}

    template <class TMove, typename TMoveCount>
    void init(TMove* moves, TMoveCount nMoves) {
        tmp.write(moves, nMoves, posNext, posEnd);
        state = 1;
    }

    template <class TMove>
    void next(TMove* move) {
        tmp.read(move, posNext);
        if (posNext == posEnd) {
            state = 2;
        }
    }

    template <class TMove, typename TMoveCount>
    void test(TMove* moves, TMoveCount nMoves) {
        int pos = posNext;
        for(unsigned int i = 0; i < nMoves; ++i) {
            TMove move;
            next(&move);
            if (move == moves[i]) {
            } else {
                throw -1;
            }
        }
        posNext = pos;
        state = 1;
    }
};

//! Base class for all Tree::Node implementations
template <typename T_Act>
struct MCTSNodeBase {
    typedef T_Act ActType;
    typedef std::uint_fast32_t CountType;

    //! Dummy LockGuard, helps to keep policy transparent
    class LockGuard {
    public:
        LockGuard(MCTSNodeBase<T_Act>&) {}
    };

    CountType   N;      //!< node visit count
    double      W;      //!< number of wins
    T_Act       action; //!< e.g. card played out
    MovesMeM<T_Act>    nexts;

    MCTSNodeBase(const T_Act& action)
        : N(0), W(0.0), action(action)
    { }
};

//! Base class for multithreaded MCTreeDynamic::Node
template <typename T_Act>
struct MCTSNodeBaseMT {
    typedef T_Act ActType;
    typedef std::uint_fast32_t CountType;

    //! LockGuard inherited from std, constructor takes node
    class LockGuard : public std::lock_guard<std::mutex> {
    public:
        LockGuard(MCTSNodeBaseMT<T_Act>& node)
            : std::lock_guard<std::mutex>(node.lock)
        {}
    };

    //! Workaround, until C20 standard enables atomic operations for double
    class MyAtomicDouble {
        std::atomic<double> value;

    public:
        MyAtomicDouble() : value(0) {}
        operator double() const {
            return value;
        }
        void operator+=(double& val) {
            double prev = value;
            double next = prev+val;
            while (!value.compare_exchange_weak(prev, next)) { }
        }
    };

    std::atomic<CountType>  N;      //!< node visit count
    MyAtomicDouble          W;      //!< number of wins for each points
    T_Act                   action; //!< e.g. card played out
    std::mutex              lock;   //!< mutex for thread-safe policy
    MovesMeM<T_Act>    nexts;

    MCTSNodeBaseMT(const T_Act& action)
        : N(0), W(0.0), action(action)
    { }
};

//! Data container, store nodes detached, store childs as pointers
/*!
 * \details For each node, memory is allocated dynamically.
 *          The root is stored as a node.
 *          The childs are stored as pointers.
 *          Each node stores the pointers in a vector.
 * \author adamp87
*/
template <typename T_NodeBase>
class MCTreeDynamic {
public:
    typedef typename T_NodeBase::ActType ActType;

    //! One node with childs, interface
    class Node : public T_NodeBase {
    private:
        std::vector<std::unique_ptr<Node> > childs; //!< store childs as unique pointers

        friend class ChildIterator;
        friend class MCTreeDynamic;

        Node(const ActType& action) : T_NodeBase(action) {}
        Node(const Node&) = delete;
    };

    typedef Node* NodePtr; //!< pointer to a node, interface

    //! Iterator to get childs of a node, interface
    class ChildIterator {
        size_t pos; //!< position in the child container
        const NodePtr node; //!< get childs of this node

        friend class MCTreeDynamic;

        ChildIterator(NodePtr& node)
            : pos(0), node(node) { }

    public:
        //! Returns true if not all childs have been visited, interface
        bool hasNext() const {
            return node->childs.size() != pos;
        }

        //! Return pointer to next child, interface
        NodePtr next() {
            return NodePtr(node->childs[pos++].get());
        }
    };

private:
    std::unique_ptr<Node> root; //!< root of the tree

public:
    //! Construct tree, interface
    MCTreeDynamic() {
        // artifical root, doesnt hold valid action
        root.reset(new Node(ActType()));
    }

    //! Add a new node to parent, interface
    NodePtr addNode(NodePtr& _parent, const ActType& act) const {
        // init leaf node
        std::unique_ptr<Node> child(new Node(act));

        _parent->childs.push_back(std::move(child));
        return _parent->childs.back().get();
    }

    //! Get pointer to the root, interface
    NodePtr getRoot() const {
        return root.get();
    }

    //! Return an iterator to get childs of node, interface
    ChildIterator getChildIterator(NodePtr& node) const {
        return ChildIterator(node);
    }

    //! Can be used for debug, interface for debug
    size_t getNodeId(NodePtr& node) const {
        return reinterpret_cast<size_t>(node);
    }
};

//! Data container, store nodes contiguously, store childs indices in a fixed length array
/*!
 * \details Nodes are stored contiguously in one vector.
 *          The root is the first node in the vector.
 *          The childs are stored as indices.
 *          Each node stores the indices in fixed length array, bounded by N_MaxChild.
 * \author adamp87
*/
template <typename T_Act, unsigned int N_MaxChild>
class MCTreeStaticArray {
public:
    typedef T_Act ActType;

    //! One node with childs, interface
    class Node : public MCTSNodeBase<ActType> {
        size_t childs[N_MaxChild]; //!< child indices as fixed length array

        friend class ChildIterator;
        friend class MCTreeStaticArray;

        Node(ActType act) : MCTSNodeBase<ActType>(act), childs{0} {}
    };

    //!< pointer to a node, interface
    class NodePtr {
        size_t idx; //!< index of current node
        std::vector<Node>& nodes; //!< reference to tree

        friend class ChildIterator;
        friend class MCTreeStaticArray;

        NodePtr(size_t idx, std::vector<Node>& nodes)
            : idx(idx), nodes(nodes) { }
    public:
        void operator=(const NodePtr& other) {
            idx = other.idx;
        }

        Node* operator->() {
            return &(nodes[idx]);
        }

        const Node* operator->() const {
            return &(nodes[idx]);
        }

        Node& operator*() {
            return nodes[idx];
        }

        const Node& operator*() const {
            return nodes[idx];
        }
    };

    //! Iterator to get childs of a node, interface
    class ChildIterator {
        size_t pos; //!< position of the next child in container
        const Node& node; //!< nodes whos childs are iterated
        std::vector<Node>& nodes; //!< reference to tree

        friend class MCTreeStaticArray;

        ChildIterator(NodePtr& node, std::vector<Node>& nodes)
            : pos(0), node(*node), nodes(nodes) { }

    public:
        //! Returns true if not all childs have been visited, interface
        bool hasNext() const {
            return node.childs[pos] != 0;
        }

        //! Return pointer to next child, interface
        NodePtr next() {
            return NodePtr(node.childs[pos++], nodes);
        }
    };

private:
    std::vector<Node> nodes; //!< array of the nodes, tree

public:
    //! Construct tree, interface
    MCTreeStaticArray() {
        // artifical root, doesnt hold valid action
        ActType dummy = ActType();
        Node root(dummy);
        nodes.push_back(root);
    }

    //! Add a new node to parent, interface
    NodePtr addNode(NodePtr& _parent, ActType act) {
        // init leaf node
        Node child(act);

        // find next free idx of parent to put child
        unsigned int idx = 0;
        Node& parent = *_parent;
        for (; idx < N_MaxChild; ++idx) {
            if (parent.childs[idx] == 0) {
                break;
            }
        }
        if (idx == N_MaxChild)
            std::cerr << "Error in child saving" << std::endl;

        // add leaf to vector and init space for childs
        size_t child_idx = nodes.size();
        parent.childs[idx] = child_idx;
        nodes.push_back(child);
        return NodePtr(child_idx, nodes);
    }

    //! Get pointer to the root, interface
    NodePtr getRoot() {
        return NodePtr(0, nodes);
    }

    //! Return an iterator to get childs of node, interface
    ChildIterator getChildIterator(NodePtr& node) {
        return ChildIterator(node, nodes);
    }

    //! Can be used for debug, interface for debug
    size_t getNodeId(NodePtr& node) const {
        return node.idx;
    }
};

//! Data container, store nodes contiguously, store childs indices as a linked list
/*!
 * \details Nodes are stored contiguously in one vector.
 *          The root is the first node in the vector.
 *          The childs are stored as indices.
 *          Each node stores the indices for a next child and for its next brother.
 * \author adamp87
*/
template <typename T_Act>
class MCTreeStaticList {
    typedef T_Act ActType;

public:
    //! One node with childs, interface
    class Node : public MCTSNodeBase<ActType> {
        size_t child_head; //!< head of linked list for next child, breadth one level down
        size_t parent_next; //!< link to next child node of parent (brother), same breadth level

        friend class ChildIterator;
        friend class MCTreeStaticList;

        Node(ActType act) : MCTSNodeBase<ActType>(act), child_head(0), parent_next(0) {}
    };

    //!< pointer to a node, interface
    class NodePtr {
        size_t idx; //!< index of current node
        std::vector<Node>& nodes; //!< reference to tree

        friend class ChildIterator;
        friend class MCTreeStaticList;

        NodePtr(size_t idx, std::vector<Node>& nodes)
            : idx(idx), nodes(nodes) { }
    public:
        void operator=(const NodePtr& other) {
            idx = other.idx;
        }

        Node* operator->() {
            return &(nodes[idx]);
        }

        const Node* operator->() const {
            return &(nodes[idx]);
        }

        Node& operator*() {
            return nodes[idx];
        }

        const Node& operator*() const {
            return nodes[idx];
        }
    };

    //! Iterator to get childs of a node, interface
    class ChildIterator {
        size_t child_idx;//!< index of current child in tree container
        std::vector<Node>& nodes; //!< reference to tree

        friend class MCTreeStaticList;

        ChildIterator(NodePtr& node, std::vector<Node>& nodes)
            : child_idx(node->child_head), nodes(nodes) { }

    public:
        //! Returns true if not all childs have been visited, interface
        bool hasNext() const {
            return child_idx != 0;
        }

        //! Return pointer to next child, interface
        NodePtr next() {
            NodePtr ptr(child_idx, nodes);
            Node& child = nodes[child_idx];
            child_idx = child.parent_next;
            return ptr;
        }
    };

private:
    std::vector<Node> nodes; //!< array of the nodes, tree

public:
    //! Construct tree, interface
    MCTreeStaticList() {
        // artifical root, doesnt hold valid action
        ActType dummy = ActType();
        Node root(dummy);
        nodes.push_back(root);
    }

    //! Add a new node to parent, interface
    NodePtr addNode(NodePtr& _parent, ActType act) {
        // init leaf node
        Node child(act);
        Node& parent = *_parent;
        if (parent.child_head == 0) { // first child
            parent.child_head = nodes.size();
        } else { // find next free container of parent to put child
            size_t elem_idx = parent.child_head;
            while (nodes[elem_idx].parent_next != 0) {
                elem_idx = nodes[elem_idx].parent_next;
            }
            nodes[elem_idx].parent_next = nodes.size();
        }

        // add leaf to vector and init space for childs
        nodes.push_back(child);
        return NodePtr(nodes.size() - 1, nodes);
    }

    //! Get pointer to the root, interface
    NodePtr getRoot() {
        return NodePtr(0, nodes);
    }

    //! Return an iterator to get childs of node, interface
    ChildIterator getChildIterator(NodePtr& node) {
        return ChildIterator(node, nodes);
    }

    //! Can be used for debug, interface for debug
    size_t getNodeId(NodePtr& node) const {
        return node.idx;
    }
};

#endif //MCTREE_HPP
