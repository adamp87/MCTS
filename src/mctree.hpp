#ifndef MCTREE_HPP
#define MCTREE_HPP

#include <vector>
#include <memory>

#include <mutex>
#include <atomic>

//! Base class for all Tree::Node implementations
template <typename T_Move>
struct MCTSNodeBase {
    typedef T_Move MoveType;
    typedef std::uint_fast32_t CountType;

    //! Dummy LockGuard, helps to keep policy transparent
    class LockGuard {
    public:
        LockGuard(MCTSNodeBase<T_Move>&) {}
    };

    T_Move      move;   //!< e.g. card played out
    CountType   visits; //!< node visit count
    double      wins;   //!< number of wins

    MCTSNodeBase(T_Move move)
        : move(move), visits(1), wins(0.0)
    { }
};

//! Base class for multithreaded MCTreeDynamic::Node
template <typename T_Move>
struct MCTSNodeBaseMT {
    typedef T_Move MoveType;
    typedef std::uint_fast32_t CountType;

    //! LockGuard inherited from std, constructor takes node
    class LockGuard : public std::lock_guard<std::mutex> {
    public:
        LockGuard(MCTSNodeBaseMT<T_Move>& node)
            : std::lock_guard<std::mutex>(node.lock)
        {}
    };

    //! Workaround, until C20 standard enables atomic for double
    class MyAtomicDouble {
        double value;
        std::mutex lock;

    public:
        MyAtomicDouble() : value(0) {}
        operator double() const {
            return value;
        }
        void operator+=(double& val) {
            std::lock_guard<std::mutex> guard(lock); (void)guard;
            value += val;
        }
    };

    T_Move                  move;   //!< e.g. card played out
    std::mutex              lock;   //!< mutex for thread-safe policy
    std::atomic<CountType>  visits; //!< node visit count
    MyAtomicDouble          wins;   //!< number of wins for each points

    MCTSNodeBaseMT(T_Move move)
        : move(move), visits(1)
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
    typedef typename T_NodeBase::MoveType MoveType;

    //! One node with childs, interface
    class Node : public T_NodeBase {
    private:
        std::vector<std::unique_ptr<Node> > childs; //!< store childs as unique pointers

        friend class ChildIterator;
        friend class MCTreeDynamic;

        Node(MoveType move) : T_NodeBase(move) {}
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
        // artifical root, doesnt hold valid move
        root.reset(new Node(0));
    }

    //! Add a new node to parent, interface
    NodePtr addNode(NodePtr& _parent, MoveType move) const {
        // init leaf node
        std::unique_ptr<Node> child(new Node(move));

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
template <typename T_Move, unsigned int N_MaxChild>
class MCTreeStaticArray {
public:
    typedef T_Move MoveType;

    //! One node with childs, interface
    class Node : public MCTSNodeBase<MoveType> {
        size_t childs[N_MaxChild]; //!< child indices as fixed length array

        friend class ChildIterator;
        friend class MCTreeStaticArray;

        Node(MoveType move) : MCTSNodeBase<MoveType>(move), childs{0} {}
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
        // artifical root, doesnt hold valid move
        Node root(0);
        nodes.push_back(root);
    }

    //! Add a new node to parent, interface
    NodePtr addNode(NodePtr& _parent, MoveType move) {
        // init leaf node
        Node child(move);

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
template <typename T_Move>
class MCTreeStaticList {
    typedef T_Move MoveType;

public:
    //! One node with childs, interface
    class Node : public MCTSNodeBase<MoveType> {
        size_t child_head; //!< head of linked list for next child, breadth one level down
        size_t parent_next; //!< link to next child node of parent (brother), same breadth level

        friend class ChildIterator;
        friend class MCTreeStaticList;

        Node(MoveType move) : MCTSNodeBase<MoveType>(move), child_head(0), parent_next(0) {}
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
        // artifical root, doesnt hold valid move
        Node root(0);
        nodes.push_back(root);
    }

    //! Add a new node to parent, interface
    NodePtr addNode(NodePtr& _parent, MoveType move) {
        // init leaf node
        Node child(move);
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
