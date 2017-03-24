#ifndef MCTREE_HPP
#define MCTREE_HPP

#include <vector>
#include <memory>

#include "defs.hpp"

template <typename T_Card = uint8, typename T_Count = uint32>
struct MCTSNodeBase {
    typedef T_Count CountType;

    uint8 expanded;
    T_Card card; // card played out
    CountType visits; // node visit count
    CountType wins[28]; // number of results for each points

    MCTSNodeBase() {
        expanded = 0;
        card = 255;
        visits = 1;
        for (uint8 i = 0; i < 28; ++i) {
            wins[i] = 0;
        }
    }
};

class MCTreeDynamic {
public:
    class Node : public MCTSNodeBase<> {
    private:
        std::vector<std::unique_ptr<Node> > childs;

        friend class ChildIterator;
        friend class MCTreeDynamic;

        Node() {}
        Node(const Node&) = delete;
    };

    typedef Node* NodePtr;

    class ChildIterator {
        size_t pos;
        const NodePtr node;

        friend class MCTreeDynamic;

        ChildIterator(NodePtr& node)
            : pos(0), node(node) { }

    public:
        bool hasNext() const {
            return node->childs.size() != pos;
        }

        NodePtr next() {
            return NodePtr(node->childs[pos++].get());
        }
    };

private:
    std::unique_ptr<Node> root;

public:
    MCTreeDynamic() {
        root.reset(new Node());
    }

    NodePtr addNode(NodePtr& _parent, uint8 card) {
        // init leaf node
        std::unique_ptr<Node> child(new Node());
        child->card = card;

        _parent->childs.push_back(std::move(child));
        return _parent->childs.back().get();
    }

    NodePtr getRoot() {
        return root.get();
    }

    ChildIterator getChildIterator(NodePtr& node) {
        return ChildIterator(node);
    }

    size_t getNodeId(NodePtr& node) const {
        return reinterpret_cast<size_t>(node);
    }
};

class MCTreeStaticArray {
public:
    class Node : public MCTSNodeBase<> {
        size_t childs[39]; // child indices

        friend class ChildIterator;
        friend class MCTreeStaticArray;

        Node() {
            for (uint8 i = 0; i < 39; ++i) {
                childs[i] = 0;
            }
        }
    };

    class NodePtr {
        size_t idx;
        std::vector<Node>& nodes;

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

    class ChildIterator {
        size_t pos;
        const Node& node;
        std::vector<Node>& nodes;

        friend class MCTreeStaticArray;

        ChildIterator(NodePtr& node, std::vector<Node>& nodes)
            : pos(0), node(*node), nodes(nodes) { }

    public:
        bool hasNext() const {
            return node.childs[pos] != 0;
        }

        NodePtr next() {
            return NodePtr(node.childs[pos++], nodes);
        }
    };

private:
    std::vector<Node> nodes;

public:
    MCTreeStaticArray() {
        Node root;
        nodes.push_back(root);
    }

    NodePtr addNode(NodePtr& _parent, uint8 card) {
        // init leaf node
        Node child;
        child.card = card;

        // find next free idx of parent to put child
        uint8 idx = 0;
        Node& parent = *_parent;
        for (; idx < 39; ++idx) {
            if (parent.childs[idx] == 0) {
                break;
            }
        }
        if (idx == 39)
            std::cerr << "Error in child saving" << std::endl;

        // add leaf to vector and init space for childs
        size_t child_idx = nodes.size();
        parent.childs[idx] = child_idx;
        nodes.push_back(child);
        return NodePtr(child_idx, nodes);
    }

    NodePtr getRoot() {
        return NodePtr(0, nodes);
    }

    ChildIterator getChildIterator(NodePtr& node) {
        return ChildIterator(node, nodes);
    }

    size_t getNodeId(NodePtr& node) const {
        return node.idx;
    }
};

class MCTreeStaticList {
public:
    class Node : public MCTSNodeBase<> {
        struct child_element {
            size_t next;
            size_t child;
        };

        size_t child_first;

        friend class ChildIterator;
        friend class MCTreeStaticList;

        Node() {
            child_first = 0;
        }
    };

    class NodePtr {
        size_t idx;
        std::vector<Node>& nodes;

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

    class ChildIterator {
        size_t child_idx;
        std::vector<Node>& nodes;
        std::vector<Node::child_element>& childs;

        friend class MCTreeStaticList;

        ChildIterator(NodePtr& node, std::vector<Node>& nodes, std::vector<Node::child_element>& childs)
            : child_idx(node->child_first), nodes(nodes), childs(childs) { }

    public:
        bool hasNext() const {
            return child_idx != 0;
        }

        NodePtr next() {
            Node::child_element& child = childs[child_idx];
            NodePtr ptr(child.child, nodes);
            child_idx = child.next;
            return ptr;
        }
    };

private:
    std::vector<Node> nodes;
    std::vector<Node::child_element> childs;

public:
    MCTreeStaticList() {
        Node root;
        nodes.push_back(root);

        Node::child_element elem;
        elem.child = elem.next = 0;
        childs.push_back(elem);
    }

    NodePtr addNode(NodePtr& _parent, uint8 card) {
        // init leaf node
        Node child;
        child.card = card;

        Node& parent = *_parent;
        if (parent.child_first == 0) { // first child
            parent.child_first = childs.size();
        } else { // find next free container of parent to put child
            size_t elem_idx = parent.child_first;
            while (childs[elem_idx].next != 0) {
                elem_idx = childs[elem_idx].next;
            }
            childs[elem_idx].next = childs.size();
        }

        size_t child_idx = nodes.size();
        Node::child_element elem;
        elem.child = child_idx;
        elem.next = 0;
        childs.push_back(elem);

        // add leaf to vector and init space for childs
        nodes.push_back(child);
        return NodePtr(child_idx, nodes);
    }

    NodePtr getRoot() {
        return NodePtr(0, nodes);
    }

    ChildIterator getChildIterator(NodePtr& node) {
        return ChildIterator(node, nodes, childs);
    }

    size_t getNodeId(NodePtr& node) const {
        return node.idx;
    }
};

#endif //MCTREE_HPP
