#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <math.h>

#include "routing.hh"
#include "matcher.hh"
#include "filter.hh"
#include "patricia_predicate.hh"

//
// This is a predicate implemented as a prefix trie.  More
// specifically, the trie is implemented as a simplified PATRICIA trie.
//
// Each node N in the PATRICIA trie represents a filter (N.key) and
// also a prefix that is common to all nodes in the sub-tree rooted at
// N (including N itself).  The prefix is defined by N.pos, which
// indicates the length of the prefix.  N.pos can be filter_t::WIDTH,
// meaning that the prefix is the entire filter in N.
//

nodes that are below , which in our case
// is a filter.  Each node also stores a bit position in the filter
// that represents the length of the prefix

class predicate::node {
public:
	node * left;
	node * right;
	const filter_t key;
	const filter_pos_t pos;

private:
	static const uint16_t EXT_PAIRS_ALLOCATION_UNIT = 16; 
	static const unsigned int LOCAL_PAIRS_CAPACITY = 4;

	uint16_t pairs_count;

	union {
		tree_interface_pair local_pairs[LOCAL_PAIRS_CAPACITY];
		tree_interface_pair * external_pairs;
	};

public:
	void add_pair(tree_t t, interface_t i);
	void remove_pair(tree_t t, interface_t i);

	// pointer to the first tree--interface pair
	//
	tree_interface_pair * tip_begin() {
		return (pairs_count <= LOCAL_PAIRS_CAPACITY) ? local_pairs : external_pairs;
	}

	// pointer to the first tree--interface pair
	//
	const tree_interface_pair * tip_begin() const {
		return (pairs_count <= LOCAL_PAIRS_CAPACITY) ? local_pairs : external_pairs;
	}

	// pointer to one-past the tree--interface pair
	//
	tree_interface_pair * tip_end() {
		return tip_begin() + pairs_count;
	}

	// pointer to one-past the tree--interface pair
	//
	const tree_interface_pair * tip_end() const {
		return tip_begin() + pairs_count;
	}

	// creates a new node connected to another (child) node
	//
	node(const filter_t & k, filter_pos_t d_pos)
		: left(nullptr), right(nullptr), key(k), pos(d_pos), pairs_count(0) {}

	~node() {
		if (pairs_count > LOCAL_PAIRS_CAPACITY)
			free(external_pairs);
	}
};

predicate::predicate () : root(nullptr) {}

predicate::~predicate () {
	clear();
}

void predicate::node::add_pair(tree_t t, interface_t i) {
    // here we don't check if the tree-interface pair already exists
    // becuase we assume that triple filter,tree,interface is unique
    // becuase of the compresion algorithm 
	if (pairs_count < LOCAL_PAIRS_CAPACITY) {
		// if the local table is not yet full, we simply add the new
		// pair to the local table
		local_pairs[pairs_count].assign(t, i);
		pairs_count += 1;
	} else if (pairs_count == LOCAL_PAIRS_CAPACITY) {
		// if we have a full local table we create an external table.
		// We compute the bytes needed to store the pairs already
		// stored locally, plus the new one
		size_t bytes_needed = (pairs_count + 1) * sizeof(tree_interface_pair);
		// round it up to the next EXT_PAIRS_ALLOCATION_UNIT
		bytes_needed += (EXT_PAIRS_ALLOCATION_UNIT - bytes_needed % EXT_PAIRS_ALLOCATION_UNIT);

		tree_interface_pair * new_table = (tree_interface_pair *)malloc(bytes_needed);

		// copy the local pairs to the external storage
		memcpy(new_table, local_pairs, sizeof(local_pairs));
		// add the new one
		new_table[pairs_count].assign(t, i);
		++pairs_count;
		// link the external storage
		external_pairs = new_table;
	} else {
		size_t byte_pos = pairs_count * sizeof(tree_interface_pair);
		if (byte_pos % EXT_PAIRS_ALLOCATION_UNIT == 0) {
			// if we have a full (external) table, we reallocate the
			// external table with an extra EXT_PAIRS_ALLOCATION_UNIT bytes
			external_pairs = (tree_interface_pair *)realloc(external_pairs, 
															byte_pos + EXT_PAIRS_ALLOCATION_UNIT);
		}
		external_pairs[pairs_count].assign(t, i);
		pairs_count += 1;
	}
}

void predicate::clear() {
	// This is a simple, full walk of the whole trie.  We do not
	// implement this as a recursive walk, but rather use an explicit
	// stack (S).
	// 
	if (root) {
		node * S[filter_t::WIDTH];
		unsigned int head = 0;
		S[0] = root;
		root = nullptr;
        
		for (;;) {
			node * n = S[head];
			if (n->left) {
				S[++head] = n->left;
				n->left = 0;
				continue;
			} 
			if (n->right) {
				S[++head] = n->right;
				n->right = 0;
				continue;
			}
			delete(n);
			if (head == 0)
				return;
			--head;
		}
	}
};

predicate::node * predicate::add(const filter_t & x) {
	// We insert a new node if we don't find exactly the same filter
	// in the trie.  The key to understanding the insertion algorithm
	// is this: we insert a new node into a pointer pointed to by
	// source_p (pointer to pointer).  If the pointer points to
	// nothing (nullptr), which is the case at the very beginning,
	// then we insert a new node N with N.pos = filter_t::WIDTH.
	// Otherwise, if the new filter (x) shares a *longer* prefix with
	// the key in the current node N (prefix length defined by N.pos),
	// then we follow the left or right pointer, that is we shift the
	// source pointer.  Otherwise, if the new filter has a shorter
	// prefix, we insert the node right there, in between the source
	// and *source_p.
	// 
	assert(x.popcount() > 0);
	node ** source_p = &root;

	while (*source_p) {
		filter_pos_t d_pos = x.leftmost_diff((*source_p)->key);

		if (d_pos == filter_t::WIDTH)
			// x == (*source_p) ==> we found key x
			return *source_p;

		if (d_pos < (*source_p)->pos) { 
			// x does not share the same prefix defined by (*source_p)
			// so we must insert the new node here, using source_p
			node * n = new node(x, d_pos);
			if ((*source_p)->key[d_pos])
				n->right = *source_p;
			else
				n->left = *source_p;
			*source_p = n;
			return n;
		} else { 				// follow the trie
			// x shares the same prefix defined by (*source_p) so we
			// follow the right ling from (*source_p)
			if (x[(*source_p)->pos])
				source_p = &((*source_p)->right);
			else
				source_p = &((*source_p)->left);
		}
	}
	return *source_p = new node(x, filter_t::WIDTH);
}

void predicate::add(const filter_t & x,
					const tree_interface_pair * begin, const tree_interface_pair * end) {
	node * n = add(x);
	for (; begin != end; ++begin)
		n->add_pair(begin->tree(), begin->interface());
}

void predicate::add(const filter_t & x, tree_t t, interface_t i) {
	node * n = add(x);
	n->add_pair(t, i);
}

void predicate::match(const filter_t & x, match_handler & h) const {
	//
	// this is a non-recursive (i.e., iterative) exploration of the
	// PATRICIA trie that looks for subsets.  We use a stack S to keep
	// track of the visited nodes, and we visit new nodes along a
	// subset prefix.
	// 
	if (root) {
		// if the trie is not empty we push the root node onto the stack.
		//
		node * S[filter_t::WIDTH];
		unsigned int head = 0;
        
		S[head++] = root;

		while(head > 0) {
			assert(head <= filter_t::WIDTH);
			--head;
			const node * n = S[head];

			if (n->key.subset_of(x)) {
				for (const tree_interface_pair * tip = n->tip_begin(); tip != n->tip_end(); ++tip)
					if (h.match(n->key, tip->tree(), tip->interface()))
						return;
			} else if (n->pos > 0 && !n->key.prefix_subset_of(x, n->pos - 1))
				continue;

			if (n->left) 
				S[head++] = n->left;

			if (n->right && x[n->pos])
				S[head++] = n->right;
		}
	}
}

