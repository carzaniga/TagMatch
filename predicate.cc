#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define NDEBUG

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include "predicate.hh"

void predicate::node::add_pair(tree_t t, interface_t i) {
	if (pairs_count < LOCAL_PAIRS_CAPACITY) {
		// if the local table is not yet full, we simply add the new
		// pair to the local table
		local_pairs[pairs_count].tree = t;
		local_pairs[pairs_count].interface = i;
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
		new_table[pairs_count].tree = t;
		new_table[pairs_count].interface = i;
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
		external_pairs[pairs_count].tree = t;
		external_pairs[pairs_count].interface = i;
		pairs_count += 1;
	}
}

void predicate::destroy() {
	if (root.pos <= root.left->pos)
		return;

	node * S[filter_t::WIDTH];
	unsigned int head = 0;
	S[0] = root.left;

	for (;;) {
		node * n = S[head];
		if (n->left) {
			if (n->pos > n->left->pos) {
				S[++head] = n->left;
				n->left = 0;
				continue;
			} 
			n->left = 0;
		}
		if (n->right) {
			if (n->pos > n->right->pos) {
				S[++head] = n->right;
				n->right = 0;
				continue;
			}
		}
		delete(n);
		if (head == 0)
			return;
		--head;
	}
};

// this is the handler we use to perform the tree matching.  The
// predicate subset search finds subsets of the given filter, and
// this handler does the tree matching on the corresponding
// tree_interface pairs.
// 
class tree_matcher : public filter_const_handler {
public:
	tree_matcher(tree_t t, match_handler & mh): tree(t), matcher(mh) {}
	virtual bool handle_filter(const filter_t & filter, const predicate::node & n);
private:
	const tree_t tree;
	match_handler & matcher;
};

bool tree_matcher::handle_filter(const filter_t & filter, const predicate::node & n) {
	for(const tree_interface_pair * ti = n.ti_begin(); ti != n.ti_end(); ++ti)
		if (ti->tree == tree)
			if (matcher.match(filter, tree, ti->interface))
				return true;
	return false;
}

void predicate::match(const filter_t & x, tree_t t, match_handler & h) const {
	//
	// this is the modular matching function that uses the above match
	// handler through the modular find_subset_of function
	//
	tree_matcher matcher(t,h);
	find_subsets_of(x, matcher);
}

void predicate::match(const filter_t & x, tree_t t) const {
	// 
	// this is the non-modular matching function.  It basically
	// replicates the functionality of find_subsets_of() but then it
	// directly uses the results to select the tree--interface pair
	// corresponding to the given tree t.
	//
	const node * S[filter_t::WIDTH];
	unsigned int head = 0;

	std::cout << "->";

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= filter_t::WIDTH);

		const node * n = S[--head];
		if (n->key.subset_of(x)) {
			for(const tree_interface_pair * ti = n->ti_begin(); ti != n->ti_end(); ++ti) 
				if (ti->tree == t)
					std::cout << ' ' << ti->interface;
		}
		if (n->pos > n->left->pos) 
			S[head++] = n->left;

		if (x[n->pos] && n->pos > n->right->pos)
			S[head++] = n->right;
	}
	std::cout << std::endl;
}

predicate::node * predicate::add(const filter_t & x, tree_t t, interface_t i) {
	node * n = add(x);
	n->add_pair(t, i);
	return n;
}

predicate::node * predicate::add(const filter_t & x) {
	node * prev = &root;
	node * curr = root.left;

	while(prev->pos > curr->pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}
	if (x == curr->key)
		return curr;

	filter_t::pos_t pos = filter_t::most_significant_diff_pos(curr->key, x);

	prev = &root;
	curr = root.left;
	
	while(prev->pos > curr->pos && curr->pos > pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}

	// now we insert the new node between prev and curr
	++filter_count;
	if (prev->pos < filter_t::NULL_POSITION && x[prev->pos]) {
		return prev->right = new node(pos, x, curr);
	} else {
		return prev->left = new node(pos, x, curr);
	}
}

const predicate::node * predicate::find(const filter_t & x) const {
	const node * prev = &root;
	const node * curr = root.left;

	while(prev->pos > curr->pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}
	return (x == curr->key) ? curr : 0;
}

predicate::node * predicate::find(const filter_t & x) {
	const node * prev = &root;
	node * curr = root.left;

	while(prev->pos > curr->pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}
	return (x == curr->key) ? curr : 0;
}

void predicate::find_subsets_of(const filter_t & x, filter_const_handler & h) const {
	//
	// this is a non-recoursive (i.e., iterative) exploration of the
	// PATRICIA trie that looks for subsets.  The pattern is almost
	// exactly the same for supersets (see below).  We use a stack S
	// to keep track of the visited nodes, and we visit new nodes
	// along a subset (resp. superset) prefix.
	// 
	const node * S[filter_t::WIDTH];
	unsigned int head = 0;

	// if the trie is not empty we push the root node onto the stack.
	// The true root is root.left, not root, which is a sentinel node.
	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= filter_t::WIDTH);

		const node * n = S[--head];		// for each visited node n...

		if (n->key.subset_of(x)) {
			if (h.handle_filter(n->key, *n))
				return;
		} else
			// If the subset relation (above) does not hold for the prefix
			// defined by the current node --meaning the prefix from
			// position filter_t::WIDTH - 1 to position n->pos, included--
			// then we can prune the exploration from this point on.
			// 
			// NOTE: this heuristic is very effective for input filters
			// with high hamming-weight, but does not seem to do much for
			// very light input filters, in which case it incurrs a small
			// penalty, compared to simply ignoring this check.
			// 
			if (! n->key.prefix_subset_of(x, n->pos + 1))
				continue;

		if (n->pos > n->left->pos) 
			S[head++] = n->left;

		if (n->pos > n->right->pos && x[n->pos])
			S[head++] = n->right;
	}
}

void predicate::find_supersets_of(const filter_t & x, filter_const_handler & h) const {
	//
	// see above: find_subsets_of(const filter_t & x, filter_const_handler & h) const
	//
	const node * S[filter_t::WIDTH];
	unsigned int head = 0;

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= filter_t::WIDTH);

		const node * n = S[--head];

		if (x.subset_of(n->key)) {
			if (h.handle_filter(n->key, *n))
				return;
		} else if (! x.prefix_subset_of(n->key, n->pos + 1))
			continue;

		if (n->pos > n->right->pos) 
			S[head++] = n->right;

		if (n->pos > n->left->pos && !x[n->pos])
			S[head++] = n->left;
	}
}

void predicate::find_subsets_of(const filter_t & x, filter_handler & h) {
	//
	// see above: find_subsets_of(const filter_t & x, filter_const_handler & h) const
	//
	node * S[filter_t::WIDTH];
	unsigned int head = 0;

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= filter_t::WIDTH);

		node * n = S[--head];

		if (n->key.subset_of(x)) {
			if (h.handle_filter(n->key, *n))
				return;
		} else if (! n->key.prefix_subset_of(x, n->pos + 1))
			continue;

		if (n->pos > n->left->pos) 
			S[head++] = n->left;

		if (n->pos > n->right->pos && x[n->pos])
			S[head++] = n->right;
	}
}

void predicate::find_supersets_of(const filter_t & x, filter_handler & h) {
	//
	// see above: find_subsets_of(const filter_t & x, filter_const_handler & h) const
	//
	node * S[filter_t::WIDTH];
	unsigned int head = 0;

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= filter_t::WIDTH);

		node * n = S[--head];

		if (x.subset_of(n->key)) {
			if (h.handle_filter(n->key, *n))
				return;
		} else if (! x.prefix_subset_of(n->key, n->pos + 1))
			continue;

		if (n->pos > n->right->pos) 
			S[head++] = n->right;

		if (n->pos > n->left->pos && !x[n->pos])
			S[head++] = n->left;
	}
}
