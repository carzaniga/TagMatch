#ifndef COMPACT_PATRICIA_PREDICATE_HH_INCLUDED
#define COMPACT_PATRICIA_PREDICATE_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <map>

#include "filter.hh"

//
// This is a generic predicate (template), namely a map: filter_t-->T
// that supports addition and subset search.  The data structure is
// conceptually a prefix trie, or more specifically a PATRICIA trie,
// implemented as a compact array.
//
// Each node that is not a leaf node has a right child.  The offset of
// the right child stored in the node serves as an explicit link to
// the right child.  If the node also has a left child, then the link
// is implicit, and that child is the next node in the array.
//
// The structure is built starting from an array of nodes that are
// sorted in increasing order by their bitvector keys (example below).
// The order is lexicographical, but it also corresponds to the the
// numeric order if one interprets bitvectors as binary numbers.
//
/* EXAMPLE:

        key              pos     offset

   >0 0 0 0 0 1 ----.     0        6
    0>0 0 1 0 1 --.  \    1        2
    0 0 0 1 1 1>   |  |   W        0 (leaf)
    0 1 1>0 0 1 <-'   |   3        2
    0 1 1 0 1 0> \    |   W        0 (leaf)
    0 1 1 1 0 0><'   /    W        0 (leaf)
    1 0>0 0 0 1 <--'      2        2
    1 0 0 1 1 0> \        W        0 (leaf)
    1 0 1>0 1 1 <'        3        1 (next node is the RIGHT child, no LEFT child)
    1 0 1 1 1 1><'        W        0 (leaf)

   The bit position (pos) for each node is indicated by '>', the bit
   position 'W' (6 in this simplified example) is the width of the
   bitvector, which implies that the node is a leaf in the trie, and
   represents a full prefix, consisting of the whole bitvector.
*/
// The data structure must be "consolidated" before it can be used for
// matching.  The consolidation process is what compiles the links
// etc.
//
// Before consolidation there are two ways to build the data
// structure.  One is to add filters in no particular order.  In this
// case, which is the default case, the predicate uses a temporary
// index (std::map) to sort the filters and also to avoid duplicates.
//
// Another way is to store the filters directly in the final array.
// This saves the memory of the temporary index, but it requires that
// that the filters are added in order.
//
template <typename T>
class compact_patricia_predicate {
public:
	compact_patricia_predicate () : tmp_nodes(), nodes(nullptr) {}
	~compact_patricia_predicate () { if (nodes) delete[](nodes); }

	// Uses the direct insertion method.  This requires the
	// initialization of the array with its SIZE, namely the maximal
	// number of filters stored in the structure.  The reverse
	// parameter indicates the expected insertion order.  Notice that
	// the default value is increasing order, which is is the reverse
	// of what the structure then uses after consolidation.
	void use_pre_sorted_filters(unsigned int s, bool reverse = true);

	// Clears the data structure completely.
    void clear();

	// Adds a filter and returns a reference to the mapped value.
	T & add(const filter_t & x);

	// Consolidates the data structure.
	void consolidate();

	class match_handler {
	public:
		// this will be called by predicate::match().  The return value
		// indicates whether the search for matching filters should stop.
		// So, if this function returns TRUE, match() will terminate
		// immediately.
		//
		virtual bool match(T &) = 0;
	};

	// Finds all subsets of x in the map, for each matching set S_i,
	// pass its mapped value Value_i to the match handler h through
	// its match() callback.
	void find_all_subsets(const filter_t & x, match_handler & h);

private:
	void reverse_nodes();
	void compute_offsets(unsigned int first, unsigned int last);
#if USING_POPCOUNT_LIMITS
	void compute_popcount_limits(unsigned int n);
#endif
	class node {
	public:
		filter_t key;
		unsigned int right_offset;
		filter_pos_t pos;
#if USING_POPCOUNT_LIMITS
		filter_pos_t popcount_prefix;
		filter_pos_t popcount_min;
		filter_pos_t popcount_max;
#endif
		T value;

		node() {}
	};

	typedef std::map<filter_t, T> map_t;
	map_t tmp_nodes;
	node * nodes;
	unsigned int size;
	bool reverse_order;

	bool is_leaf_node(unsigned int n) {
		return nodes[n].right_offset == 0;
	}

	bool has_right_child(unsigned int n) {
		return nodes[n].right_offset > 0;
	}

	bool has_left_child(unsigned int n) {
		return nodes[n].right_offset > 1;
	}

	unsigned int left_child(unsigned int n) {
		assert(has_left_child(n));
		return n + 1;
	}

	unsigned int right_child(unsigned int n) {
		assert(has_right_child(n));
		return n + nodes[n].right_offset;
	}
};

template <typename T>
void compact_patricia_predicate<T>::use_pre_sorted_filters(unsigned int s, bool rev) {
	clear();
	nodes = new node[s];
	size = 0;
	reverse_order = rev;
}


template <typename T>
void compact_patricia_predicate<T>::clear() {
	tmp_nodes.clear();
	if (nodes) {
		delete[](nodes);
		nodes = nullptr;
	}
	size = 0;
}

template <typename T>
T & compact_patricia_predicate<T>::add(const filter_t & x) {
	if (nodes) {
		node & n = nodes[size++];
		n.key = x;
		n.right_offset = 0;
		n.pos = filter_t::WIDTH;
		return n.value;
	} else
		return tmp_nodes[x];
}

template <typename T>
void compact_patricia_predicate<T>::reverse_nodes() {
	if (size == 0)
		return;

	unsigned int i = 0;
	unsigned int j = size - 1;
	while (i < j) {
		filter_t tmp_key = std::move(nodes[i].key);
		T tmp_value = std::move(nodes[i].value);

		nodes[i].key = std::move(nodes[j].key);
		nodes[i].value = std::move(nodes[j].value);

		nodes[j].key = std::move(tmp_key);
		nodes[j].value = std::move(tmp_value);

		++i;
		--j;
	}
}

template <typename T>
void compact_patricia_predicate<T>::consolidate() {
	if (!nodes) {
		size = tmp_nodes.size();
		if (size == 0)
			return;
		nodes = new node[size];
		node * n = nodes;
		for (typename map_t::iterator i = tmp_nodes.begin(); i != tmp_nodes.end(); ++i) {
			n->key = i->first;
			n->value = std::move(i->second);
			n->pos = filter_t::WIDTH;
			++n;
		}
		tmp_nodes.clear();
	} else {
		if (reverse_order)
			reverse_nodes();
	}
	compute_offsets(0, size - 1);

#if USING_POPCOUNT_LIMITS
	compute_popcount_limits(0);
#endif
}

template <typename T>
void compact_patricia_predicate<T>::compute_offsets(unsigned int first, unsigned int last) {
	assert(first <= last);

	if (first == last) {
		nodes[first].right_offset = 0;
		nodes[first].pos = filter_t::WIDTH;
		return;
	}

	filter_pos_t d_pos = nodes[first].key.leftmost_diff(nodes[last].key);
	assert(d_pos < filter_t::WIDTH); // no duplicate keys!

	// We now run a binary search to find the first position (child)
	// in which the bit in position d_pos changes (here from 0 to 1,
	// since filters are sorted in increasing order).
	unsigned int zero_pos = first;
	unsigned int child = last;
	while (child - zero_pos > 1) {
		// LOOP INVARIANT:
		assert(nodes[zero_pos].key[d_pos] == 0 && nodes[child].key[d_pos] == 1);

		unsigned int m = (child + zero_pos) / 2;
		if (nodes[m].key[d_pos])
			child = m;
		else
			zero_pos = m;
	}
	nodes[first].right_offset = child - first;
	nodes[first].pos = d_pos;

	if (first + 1 < child)
		compute_offsets(first + 1, child - 1);

	compute_offsets(child, last);
}

#if USING_POPCOUNT_LIMITS
template <typename T>
void compact_patricia_predicate<T>::compute_popcount_limits(unsigned int n) {
	//
	nodes[n].popcount_prefix = nodes[n].key.prefix_popcount(nodes[n].pos);
	nodes[n].popcount_min = nodes[n].key.popcount();
	nodes[n].popcount_max = nodes[n].popcount_min;
	if (has_left_child(n)) {
		unsigned int child = left_child(n);
		compute_popcount_limits(child);
		if 	(nodes[n].popcount_min > nodes[child].popcount_min)
			nodes[n].popcount_min = nodes[child].popcount_min;
		if 	(nodes[n].popcount_max < nodes[child].popcount_max)
			nodes[n].popcount_max = nodes[child].popcount_max;
	}
	if (has_right_child(n)) {
		unsigned int child = right_child(n);
		compute_popcount_limits(child);
		if 	(nodes[n].popcount_min > nodes[child].popcount_min)
			nodes[n].popcount_min = nodes[child].popcount_min;
		if 	(nodes[n].popcount_max < nodes[child].popcount_max)
			nodes[n].popcount_max = nodes[child].popcount_max;
	}
}
#endif

#if USING_POPCOUNT_LIMITS
static void compute_suffix_popcounts(filter_pos_t * P, const filter_t & f) {
	filter_pos_t i = 0;
	filter_pos_t c = f.popcount();
	filter_pos_t p = f.next_bit(0);
	while (i < filter_t::WIDTH) {
		P[i] = c;
		if (i == p) {
			--c;
			p = f.next_bit(++i);
		} else
			++i;
	}
}
#endif

template <typename T>
void compact_patricia_predicate<T>::find_all_subsets(const filter_t & x,
													 compact_patricia_predicate<T>::match_handler & h) {
	if (nodes) {
		unsigned int S[filter_t::WIDTH];
		unsigned int head = 0;
		unsigned int n = 0;
#if USING_POPCOUNT_LIMITS
		filter_pos_t x_suffix_popcounts[filter_t::WIDTH];
		compute_suffix_popcounts(x_suffix_popcounts, x);
#endif
		for (;;) {
			if (nodes[n].key.subset_of(x))
				if (h.match(nodes[n].value))
					return;
			if (!is_leaf_node(n)
				// here, if n is not a leaf node, we check whether we
				// have to consider n's subtree or if we can ignore it
				// altogether for the purpose of a subset search.
				// These are essentially optimizations for the trie
				// walk, in the sense that the subset search should be
				// correct even if these additional conditions are
				// always true.
#if  USING_POPCOUNT_LIMITS
				&& nodes[n].popcount_min <= nodes[n].popcount_prefix + x_suffix_popcounts[nodes[n].pos]
#endif
				&& nodes[n].key.prefix_subset_of(x, nodes[n].pos)) {

				// Structural properties:
				//   1. !is_leaf_node(n, x) ==> (has_left_child(n) || has_right_child(n))
				//   2. has_left_child(n) ==> has_right_child(n)
				// so...
				assert(has_right_child(n));

				if (x[nodes[n].pos]) {
#if GO_RIGHT_FIRST
					if (has_left_child(n))
						S[head++] = left_child(n);
					n = right_child(n);
#else
					if (has_left_child(n)) {
						S[head++] = right_child(n);
						n = left_child(n);
					} else
						n = right_child(n);
#endif
					continue;
				} else if (has_left_child(n)) {
					n = left_child(n);
					continue;
				}
			}
			if (head > 0) {
				n = S[--head];
			} else
				break;
		}
	}
}

#endif // include guard
