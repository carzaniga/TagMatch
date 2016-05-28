#ifndef COMPACT_PATRICIA_PREDICATE_HH_INCLUDED
#define COMPACT_PATRICIA_PREDICATE_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if TRACING_WALK
#include <iostream>
#endif

#include <map>
#include <functional>
#include "filter.hh"

//
// This is a predicate implemented conceptually as a prefix trie.
// More specifically, the trie is implemented as a PATRICIA trie.
// However, the implementation is encoded as a compact array in which
// the left links are explicit but the right links are implicit to the
// next node in the array.  The left links are implemented as offsets
// in the array.
//
// The encoding is as in the following example.  Notice that filters
// are sorted in descending lexicographical order.
/*
   EXAMPLE:
                     -0010011011_
                    /    ^       \ (implicit right link)
                   / /0010010111<'
                  | |          ^
                  |  >0010010100
                   \            ^
                    ->0001111010_
                    /     ^      \ (implicit right link)
                   |  0001111010<'
                   \            ^
                    `>0001011010
                                ^
*/
template <typename T>
class compact_patricia_predicate {
public:
	compact_patricia_predicate () : tmp_nodes(), nodes() {}
	~compact_patricia_predicate () { clear(); }

    void clear();
	T & add(const filter_t & x);
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

	void find_all_subsets(const filter_t & x, match_handler & h);

private:
	void compute_offset(unsigned int first, unsigned int last);

	class node {
	public:
		filter_t key;
		unsigned int left_offset;

		filter_pos_t pos;
		T value;

		node() {}
	};

    struct reverse_order_comparator {
		bool operator()(const filter_t & x, const filter_t & y) const {
			return (x > y);
		}
    };

	typedef std::map<filter_t, T, reverse_order_comparator> map_t;
	map_t tmp_nodes;
	node * nodes;
	unsigned int size;

	bool has_right_child(unsigned int n) {
		return nodes[n].left_offset > 1;
	}

	bool has_left_child(unsigned int n) {
		return nodes[n].left_offset > 0;
	}

	unsigned int right_child(unsigned int n) {
		assert(has_right_child(n));
		return n + 1;
	}

	unsigned int left_child(unsigned int n) {
		assert(has_left_child(n));
		return n + nodes[n].left_offset;
	}
};

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
	//
	return tmp_nodes[x];
}

template <typename T>
void compact_patricia_predicate<T>::consolidate() {
	//
	size = tmp_nodes.size();
	if (size == 0)
		return;
	nodes = new node[size];
	node * n = nodes;
	for (typename map_t::iterator i = tmp_nodes.begin(); i != tmp_nodes.end(); ++i) {
		n->key = i->first;
		n->left_offset = 0;
		n->value = std::move(i->second);
		n->pos = filter_t::WIDTH;
		++n;
	}
	tmp_nodes.clear();

	if (size > 1)
		compute_offset(0, size - 1);
}

template <typename T>
void compact_patricia_predicate<T>::compute_offset(unsigned int first, unsigned int last) {
	//
	filter_pos_t d_pos = nodes[first].key.leftmost_diff(nodes[last].key);
	unsigned int one_pos = first;
	unsigned int child = last;
	while (child - one_pos > 1) {
		unsigned int m = (child + one_pos) / 2;
		if (nodes[m].key[d_pos])
			one_pos = m;
		else
			child = m;
	}
	nodes[first].left_offset = child - first;
	nodes[first].pos = d_pos;
	if (first + 2 < child)
		compute_offset(first + 1, child - 1);
	if (child < last)
		compute_offset(child, last);
}

template <typename T>
void compact_patricia_predicate<T>::find_all_subsets(const filter_t & x,
													 compact_patricia_predicate<T>::match_handler & h) {
	if (nodes) {
		unsigned int S[filter_t::WIDTH];
		unsigned int head = 0;
		unsigned int n = 0;

		for (;;) {
			if (nodes[n].pos < filter_t::WIDTH) { // internal node
				if (nodes[n].key.prefix_subset_of(x, nodes[n].pos)) {
					if (x[nodes[n].pos]) {
						if (nodes[n].key.subset_of(x)) 
							if (h.match(nodes[n].value))
								return;
						if (has_right_child(n)) {
							if  (has_left_child(n))
								S[head++] = left_child(n);
							n = right_child(n);
							continue;
						}
					}
					if (has_left_child(n)) {
						n = left_child(n);
						continue;
					}
				}
			} else {			// leaf node
				if (nodes[n].key.subset_of(x))
					if (h.match(nodes[n].value))
						return;
			}
			if (head > 0) {
				n = S[--head];
			} else
				break;
		}
	}
}

#endif
