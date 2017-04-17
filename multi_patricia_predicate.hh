#ifndef PATRICIA_PREDICATE_HH_INCLUDED
#define MULTI_PATRICIA_PREDICATE_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "filter.hh"

//
// This is a predicate implemented as a prefix trie.  More
// specifically, the trie is implemented as a simplified PATRICIA trie.
//
// Each node N in the PATRICIA trie represents a filter (N.key) and
// also a prefix that is common to all nodes in the sub-tree rooted at
// N (including N itself).  The prefix is defined by N.key and N.pos,
// which indicates the length of the prefix.  N.pos can be
// filter_t::WIDTH, meaning that the prefix is the entire filter in N.
// Node N and all the nodes below it share the first N.pos bits (of
// their filters).
//
/*
   EXAMPLE:     root:  *\
                         \
   common prefix -->  --- \
                      0010011011
                      /  ^-pos \
                 left/          \right
                    /            \
             0010010110        0011101110
                       ^        /    ^
					       left/
                              /
                         0011100001
                                   ^
   All left/right links are either indicated or equal to nullptr.
*/
template <typename T>
class multi_patricia_predicate {
public:
	multi_patricia_predicate () : roots{nullptr} {}
	~multi_patricia_predicate () { clear(); }

    void clear();
	T & add(const filter_t & x);

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
	class node {
	public:
		node * left;
		node * right;
		const filter_t key;
		const filter_pos_t pos;
		T value;

		// creates a new node connected to another (child) node
		//
		node(const filter_t & k, filter_pos_t d_pos)
			: left(nullptr), right(nullptr), key(k), pos(d_pos) {}
	};

    node * roots[filter_t::WIDTH];
};

template <typename T>
void multi_patricia_predicate<T>::clear() {
	// This is a simple, full walk of the whole trie.  We do not
	// implement this as a recursive walk.  Instead, we use an
	// explicit stack (S).
	//
	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i) {
		if (roots[i]) {
			node * S[filter_t::WIDTH];
			unsigned int head = 0;
			S[0] = roots[i];
			roots[i] = nullptr;

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
	}
}

template <typename T>
T & multi_patricia_predicate<T>::add(const filter_t & x) {
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
	node ** source_p = roots + (x.popcount() - 1);

	while (*source_p) {
		filter_pos_t d_pos = x.leftmost_diff((*source_p)->key);

		if (d_pos == filter_t::WIDTH)
			// x == (*source_p) ==> we found key x
			return (*source_p)->value;

		if (d_pos < (*source_p)->pos) {
			// x does not share the same prefix defined by (*source_p)
			// so we must insert the new node here, using source_p
			node * n = new node(x, d_pos);
			if ((*source_p)->key[d_pos]) {
				n->left = nullptr;
				n->right = *source_p;
			} else {
				n->left = *source_p;
				n->right = nullptr;
			}
			*source_p = n;
			return n->value;
		} else { 				// follow the trie
			// x shares the same prefix defined by (*source_p) so we
			// follow the right ling from (*source_p)
			if (x[(*source_p)->pos])
				source_p = &((*source_p)->right);
			else
				source_p = &((*source_p)->left);
		}
	}
	node * n = new node(x, filter_t::WIDTH);
	n->left = nullptr;
	n->right = nullptr;
	*source_p = n;
	return n->value;
}

template <typename T>
void multi_patricia_predicate<T>::find_all_subsets(const filter_t & x,
												   multi_patricia_predicate<T>::match_handler & h) {
	//
	// this is a non-recursive (i.e., iterative) exploration of the
	// PATRICIA trie that looks for subsets.  We use a stack S to keep
	// track of the visited nodes, and we visit new nodes along a
	// subset prefix.
	//
	struct exploration_state {
		node * n;
		filter_pos_t slack;
	};

	filter_pos_t i = x.popcount(); 
	for (filter_pos_t max_slack = 0; i > 0; ++max_slack) {
		--i;
		if (roots[i]) {
			// if the trie is not empty we push the root node onto the stack.
			//
			exploration_state S[filter_t::WIDTH];
			unsigned int head = 0;

			node * n = roots[i];	// current node in the trie exploration
			filter_pos_t slack = max_slack; // number of times we can
											// ignore a 1-bits in the
											// current node
			for (;;) {
				if (n->key.subset_of(x)) {
					if (h.match(n->value))
						return;
					if (n->pos < filter_t::WIDTH) {
						if (n->right) // we don't need to check
									  // x[n->pos], which is implied
									  // by the branch condition
									  // n->key.subset_of(x)
							goto explore_right_subtree_first;
						else if (n->left)
							goto explore_left_subtree_only_if_have_slack;
					}
				} else if (n->pos < filter_t::WIDTH && n->key.prefix_subset_of(x, n->pos)) {
					if (x[n->pos]) {
						if (n->right) {
						explore_right_subtree_first:
							if (slack > 0 && n->left) {
								S[head].n = n->left;
								S[head].slack = slack - 1;
								++head;
							}
							n = n->right;
							continue;
						} else if (n->left) {
						explore_left_subtree_only_if_have_slack:
							if (slack > 0) {
								n = n->left;
								--slack;
								continue;
							}
						}
					} else if (n->left) {
						n = n->left;
						continue;
					}
				}
				if (head > 0) {
					--head;
					n = S[head].n;
					slack = S[head].slack;
				} else
					break;
			}
		}
	}
}

#endif
