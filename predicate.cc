#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <string>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <malloc.h>

#ifndef NODE_USES_MALLOC
#define NODE_USES_MALLOC
#endif

#ifdef NODE_USES_MALLOC
#include <new>
#endif

#include "bv192.h"

#define NDEBUG

// this is to explicitly keep track of our own memory allocations
// 
static size_t allocated_bytes = 0;

// ASSUMPTION: the router has at most 2^13 = 8192 interfaces
// 
typedef uint16_t interface_t;

// ASSUMPTION: the are at most 2^3 = 8 interfaces
// 
typedef uint8_t tree_t;

class tree_interface_pair {
public:
	tree_interface_pair(tree_t t, interface_t ifx)
		: tree(t), interface(ifx) {};
	
	tree_t tree : 3;
	interface_t interface : 13;

};

// 
// GENERIC DESIGN OF THE PREDICATE DATA STRUCTURE:
// 
// the predicate class is intended to be generic with respect to the
// processing of filter entries.  What this means is that the
// predicate class implements generic methods to add or find filters,
// or to find subsets or supersets of a given filter.  However, the
// predicate class does not itself implement the actual matching or
// processing functions that operate on the marching or subset or
// superset filters.  Those are instead delegated to some "handler"
// functions defined in the three interface classes filter_handler,
// filter_const_handler, and match_handler.
// 
class filter_handler; 			// defined after predicate
class filter_const_handler; 	// defined after predicate
class match_handler;			// defined after predicate

class predicate {
public:
    predicate(): root(), size(0) {};
    ~predicate() { destroy(); }

	class node;

	// adds a filter, without adding anything to it
	// 
	node * add(const bv192 & x);

	// adds a filter together with the association to a
	// tree--interface pair
	// 
	node * add(const bv192 & x, tree_t t, interface_t i);

	// non-modular, basic matching function (subset search)
	//
	void match(const bv192 & x, tree_t t) const;

	// modular matching function (subset search)
	//
	void match(const bv192 & x, tree_t t, match_handler & h) const;

	// exact-match filter search
	//
	const node * find(const bv192 & x) const;
	node * find(const bv192 & x);

	void find_subsets_of(const bv192 & x, filter_const_handler & h) const;
	void find_supersets_of(const bv192 & x, filter_const_handler & h) const;
	void find_subsets_of(const bv192 & x, filter_handler & h);
	void find_supersets_of(const bv192 & x, filter_handler & h);


	void clear() {
		destroy();
		root.left = &root;
		root.right = &root;
		size = 0;
	}

	static size_t sizeof_node() {
		return sizeof(node);
	}

	unsigned long get_size() const {
		return size;
	}

    class node {
		friend class predicate;

		node * left;
		node * right;

	public:
		const bv192 key;

	private:
		const bv192::pos_t pos;

		static const uint16_t EXT_PAIRS_ALLOCATION_UNIT = 16; 

		static const unsigned int LOCAL_PAIRS_CAPACITY = 4;

		uint16_t pairs_count;

		union {
			tree_interface_pair local_pairs[LOCAL_PAIRS_CAPACITY];
			tree_interface_pair * external_pairs;
		};

	public:
		void add_pair(tree_t t, interface_t i);

		// number of tree--interface pairs associated with this filter
		//
		uint16_t ti_size() const {
			return pairs_count;
		}

		// pointer to the first tree--interface pair
		//
		tree_interface_pair * ti_begin() {
			return (pairs_count <= LOCAL_PAIRS_CAPACITY) ? local_pairs : external_pairs;
		}

		// pointer to the first tree--interface pair
		//
		const tree_interface_pair * ti_begin() const {
			return (pairs_count <= LOCAL_PAIRS_CAPACITY) ? local_pairs : external_pairs;
		}

		// pointer to one-past the tree--interface pair
		//
		tree_interface_pair * ti_end() {
			return ti_begin() + pairs_count;
		}

		// pointer to one-past the tree--interface pair
		//
		const tree_interface_pair * ti_end() const {
			return ti_begin() + pairs_count;
		}

	private:
		// create a stand-alone NULL node, this constructor is used
		// ONLY for the root node of the PATRICIA trie.
		//
		node() 
			: left(this), right(this), key(), pos(bv192::NULL_POSITION), 
			  pairs_count(0) {}

		// creates a new node connected to another (child) node
		//
		node(bv192::pos_t p, const bv192 & k, node * next) 
			: key(k), pos(p), pairs_count(0) {
			if (k[p]) {
				left = next;
				right = this;
			} else {
				left = this;
				right = next;
			}
		}

		~node() {
			if (pairs_count > LOCAL_PAIRS_CAPACITY)
			    free(external_pairs);
		}

#ifdef NODE_USES_MALLOC
		static void * operator new (size_t s) {
			allocated_bytes += s;
			return malloc(s);
		}

		static void operator delete (void * p) {
			free(p);
		}
#endif
    };

    node root;
	unsigned long size;

    void destroy();
};

class filter_handler {
public:
	// this will be called by predicate::find_subsets_of()
	// and predicate::find_supersets_of().  The return value
	// indicates whether the search for subsets or supersets should
	// stop.  So, if this function returns TRUE, find_subsets_of or
	// find_supersets_of will terminate immediately.
	// 
	virtual bool handle_filter(const bv192 & filter, predicate::node & n) = 0;
};

class filter_const_handler {
public:
	// this will be called by predicate::find_subsets_of()
	// and predicate::find_supersets_of().  The return value
	// indicates whether the search for subsets or supersets should
	// stop.  So, if this function returns TRUE, find_subsets_of or
	// find_supersets_of will terminate immediately.
	// 
	virtual bool handle_filter(const bv192 & filter, const predicate::node & n) = 0;
};

class match_handler {
public:
	// this will be called by predicate::match().  The return value
	// indicates whether the search for matching filters should stop.
	// So, if this function returns TRUE, match() will terminate
	// immediately.
	// 
	virtual bool match(const bv192 & filter, tree_t tree, interface_t ifx) = 0;
};

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
		allocated_bytes += bytes_needed;
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
			allocated_bytes += EXT_PAIRS_ALLOCATION_UNIT;
		}
		external_pairs[pairs_count].tree = t;
		external_pairs[pairs_count].interface = i;
		pairs_count += 1;
	}
}

void predicate::destroy() {
	if (root.pos <= root.left->pos)
		return;

	node * S[192];
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
	virtual bool handle_filter(const bv192 & filter, const predicate::node & n);
private:
	const tree_t tree;
	match_handler & matcher;
};

bool tree_matcher::handle_filter(const bv192 & filter, const predicate::node & n) {
	for(const tree_interface_pair * ti = n.ti_begin(); ti != n.ti_end(); ++ti)
		if (ti->tree == tree)
			if (matcher.match(filter, tree, ti->interface))
				return true;
	return false;
}

void predicate::match(const bv192 & x, tree_t t, match_handler & h) const {
	//
	// this is the modular matching function that uses the above match
	// handler through the modular find_subset_of function
	//
	tree_matcher matcher(t,h);
	find_subsets_of(x, matcher);
}

void predicate::match(const bv192 & x, tree_t t) const {
	// 
	// this is the non-modular matching function.  It basically
	// replicates the functionality of find_subsets_of() but then it
	// directly uses the results to select the tree--interface pair
	// corresponding to the given tree t.
	//
	const node * S[192];
	unsigned int head = 0;

	std::cout << "->";

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= 192);

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

predicate::node * predicate::add(const bv192 & x, tree_t t, interface_t i) {
	node * n = add(x);
	n->add_pair(t, i);
	return n;
}

predicate::node * predicate::add(const bv192 & x) {
	node * prev = &root;
	node * curr = root.left;

	while(prev->pos > curr->pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}
	if (x == curr->key)
		return curr;

	bv192::pos_t pos = bv192::most_significant_diff_pos(curr->key, x);

	prev = &root;
	curr = root.left;
	
	while(prev->pos > curr->pos && curr->pos > pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}

	// now we insert the new node between prev and curr
	++size;
	if (prev->pos < bv192::NULL_POSITION && x[prev->pos]) {
		return prev->right = new node(pos, x, curr);
	} else {
		return prev->left = new node(pos, x, curr);
	}
}

const predicate::node * predicate::find(const bv192 & x) const {
	const node * prev = &root;
	const node * curr = root.left;

	while(prev->pos > curr->pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}
	return (x == curr->key) ? curr : 0;
}

predicate::node * predicate::find(const bv192 & x) {
	const node * prev = &root;
	node * curr = root.left;

	while(prev->pos > curr->pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}
	return (x == curr->key) ? curr : 0;
}

void predicate::find_subsets_of(const bv192 & x, filter_const_handler & h) const {
	//
	// this is a non-recoursive (i.e., iterative) exploration of the
	// PATRICIA trie that looks for subsets.  The pattern is almost
	// exactly the same for supersets (see below).  We use a stack S
	// to keep track of the visited nodes, and we visit new nodes
	// along a subset (resp. superset) prefix.
	// 
	const node * S[192];
	unsigned int head = 0;

	// if the trie is not empty we push the root node onto the stack.
	// The true root is root.left, not root, which is a sentinel node.
	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= 192);

		const node * n = S[--head];		// for each visited node n...

		if (n->key.subset_of(x)) {
			if (h.handle_filter(n->key, *n))
				return;
		} 
#if 0
		// If the subset relation does not hold for the prefix defined
		// by the current node -- meaning the prefix from position 191
		// to position n->pos (excluded) -- then we can prune the
		// exploration from this point on.  The technique seems most
		// effective for longer prefixes, thus it kicks in only when
		// n->pos < 50.
		// 
		// HOWEVER: this heuristic is excluded because it does not
		// seem to be effective in practice, perhaps because the
		// prefix_subset check is too expensive to yield a good gain.
		// 
		else if (n->pos < 50
				 && ! n->key.prefix_subset_of(x, n->pos + 1))
			continue;
#endif
		if (n->pos > n->left->pos) 
			S[head++] = n->left;

		if (n->pos > n->right->pos && x[n->pos])
			S[head++] = n->right;
	}
}

void predicate::find_supersets_of(const bv192 & x, filter_const_handler & h) const {
	//
	// see above: find_subsets_of(const bv192 & x, filter_const_handler & h) const
	//
	const node * S[192];
	unsigned int head = 0;

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= 192);

		const node * n = S[--head];

		if (x.subset_of(n->key)) 
			if (h.handle_filter(n->key, *n))
				return;

		if (n->pos > n->right->pos) 
			S[head++] = n->right;

		if (n->pos > n->left->pos && !x[n->pos])
			S[head++] = n->left;
	}
}

void predicate::find_subsets_of(const bv192 & x, filter_handler & h) {
	//
	// see above: find_subsets_of(const bv192 & x, filter_const_handler & h) const
	//
	node * S[192];
	unsigned int head = 0;

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= 192);

		node * n = S[--head];

		if (n->key.subset_of(x)) {
			if (h.handle_filter(n->key, *n))
				return;
		}
		if (n->pos > n->left->pos) 
			S[head++] = n->left;

		if (n->pos > n->right->pos && x[n->pos])
			S[head++] = n->right;
	}
}

void predicate::find_supersets_of(const bv192 & x, filter_handler & h) {
	//
	// see above: find_subsets_of(const bv192 & x, filter_const_handler & h) const
	//
	node * S[192];
	unsigned int head = 0;

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= 192);

		node * n = S[--head];

		if (x.subset_of(n->key)) 
			if (h.handle_filter(n->key, *n))
				return;

		if (n->pos > n->right->pos) 
			S[head++] = n->right;

		if (n->pos > n->left->pos && !x[n->pos])
			S[head++] = n->left;
	}
}

class filter_printer : public filter_const_handler {
public:
	filter_printer(std::ostream & s): os(s) {};

	virtual bool handle_filter(const bv192 & filter, const predicate::node & n);
private:
	std::ostream & os;
};

bool filter_printer::handle_filter(const bv192 & filter, const predicate::node & n) {
	filter.print(os);
	return false;
}

class match_printer : public match_handler {
public:
	match_printer(std::ostream & s): os(s) {};

	virtual bool match(const bv192 & filter, tree_t tree, interface_t ifx);
private:
	std::ostream & os;
};

bool match_printer::match(const bv192 & filter, tree_t tree, interface_t ifx) {
	os << " -> " << ifx << std::endl;
	filter.print(os);
	os << std::endl;
	return false;
}

class match_counter : public match_handler {
public:
	match_counter(): count(0) {};

	virtual bool match(const bv192 & filter, tree_t tree, interface_t ifx);
	unsigned long get_match_count() const {
		return count;
	}
private:
	unsigned long count;
};

bool match_counter::match(const bv192 & filter, tree_t tree, interface_t ifx) {
	++count;
	return false;
}

int main(int argc, char *argv[]) {

	std::string command;
	std::string tree;
	std::string interface;
	std::string filter_string;
	
	predicate P;

	filter_printer filter_output(std::cout);
	match_printer match_output(std::cout);
	match_counter match_count;

	unsigned int count = 0;
	unsigned int query_count = 0;

	while(std::cin >> command >> tree >> interface >> filter_string) {
		if (command == "+") {
			bv192 filter(filter_string);
			interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
			P.add(filter,t,i);
			++count;
			if ((count & 0xfff) == 0) {
				std::cout << "N=" << count << "  N'=" << P.get_size() 
						  << "  Mem=" << (allocated_bytes >> 20) << "MB  Avg=" << (allocated_bytes / count) << "B  Avg'=" << (allocated_bytes / P.get_size()) << "B      \r";
			}
		} else if (command == "?") {
			bv192 filter(filter_string);
			std::cout << "matching: " << std::endl << filter_string << std::endl;
			tree_t t = atoi(tree.c_str());
			P.match(filter, t);
		} else if (command == "!") {
			bv192 filter(filter_string);
			tree_t t = atoi(tree.c_str());
			P.match(filter, t, match_count);
			if (query_count==0)
				std::cout << std::endl;
			++query_count;
			if ((query_count & 0xff) == 0) {
				std::cout << "Q=" << query_count << "  Match=" << match_count.get_match_count() << " \r";
			}
		} else if (command == "p") {
			bv192 filter(filter_string);
			P.find_subsets_of(filter, filter_output);
			P.find_supersets_of(filter, filter_output);
		} else {
			std::cerr << "unknown command: " << command << std::endl;
		}
	}
	std::cout << std::endl << "Final statistics:" << std::endl
			  << "N=" << count << "  N'=" << P.get_size() 
			  << "  Mem=" << (allocated_bytes >> 20) 
			  << "MB  Avg=" << (allocated_bytes / count) 
			  << "B  Avg'=" << (allocated_bytes / P.get_size()) << std::endl
			  << "Q=" << query_count << "  Match=" << match_count.get_match_count() << std::endl;

	return 0;
}
