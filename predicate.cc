#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <string>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>

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

// A table of tree--interface pairs.  This is essentially a wrapper
// for a pointer to an array of tree_interface_pair objects plus a
// size.
//
class tree_interface_table {
public:
	tree_interface_table(): table(0) {};
	~tree_interface_table() {
		if (table)
			free(table);
	}

	void add_tree_interface_pair(tree_t t, interface_t i) {
		table = ti_array::add_pair(table, t, i);
	}

	const tree_interface_pair * begin() const {
		if (!table)
			return 0;
		return table->begin();
	}

	const tree_interface_pair * end() const {
		if (!table)
			return 0;
		return table->end();
	}

private:
	class ti_array {
		uint16_t size;
		tree_interface_pair pairs[1];

	public:
		static ti_array * add_pair(ti_array * entry, tree_t t, interface_t i);

		const tree_interface_pair * begin() const {
			return pairs;
		}

		const tree_interface_pair * end() const {
			return pairs + size;
		}
		
	private:
		ti_array();
		static const uint16_t ALLOCATION_UNIT_SIZE = 16; // bytes
	};

	ti_array * table;
};

tree_interface_table::ti_array * 
tree_interface_table::ti_array::add_pair(ti_array * entry, tree_t t, interface_t i) {
	if (!entry) {
		entry = (ti_array *)malloc(ALLOCATION_UNIT_SIZE);
		allocated_bytes += ALLOCATION_UNIT_SIZE;
		entry->size = 0;
	} else {
		assert(entry->size < 0xffff);

#ifdef PARANOIA_CHECKS
		tree_interface_pair ti(t,i);
		for(const tree_interface_pair * i = entry->begin(); i != entry->end(); ++i) 
			if (*i == ti)
				return entry;
#endif

		size_t byte_pos = offsetof(ti_array, pairs) + entry->size * sizeof(tree_interface_pair);
		if (byte_pos % ALLOCATION_UNIT_SIZE == 0) {// full table
			entry = (ti_array *)realloc(entry, byte_pos + ALLOCATION_UNIT_SIZE);
			allocated_bytes += ALLOCATION_UNIT_SIZE;
		}
	}
	entry->pairs[entry->size] = tree_interface_pair(t,i);
	entry->size += 1;
	return entry;
}

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
// functions defined in the following three interface classes.
// 
class filter_handler {
public:
	// this will be called by predicate::find_subsets_of()
	// and predicate::find_supersets_of().  The return value
	// indicates whether the search for subsets or supersets should
	// stop.  So, if this function returns TRUE, find_subsets_of or
	// find_supersets_of will terminate immediately.
	// 
	virtual bool handle_filter(const bv192 & filter, tree_interface_table & table) = 0;
};

class filter_const_handler {
public:
	// this will be called by predicate::find_subsets_of()
	// and predicate::find_supersets_of().  The return value
	// indicates whether the search for subsets or supersets should
	// stop.  So, if this function returns TRUE, find_subsets_of or
	// find_supersets_of will terminate immediately.
	// 
	virtual bool handle_filter(const bv192 & filter, const tree_interface_table & table) = 0;
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

class predicate {
public:
    predicate(): root(), size(0) {};
    ~predicate() { destroy(); }

	// non-modular, basic matching function
	//
	void match(const bv192 & x, tree_t t) const;
	void add(const bv192 & x, tree_t t, interface_t i);

	// modular matching function
	//
	void match(const bv192 & x, tree_t t, match_handler & h) const;

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

private:
    class node {
    public:
		bv192::pos_t pos;
		const bv192 key;
		tree_interface_table ti_table;
		node * left;
		node * right;

		// creates a stand-alone NULL node, this constructor is used
		// ONLY for the root node of the PATRICIA trie.
		//
		node() 
			: pos(bv192::NULL_POSITION), key(), left(this), right(this) {}

		// creates a new node connected to another (child) node
		//
		node(bv192::pos_t p, const bv192 & k, node * next) 
			: pos(p), key(k) {
			if (k[p]) {
				left = next;
				right = this;
			} else {
				left = this;
				right = next;
			}
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

	const node * find(const bv192 & x) const;
	node * add(const bv192 & x);

    node root;
	unsigned long size;

    void destroy();

	// this is the handler we use to perform the tree matching.  The
	// predicate subset search finds subsets of the given filter, and
	// this handler does the tree matching on the corresponding
	// tree_interface pairs.
	// 
	class tree_matcher : public filter_const_handler {
	public:
		tree_matcher(tree_t t, match_handler & mh): tree(t), matcher(mh) {}
		virtual bool handle_filter(const bv192 & filter, const tree_interface_table & table);
	private:
		const tree_t tree;
		match_handler & matcher;
	};
};

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

bool predicate::tree_matcher::handle_filter(const bv192 & filter, const tree_interface_table & table) {
	for(const tree_interface_pair * ti = table.begin(); ti != table.end(); ++ti)
		if (ti->tree == tree)
			if (matcher.match(filter, tree, ti->interface))
				return true;
	return false;
}

void predicate::match(const bv192 & x, tree_t t, match_handler & h) const {
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
			for(const tree_interface_pair * ti = n->ti_table.begin(); ti != n->ti_table.end(); ++ti)
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

void predicate::add(const bv192 & x, tree_t t, interface_t i) {
	node * n = add(x);
	n->ti_table.add_tree_interface_pair(t, i);
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

void predicate::find_subsets_of(const bv192 & x, filter_const_handler & h) const {
	const node * S[192];
	unsigned int head = 0;

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= 192);

		const node * n = S[--head];
		if (n->key.subset_of(x)) {
			if (h.handle_filter(n->key, n->ti_table))
				return;
		}
		if (n->pos > n->left->pos) 
			S[head++] = n->left;

		if (x[n->pos] && n->pos > n->right->pos)
			S[head++] = n->right;
	}
}

void predicate::find_supersets_of(const bv192 & x, filter_const_handler & h) const {
	const node * S[192];
	unsigned int head = 0;

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= 192);
		const node * n = S[--head];
		if (x.subset_of(n->key)) {
			if (h.handle_filter(n->key, n->ti_table))
				return;
		}
		if (n->pos > n->right->pos) 
			S[head++] = n->right;

		if (!x[n->pos] && n->pos > n->left->pos)
			S[head++] = n->left;
	}
}

void predicate::find_subsets_of(const bv192 & x, filter_handler & h) {
	node * S[192];
	unsigned int head = 0;

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= 192);

		node * n = S[--head];
		if (n->key.subset_of(x)) {
			if (h.handle_filter(n->key, n->ti_table))
				return;
		}
		if (n->pos > n->left->pos) 
			S[head++] = n->left;

		if (x[n->pos] && n->pos > n->right->pos)
			S[head++] = n->right;
	}
}

void predicate::find_supersets_of(const bv192 & x, filter_handler & h) {
	node * S[192];
	unsigned int head = 0;

	if (root.pos > root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= 192);
		node * n = S[--head];
		if (x.subset_of(n->key)) {
			if (h.handle_filter(n->key, n->ti_table))
				return;
		}
		if (n->pos > n->right->pos) 
			S[head++] = n->right;

		if (!x[n->pos] && n->pos > n->left->pos)
			S[head++] = n->left;
	}
}

class filter_printer : public filter_const_handler {
public:
	filter_printer(std::ostream & s): os(s) {};

	virtual bool handle_filter(const bv192 & filter, const tree_interface_table & table);
private:
	std::ostream & os;
};

bool filter_printer::handle_filter(const bv192 & filter, const tree_interface_table & table) {
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
		} else if (command == "p") {
			bv192 filter(filter_string);
			P.find_subsets_of(filter, filter_output);
			P.find_supersets_of(filter, filter_output);
		} else {
			std::cerr << "unknown command: " << command << std::endl;
		}
	}
	if (match_count.get_match_count() != 0) 
		std::cout << "match count: " << match_count.get_match_count() << std::endl;

	return 0;
}
