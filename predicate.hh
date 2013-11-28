#ifndef PREDICATE_HH_INCLUDED
#define PREDICATE_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef NODE_USES_MALLOC
#define NODE_USES_MALLOC
#endif

#include <iostream>
#include <cstdint>
#include <cstdlib>

#ifdef NODE_USES_MALLOC
#include <new>
#endif

#include "bv192.hh"
#include "params.h"

typedef bv192 filter_t;

/** interface identifier */ 
typedef uint16_t interface_t;

/** tree identifier */ 
typedef uint16_t tree_t;

/** set of special tags that can be used in the matching algorithm */
typedef struct tags{
    filter_t yt;
    filter_t tw;
    filter_t blog;
    filter_t del;
    filter_t bt;
    
    tags(std::string const& y, std::string const& t, std::string const& bl,
         std::string const& d, std::string const& b): yt(y), tw(t), blog(bl), del(d), bt(b) {}

} tags_t;

/** tree--interface pair */ 
class tree_interface_pair {
// 
// ASSUMPTIONS: 
//   1. the router has at most 2^13 = 8192 interfaces
//   2. the are at most 2^3 = 8 interfaces
//
public:
	tree_interface_pair(tree_t t, interface_t ifx)
		: tree(t), interface(ifx) {};
	
	tree_t tree : 3;
	interface_t interface : 13;

	bool operator < (const tree_interface_pair &x) const {
		return (tree < x.tree || (tree == x.tree && interface < x.interface));
	}
	bool operator == (const tree_interface_pair & rhs) const {
		return (tree == rhs.tree && interface == rhs.interface);
	}
	bool equals(tree_t t, interface_t ifx) const {
		return (tree == t && interface == ifx) ;
	}
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
class filter_handler;
class filter_const_handler;
class match_handler;

class predicate {   

public:
    predicate(): root(), filter_count(0), t(YOUTUBE_TAG,TWITTER_TAG,BLOG_TAG,DEL_TAG,BTORRENT_TAG){};
    ~predicate() { destroy(); }


	class node;

    /** sets the tree mask for the filter x
	 */ 
	void set_mask(const filter_t & x, tree_t tree);

	/** adds a filter, without adding anything to it
	 */ 
	node * add(const filter_t & x);

	/** adds a filter together with the association to a
	 *  tree--interface pair
	 */ 
	node * add(const filter_t & x, tree_t t, interface_t i);

	/** modular matching function (subset search)
	 */
	void match(const filter_t & x, tree_t t, match_handler & h) const;

	/** exact-match filter search
	 */
	node * find(const filter_t & x) const;

    /** return if a subset of x is found on the interface i on the tree t. 
    * uses matcher_exists defined in router.hh
    */
    void exists_subset(const filter_t & x, tree_t t, interface_t i, match_handler & h) const;

    /** returns true if the filter exists on interface i and on tree t. 
     */
    bool exists_filter(const filter_t & x, tree_t t, interface_t i) const;

    /** iteretes over the trie to find all the super set of x on the give tree and interface
    * matcher_collect_supersets defines in router.hh collects all the found supersets
    */
    void find_supersets_on_ifx(const filter_t & x, tree_t t, interface_t i, match_handler & h) const;

    /** iteretes over the trie to find all the super set of x on the give tree on all the interfaces
    * matcher_collect_supersets defines in router.hh collects all the found supersets
    */
    void find_supersets(const filter_t & x, tree_t t, match_handler & h) const;

    /** count all the subsets of x on a given trie and return them divided by interfaces
    * matcher_count_subsets_by_ifx is defined in router.hh and stores the results in a map
    */
    void count_subsets_by_ifx(const filter_t & x, tree_t t, match_handler & h) const;


	/** processes the subsets of the given filter
	 *
	 *  This is the moduler subset-search.  The predicate applies the
	 *  given handler to the subsets of the given filter.  More
	 *  specifically, the predicate calls the handle_filter method of
	 *  the handler, which may terminate the search by returning true.
	 */
	void find_subsets_of(const filter_t & x, filter_const_handler & h) const;
	void find_supersets_of(const filter_t & x, filter_const_handler & h) const;
    void find_subsets_of(const filter_t & x, tree_t t, filter_const_handler & h) const;
	void find_subsets_of(const filter_t & x, filter_handler & h);
	void find_supersets_of(const filter_t & x, filter_handler & h);

	void remove(const filter_t & x,tree_t t, interface_t i);
	void clear();

	/** number of unique filters in the predicate */
	unsigned long size() const {
		return filter_count;
	}

	/** a node in the PATRICIA trie representing the predicate, and
	 *  also a filter descriptor with an associated set of
	 *  tree--interface pairs.
	 */
    class node {
		friend class predicate;
    public:
        const filter_t key;

		node * left;
		node * right;
        
//	public:
		
        unsigned char tree_mask;
        
	private:
		const filter_t::pos_t pos;

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

        void add_tree_to_mask(tree_t t){
             tree_mask |= 1 << t;
        }

        void init_tree_mask(node * n){
            tree_mask |= n->tree_mask;
        }
        
        bool match_tree(tree_t t){
            return tree_mask & (1 << t);
        }
        

	private:
		// create a stand-alone NULL node, this constructor is used
		// ONLY for the root node of the PATRICIA trie.
		//
		node() 
			: key(), left(this), right(this), tree_mask(0), pos(filter_t::NULL_POSITION), 
			  pairs_count(0) {}

		// creates a new node connected to another (child) node
		//
		node(filter_t::pos_t p, const filter_t & k, node * next) 
			: key(k), tree_mask(0), pos(p), pairs_count(0) {
			if (k[p]) {
				left = next;
				right = this;
			} else {
				left = this;
				right = next;
			}
		}

		void remove_last_pair();
		
		~node() {
			if (pairs_count > LOCAL_PAIRS_CAPACITY)
			    free(external_pairs);
		}

#ifdef NODE_USES_MALLOC
		static void * operator new (size_t s) {
			return malloc(s);
		}

		static void operator delete (void * p) {
			free(p);
		}
#endif
    };

    node root;
	unsigned long filter_count;
    tags_t t;  

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
	virtual bool handle_filter(const filter_t & filter, predicate::node & n) = 0;
};

class filter_const_handler {
public:
	// this will be called by predicate::find_subsets_of()
	// and predicate::find_supersets_of().  The return value
	// indicates whether the search for subsets or supersets should
	// stop.  So, if this function returns TRUE, find_subsets_of or
	// find_supersets_of will terminate immediately.
	// 
	virtual bool handle_filter(const filter_t & filter, const predicate::node & n) = 0;
};

class match_handler {
public:
	// this will be called by predicate::match().  The return value
	// indicates whether the search for matching filters should stop.
	// So, if this function returns TRUE, match() will terminate
	// immediately.
	// 
	virtual bool match(const filter_t & filter, tree_t tree, interface_t ifx) = 0;
};

#endif
