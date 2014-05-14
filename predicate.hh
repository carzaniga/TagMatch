#ifndef PREDICATE_HH_INCLUDED
#define PREDICATE_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef NODE_USES_MALLOC
#define NODE_USES_MALLOC
#endif


#include "params.h"
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <map>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <boost/lockfree/queue.hpp>



#ifdef NODE_USES_MALLOC
#include <new>
#endif

#ifdef WITH_BV192
#include "bv192.hh"

typedef bv192 filter_t;
#else
#include "bv.hh"
typedef bv<192> filter_t;
#endif


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
class p_params;

class tree_matcher;
class tree_ifx_matcher;

class synch_counter{
private:
    volatile unsigned int c;


public:
    
    synch_counter(): c(0) {};

    void inc(){
        unsigned int old_s, new_s;
    
        do{
            old_s = c;
            new_s = old_s + 1;
        }while(!__sync_bool_compare_and_swap(&c, old_s, new_s));
    }   

    void dec(){
        unsigned int old_s, new_s;
    
        do{
            old_s = c;
            new_s = old_s - 1;
        }while(!__sync_bool_compare_and_swap(&c, old_s, new_s));
    }

    bool done(){
        if(c==0)
            return true;
        return false;
    }
};



class predicate {      
public:
    predicate(unsigned int nf): filter_count(0) , N_FILTERS(nf), t(YOUTUBE_TAG,TWITTER_TAG,BLOG_TAG,DEL_TAG,BTORRENT_TAG) {

		static const unsigned int TOT_FILTERS = 63651601;
		//static const unsigned int N_FILTERS = 37400000;

		// this is a static configuration parameter representing the
		// distribution of hamming weights derived from the workload.  We
		// assume that such a configuration parameter can be derived from
		// a simple statistical analysis of the workload and, more
		// importantly, that the distribution would be *stable*.
		//
		// These are the number of million filters for each hamming wight
		// in a workload with 91092205 filters, corresponding to the
		// (compressed) set of filters corresponding to a population of
		// 500 million users.
		//
		static const unsigned int Hamming_Weight_Dist[filter_t::WIDTH] = {
			0,  0,  0,  0,  0,  0,  0,  1,  1,  1,      //0   --  9
			1,  3, 18, 39,  1,  1,  1,  1,  1,  1,      //10  --  19
			1,  1,  1,  1,  1,  1,  1,  1,  1,  1,      //20  --  29
			1,  1,  1,  1,  1,  1,  1,  1,  1,  1,      //30  --  39
			1,  1,  1,  1,  1,  1,  1,  1,  1,  1,      //40  --  49
			1,  1,  1,  1,  1,  1,  1,  1,  1,  1,      //50  --  59    
			1,  1,  1,  1,  1,  1,  1,  1,  1,  1,      //60  --  69
			1,  1,  1,  1,  0,  0,  0,  0,  0,  0,      //70  --  79    
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,      //80  --  89      
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,      //90  --  99
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,      //100 --  109
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,      //110 --  119
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,      //120 --  129
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,      //130 --  139
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,      //140 --  149
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,      //150 --  159
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,      //160 --  169
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,      //170 --  179
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,      //180 --  189
			0,  0                                       //190 --  191
		};

        for(filter_t::pos_t i =0; i< filter_t::WIDTH; i++) {
			if (Hamming_Weight_Dist[i]) {
				unsigned n = N_FILTERS*Hamming_Weight_Dist[i] / TOT_FILTERS;
				if (n == 0)
					n = 1;
				roots[i].set_size(n);
			}
		}

        job_queue_head=0;
        job_queue_tail=0;
        start_threads();
    };

    ~predicate() {
        stop_threads();
        destroy(); 
    }

	class node;
    class p_node;

    /** sets the tree mask for the filter x
	 */ 
	void set_mask(const filter_t & x, tree_t tree);

	/** adds a filter, without adding anything to it
	 */ 
	node * add(const filter_t & x, node & root);

	/** adds a filter together with the association to a
	 *  tree--interface pair
	 */ 
	node * add(const filter_t & x, tree_t t, interface_t i);
    void add_set_of_filters(std::map<filter_t,std::vector<tree_interface_pair>> & x);

    //void computes_bootstrap_update(std::vector<std::map<filter_t,std::vector<tree_interface_pair>>> & output, tree_t t, interface_t i);
    //void computes_bootstrap_update_on_a_trie(std::vector<std::map<filter_t,std::vector<tree_interface_pair>>> & output, tree_t t, 
                                                   // interface_t i, filter_t::pos_t index, node & root);
     

	/** modular matching function (subset search)
	 */
	void match(const filter_t & x, tree_t t, match_handler & h) const;

	/** exact-match filter search
	 */
	node * find(const filter_t & x, node & root) const;

    /** return if a subset of x is found on the interface i on the tree t. 
    * uses matcher_exists defined in router.hh
    */
    void exists_subset(const filter_t & x, tree_t t, interface_t i, match_handler & h);

    /** returns true if the filter exists on interface i and on tree t. 
     */
    bool exists_filter(const filter_t & x, tree_t t, interface_t i) const;

    /** iteretes over the trie to find all the super set of x on the give tree and interface
    * matcher_collect_supersets defines in router.hh collects all the found supersets
    */
    void find_supersets_on_ifx(const filter_t & x, tree_t t, interface_t i, match_handler & h);

    /** iteretes over the trie to find all the super set of x on the given tree on all the interfaces
    * matcher_collect_supersets defines in router.hh collects all the found supersets
    *
    *
    */
    void find_supersets(const filter_t & x, tree_t t, match_handler & h);

    /** count all the subsets of x on a given trie and return them divided by interfaces
    * matcher_count_subsets_by_ifx is defined in router.hh and stores the results in a map
    */
    void count_subsets_by_ifx(const filter_t & x, tree_t t, match_handler & h);

	/** processes the subsets of the given filter
	 *
	 *  This is the moduler subset-search.  The predicate applies the
	 *  given handler to the subsets of the given filter.  More
	 *  specifically, the predicate calls the handle_filter method of
	 *  the handler, which may terminate the search by returning true.
	 */
	void find_subsets_of(const filter_t & x, node & root, filter_const_handler & h) const;
	void find_supersets_of(const filter_t & x, node & root, filter_const_handler & h) const;
    //void find_subsets_of(const filter_t & x, tree_t t, filter_const_handler & h) const;
    //void find_subsets_of(const filter_t & x, const * node root, filter_const_handler & h ) const;
	//void find_subsets_of(const filter_t & x, filter_handler & h);
	//void find_supersets_of(const filter_t & x, filter_handler & h);

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
    }__attribute__ ((aligned(64)));

    class p_node {
        friend class predicate;
        public: 
        filter_t::pos_t size;
        node * tries;
		p_node(): size(0), tries(0) {};

		// this is an initialization performed ONCE for each p_node
		// 
        void set_size(filter_t::pos_t n) {
			size = n;
            if (n>0) {
                tries = new node [size];
            } else
                tries=NULL;
        }

		node & trie_of(const filter_t & x) const {
			return tries[x.hash(size)];
		}

		~p_node() {
            if (size > 0)
                delete [] tries;
        }
    };

    p_node roots[filter_t::WIDTH];    
	unsigned long filter_count;
    unsigned int N_FILTERS;
    tags_t t;  

    void destroy();

    

    private:
         static const unsigned int THREAD_COUNT = 16;
    static const unsigned int JOB_QUEUE_SIZE = 1024;
    p_params * job_queue[JOB_QUEUE_SIZE];
    boost::lockfree::queue<p_params *, boost::lockfree::capacity<JOB_QUEUE_SIZE>> q;


    volatile unsigned int job_queue_head;		// position of the first element in the queue
    volatile unsigned int job_queue_tail;		// one-past position of the last element in the queue

     std::mutex job_queue_mtx;
    std::condition_variable job_queue_producer_cv;
    std::condition_variable job_queue_consumers_cv;

     std::thread * thread_pool[THREAD_COUNT];

    public:

#define MTX 1
#if MTX
    void job_enqueue(p_params * p) {
	    size_t tail_plus_one;
	    std::unique_lock<std::mutex> lock(job_queue_mtx);
    try_enqueue:
	    tail_plus_one = (job_queue_tail + 1) % JOB_QUEUE_SIZE;

	    if (tail_plus_one == job_queue_head) { // full queue 
		    job_queue_producer_cv.wait(lock);
	    	goto try_enqueue;
	    }
	    job_queue[job_queue_tail] = p;
	    job_queue_tail = tail_plus_one;

	    job_queue_consumers_cv.notify_all();
    }

    p_params * job_dequeue() {
	    std::unique_lock<std::mutex> lock(job_queue_mtx);

    try_dequeue:
	    if (job_queue_head == job_queue_tail) { // empty queue 
		    job_queue_consumers_cv.wait(lock);
		    goto try_dequeue;
	    }

        p_params * p = job_queue[job_queue_head];
	    job_queue_head = (job_queue_head + 1) % JOB_QUEUE_SIZE;

	    job_queue_producer_cv.notify_one();
        
	    return p;
    }
#else

    void job_enqueue(p_params * p) {
        while (!q.push(p))
            asm volatile("rep; nop" ::: "memory");
    }

    p_params * job_dequeue() {
        p_params * p;
        while (!q.pop(p))
            asm volatile("rep; nop" ::: "memory");
        return p;
    }

#endif
    void thread_loop(unsigned int id);
           
    void start_threads(){
        //std::cout << "START" << std::endl;
        for(unsigned int i = 0; i < THREAD_COUNT; ++i)
	        thread_pool[i] = new std::thread(&predicate::thread_loop, this, i);
    }

    void stop_threads(){
        //std::cout << "STOP" << std::endl;
        for(unsigned int i = 0; i < THREAD_COUNT; ++i)
            job_enqueue(0);
        for(unsigned int i = 0; i < THREAD_COUNT; ++i)
            thread_pool[i]->join();
        for(unsigned int i = 0; i < THREAD_COUNT; ++i)
            delete(thread_pool[i]);   
    }

    
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
