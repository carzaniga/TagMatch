#ifndef update_hh_included
#define update_hh_included

#include <set>
#include <list>
#include <map>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "predicate.hh"

using namespace std;

class synch_filter_vector {
public:
    volatile int lock;
    std::vector<filter_t> filters;
    
    synch_filter_vector(): lock(0) {};

	void add (const filter_t & x){

        //__sync_lock_test_and_set returns the initial value of the variable 
        //that &lock points to. I can get the lock only if 
        //__sync_lock_test_and_set returns 0, that means that the lock was not
        //set by any other thread. If I manage to set the lock, I am the only
        //one that can access the critical section
        
        //while true spinn
        while(__sync_lock_test_and_set(&lock,1));
        
        //lock acquired
        filters.push_back(x);
        
        //__sync_lock_release release the lock and set &lock to 0
        __sync_lock_release(&lock);
    }
};


/** sets exists_match to true and return true. used in the exists_subsets
**/
class matcher_exists : public match_handler {
public:
	matcher_exists(): exists_match(0) {};
    
    bool get_match() const {
        return exists_match;
    }
	virtual bool match(const filter_t & filter, tree_t tree, interface_t ifx);
private: 
    bool exists_match;
};

/** finds all the supersets and collect them by interface
* - in add_filter is used in combination with a tree_ifx_mathcer
* (see predicate.cc) in order to get all the supersets of one 
* particular interface
* - in remove_filter we use a tree_matcher in predicate.cc to get
* all the supersets 
**/
class matcher_collect_supersets : public match_handler {
public:
	matcher_collect_supersets(synch_filter_vector & r, filter_t::pos_t i): to_remove(r), interface(i), lock(0) {};
    
    map<interface_t,vector<filter_t>> * get_supersets() {
        return &supersets;
    }

	virtual bool match(const filter_t & filter, tree_t tree, interface_t ifx);
private: 
    map<interface_t,vector<filter_t>> supersets;
    synch_filter_vector & to_remove;
    filter_t::pos_t interface;
    volatile int lock;
};


/** count the numebr of subsets on each interface **/
class matcher_count_subsets_by_ifx : public match_handler {
public:
	matcher_count_subsets_by_ifx(): lock(0) {};
    
    //actual_ifx is the interface where we wanto to send an update
    //delta_ifx is the interface from where we received the delta
    bool exists_subsets_on_other_ifx(interface_t actual_ifx,interface_t delta_ifx){
        for (map<interface_t,unsigned int>::iterator it=subsets.begin(); it!=subsets.end(); ++it){
            if(it->second!=0 && actual_ifx!=it->first && delta_ifx!=it->first)
                return true;
        }
        return false;
    }

	virtual bool match(const filter_t & filter, tree_t tree, interface_t ifx);
private: 
    //input interface from where we received the update
    interface_t i;
    map<interface_t,unsigned int> subsets;
    volatile int lock;
};

class matcher_get_out_interfaces : public match_handler {
public:
	matcher_get_out_interfaces(): lock(0) {};
    
	virtual bool match(const filter_t & filter, tree_t tree, interface_t ifx);
private: 
    //input interface from where we received the update
    interface_t i;
    set<interface_t> ifxs;
    volatile int lock;
};


class predicate_delta {
public:
    list<filter_t> additions;
    list<filter_t> removals;

    //load the sets additions and removals with minimal sets. interface i is the one that we want to 
    //skip reading the content of the map. this procedure is not really efficient 
    void create_minimal_delta(const filter_t &, map<interface_t,vector<filter_t>> &, const interface_t);
    
    //this fucntion merges two predicate delta minimizing the list of filters 
    void merge(const predicate_delta & d);
    
    void add_removal_filter(const filter_t & t);
    void add_additional_filter(const filter_t & t);
};


class synch_ifx_delta_map {
public:
    volatile int lock;
    std::map<interface_t,predicate_delta> & output;

    synch_ifx_delta_map(std::map<interface_t,predicate_delta> & o) : lock(0), output(o) {};
    
    void add (interface_t i, const predicate_delta & d){
        while(__sync_lock_test_and_set(&lock,1));
        output[i].merge(d);
        __sync_lock_release(&lock);
    }
};

class r_params {
public:
    bool add; //true: we need to call add_filter
              //false: we need to call remove_filter
    const filter_t f; //filters to add/remove
    tree_t t;
    interface_t i;
    synch_filter_vector * to_add; //used only in add
    synch_filter_vector * to_rm;
    synch_ifx_delta_map * output; //used only in remove

    r_params(bool add_, const filter_t & f_, tree_t t_, interface_t i_, synch_ifx_delta_map * output_, synch_filter_vector * to_add_, synch_filter_vector * to_rm_): add(add_), f(f_), t(t_), i(i_), to_add(to_add_), to_rm(to_rm_), output(output_) {};
};



class router {
    
private:
    predicate P;
    map<tree_t,vector<interface_t>> interfaces;


    static const unsigned int THREAD_COUNT = 8;
    static const unsigned int JOB_QUEUE_SIZE = 1024;
    r_params * job_queue[JOB_QUEUE_SIZE];

    unsigned int job_queue_head;		// position of the first element in the queue
    unsigned int job_queue_tail;		// one-past position of the last element in the queue

    volatile unsigned int synch_queue;
    mutex synch_queue_lock;
    condition_variable synch_queue_done;

    thread * thread_pool[THREAD_COUNT];


std::mutex job_queue_mtx;
std::condition_variable job_queue_producer_cv;
std::condition_variable job_queue_consumers_cv;

void job_enqueue(r_params * p) {
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

r_params * job_dequeue() {
	std::unique_lock<std::mutex> lock(job_queue_mtx);

 try_dequeue:
	if (job_queue_head == job_queue_tail) { // empty queue 
		job_queue_consumers_cv.wait(lock);
		goto try_dequeue;
	}

    r_params * p = job_queue[job_queue_head];
	job_queue_head = (job_queue_head + 1) % JOB_QUEUE_SIZE;

	job_queue_producer_cv.notify_one();
	return p;
}


void inc_synch_queue(){
    unsigned int old_s, new_s;
    
    do{
        old_s = synch_queue;
        new_s = old_s + 1;
    }while(!__sync_bool_compare_and_swap(&synch_queue, old_s, new_s));
}


void dec_synch_queue(){
   unsigned int old_s, new_s;
    
    do{
        old_s = synch_queue;
        new_s = old_s - 1;
    }while(!__sync_bool_compare_and_swap(&synch_queue, old_s, new_s));
     
    //synch_queue can go to 0 only once!
    if(synch_queue==0){
        std::unique_lock<std::mutex> lock(synch_queue_lock);
        synch_queue_done.notify_one();
    }
}

void thread_loop(unsigned int id) {
	r_params * p;
	while((p = job_dequeue())){
        if(p->add){
            add_filter(p->f, p->t, p->i, (*p->to_add), (*p->to_rm));
        }else{
            remove_filter(p->f, p->t, p->i, (*p->output), (*p->to_rm));
        }
        dec_synch_queue();
    }
}

    
                  
public:
    //nf is the number of expected filters
    router(unsigned int nf): P(nf) {
        job_queue_head=0;
        job_queue_tail=0;
        synch_queue=0;
    };
    ~router() {}


    /** adds a new filter to predicate P without checking the existence of a subset
    the filter x. It does not remove any superset of x from P. **/
    void add_filter_without_check (const filter_t & x, tree_t t, interface_t i);
   
    /** removes a filter from P without any check **/
    void remove_filter_without_check (const filter_t & x, tree_t t, interface_t i);
 
    /** adds a new filter to predicate P. returns true if the filter has to be sent to 
    all the neighbors, fasle otherwise.**/
    bool add_filter (const filter_t & x, tree_t t, interface_t i, synch_filter_vector & to_add, synch_filter_vector & to_rm);
    
    /** removes a filter from predicate P. **/
    void remove_filter (const filter_t & x, tree_t t, interface_t i, synch_ifx_delta_map & out_delta, synch_filter_vector & to_rm);

    /**  produces a set of predicate deltas as a result of applying d to P **/
    void apply_delta(map<interface_t,predicate_delta> & output, const predicate_delta & d, interface_t i, tree_t t);

    void add_ifx_to_tree(tree_t t, interface_t i);

    void start_threads();
    void stop_threads();


};
#endif
