#ifndef update_hh_included
#define update_hh_included

#include <set>
#include <list>
#include <map>
#include <mutex>
#include <thread>

#include "predicate.hh"

using namespace std;


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
	matcher_collect_supersets() {};
    
    map<interface_t,vector<filter_t>> * get_supersets() {
        return &supersets;
    }

	virtual bool match(const filter_t & filter, tree_t tree, interface_t ifx);
private: 
    map<interface_t,vector<filter_t>> supersets;
    mutex mtx;
};


/** count the numebr of subsets on each interface **/
class matcher_count_subsets_by_ifx : public match_handler {
public:
	matcher_count_subsets_by_ifx() {};
    
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
    mutex mtx;
};

class matcher_get_out_interfaces : public match_handler {
public:
	matcher_get_out_interfaces() {};
    
	virtual bool match(const filter_t & filter, tree_t tree, interface_t ifx);
private: 
    //input interface from where we received the update
    interface_t i;
    set<interface_t> interfaces;
    mutex mtx;
};


class predicate_delta {
public:
# if 0
    interface_t ifx;
    tree_t tree;

    predicate_delta(interface_t i, tree_t t): ifx(i), tree(t) {};
#endif
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

class synch_filter_vector {
public:
    mutex mtx;
    std::vector<filter_t> filters;
    
    synch_filter_vector () {};

	void add (const filter_t & x){
        mtx.lock();
        filters.push_back(x);
        mtx.unlock();
    }
};

class synch_ifx_delta_map {
public:
    mutex mtx;
    std::map<interface_t,predicate_delta> & output;

    synch_ifx_delta_map(std::map<interface_t,predicate_delta> & o) : output(o) {};
    
    void add (interface_t i, const predicate_delta & d){
        mtx.lock();
        output[i].merge(d);
        mtx.unlock();
    }
};


class router {
    
private:
    predicate P;
    map<tree_t,vector<interface_t>> interfaces;

    vector<map<filter_t,vector<tree_interface_pair>>> to_insert;
    vector<filter_t::pos_t> index;

//TODO compute the boostrap update
    filter_t::pos_t compute_index (const filter_t & x){
        filter_t::pos_t hw = x.popcount()-1;
        return (index[hw-1] + x.hash(P.roots[hw].size));
    }    
              
public:
    //nf is the number of expected filters
    router(unsigned int nf): P(nf) {
        unsigned int size =0;
        filter_t::pos_t pos = 0;

        for (unsigned int i=0; i< filter_t::WIDTH; ++i){
            size+=P.roots[i].size;
        
            index.push_back(pos);
             if(P.roots[i].size<= 1)
                pos++;
            else
                pos+=P.roots[i].size;

        }
        size+=filter_t::WIDTH;
        for(unsigned int i=0; i< size; ++i){
            map<filter_t,vector<tree_interface_pair>> to_add_map;
            to_insert.push_back(to_add_map);
        }
    };
    ~router() {}


    /** adds a new filter to predicate P without checking the existence of a subset
    the filter x. It does not remove any superset of x from P. **/
    void add_filter_without_check (const filter_t & x, tree_t t, interface_t i);
    void add_filter_pre_process (const filter_t & x, tree_t t, interface_t i);
    void insertion ();
    unsigned int get_unique_filters();

    void computes_bootstrap_update(vector<map<filter_t,vector<tree_interface_pair>>> & output, tree_t t, interface_t i);
   
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

    void match(const filter_t & x, tree_t t, interface_t i);


};
#endif
