#ifndef update_hh_included
#define update_hh_included

#include <set>
#include <map>
#include <mutex>
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
* (see predicate.cc) in order to get all the subset of one 
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
    
    map<interface_t,unsigned int>  * get_subsets() {
        return &subsets;
    }

    bool exists_subsets_on_other_ifx(interface_t ifx){
        for (map<interface_t,unsigned int>::iterator it=subsets.begin(); it!=subsets.end(); ++it){
            if(it->second!=0 && ifx!=it->first)
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
    
    set<interface_t>  * get_interfaces() {
        return &interfaces;
    }

	virtual bool match(const filter_t & filter, tree_t tree, interface_t ifx);
private: 
    //input interface from where we received the update
    interface_t i;
    set<interface_t> interfaces;
    mutex mtx;
};





class predicate_delta {
public:
    interface_t ifx;
    tree_t tree;

    set<filter_t> additions;
    set<filter_t> removals;

    predicate_delta(interface_t i, tree_t t): ifx(i), tree(t) {};

    //load the sets additions and removals with minimal sets. interface i is the one that we want to 
    //skip reading the content of the map. this procedure is not really efficient 
    void create_minimal_delta(const filter_t & remove, map<interface_t,vector<filter_t>> & add, const interface_t i ){
        //we need to remove the filter remove and so we try to add it to removals
        add_removal_filter(remove);        
        map<interface_t,vector<filter_t>>::iterator it_map;
        for(it_map = add.begin(); it_map!=add.end(); it_map++){
            //for each inetrface which is not i we add the uncovered filters, 
            //mining supersets of remove, to the additions set  
            if(it_map->first!=i){
                vector<filter_t>::iterator it_set;
                for(it_set=it_map->second.begin(); it_set!=it_map->second.end(); it_set++){
                    add_additional_filter(*it_set);
                }                
            }
        }
    }
    
    //this fucntion merges two predicate delta minimizing the list of filters 
    void merge_deltas(const predicate_delta & d){
        //merge the removal list
        for(set<filter_t>::iterator it = d.removals.begin(); it!=d.removals.end(); it++){
            add_removal_filter(*it);
        }
        //merge additional list
        for(set<filter_t>::iterator it = d.additions.begin(); it!=d.additions.end(); it++){
            add_additional_filter(*it);
        }
    }
    
    void add_additional_filter(const filter_t t){
        if(is_needed_add(t)){
            additions.insert(t);
        }
    }
    
    void add_removal_filter(const filter_t t){
        if(is_needed_rm(t)){
            removals.insert(t);
        }
    }


private:
    //return false if the filter f or one of is subset is in removal.
    //it also remove all the superstes in case we add the filter
    //this can be improved using the sorting by hamming weight (TO DO)
    bool is_needed_rm(filter_t f){
        set<filter_t>::iterator it = removals.find(f);
        if(it!=removals.end())
            return false;
        it = additions.find(f);
        if(it!=additions.end()){
            //erase the fitler from additions (see is_needed_add below
            //for explanations)
            additions.erase(it);
            return false;
        }
        
        //???????
        //do we really need to make this set minimal?
        //remove A and AB produce a different effect than remove
        //only A!
        //??????
        
        //????????
        //is there any relation bitween removals and additions that we need to check?
        //can we have an addition filter that is a subset of a removal one?
        //can we have an addition filter that is a super set of a removal one?
        //????????


        //for(it=removals.begin(); it!=removals.end(); it++){
        //    if(it->subset_of(f))
        //        return false;
        //}
        //if we have to insert the filter we want to make sure that 
        //there is no super set of the filter in the set
        for(it=removals.begin(); it!=removals.end();){
            if(it->subset_of(f))
                return false;
            if (f.subset_of(*it)) {
                removals.erase(it++);
            }
            else {
                ++it;
            }
        }
        return true;
    }
    
    //this function is similar to the previos one but is used fro the addition
    //filters. As in the previuos case we can take advange from the sorting in order
    //to cut some checks (TO DO)
    bool is_needed_add(filter_t f){
        //check if the filter is in the removals set
        //if it is there we remove the filter from the removals 
        //set and we return false
        set<filter_t>::iterator it = removals.find(f);
        if(it!=removals.end()){
            removals.erase(it);
            return false;
        }
        //check if the filter is in the additions set
        it = additions.find(f);
        if(it!=additions.end())
            return false;
        //check if the filter is a superset of something in the additions set
        //for(it=additions.begin(); it!=additions.end(); it++){
        //    if(it->subset_of(f))
        //        return false;
        //}
        //here we need to remove all the possibile supersets of the filter that
        //we are introducing
        for(it=additions.begin(); it!=additions.end();){
            if(it->subset_of(f))
                return false;
            if (f.subset_of(*it)) {
                additions.erase(it++);
            }
            else {
                ++it;
            }
        }
        return true;
    } 
};

class router {
    
private:
    predicate P;

    map<tree_t,vector<interface_t>> interfaces;
              
public:
    //nf is the number of expected filters
    router(unsigned int nf): P(nf) {};
    ~router() {}


    /** adds a new filter to predicate P without checking the existence of a subset
    the filter x. It does not remove any superset of x from P. **/
    void add_filter_without_check (const filter_t & x, tree_t t, interface_t i);
   
    /** removes a filter from P without any check **/
    void remove_filter_without_check (const filter_t & x, tree_t t, interface_t i);
 
    /** adds a new filter to predicate P. returns true if the filter has to be sent to 
    all the neighbors, fasle otherwise **/
    bool add_filter (const filter_t & x, tree_t t, interface_t i);
    
    /** removes a filter from predicate P. **/
    void remove_filter (vector<predicate_delta> & output, const filter_t & x, tree_t t, interface_t i);

    /**  produces a set of predicate deltas as a result of applying d to P **/
    void apply_delta(vector<predicate_delta> & output, const predicate_delta & d);

    void add_ifx_to_tree(tree_t t, interface_t i);

    void match(const filter_t & x, tree_t t, interface_t i);


};
#endif
