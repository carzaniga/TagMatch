#include <iostream>
#include "router.hh"

using namespace std;

/** match always returns true becuase is called only when we alredy matched the filter,
the tree and the interface. Sets also exists_match to true in order to check the result
later **/

bool matcher_exists::match(const filter_t & filter, tree_t tree, interface_t ifx) {
    exists_match=true;
	return true;
}

/** match always returns false becuase we don't wont to stop the algorithm. In this way we
can collect all the supersets. Depending on the matcher used in predicate.cc we may collect
the filters on a particular interface or all the filter in the trie **/

bool matcher_collect_supersets::match(const filter_t & filter, tree_t tree, interface_t ifx) {
    if(ifx==interface)  
        to_remove.add(filter);
    else{
        while(__sync_lock_test_and_set(&lock,1));
        supersets[ifx].push_back(filter);
        __sync_lock_release(&lock);
    }
   	return false;
}

/** returns always false because we want to visit the whole trie. counts the number
of subsets on each interface **/
bool matcher_count_subsets_by_ifx::match(const filter_t & filter, tree_t tree, interface_t ifx) {
    if(ifx==i)
        return false;
    while(__sync_lock_test_and_set(&lock,1));
    subsets[ifx]++;
    __sync_lock_release(&lock);
	return false;
}

/**matcher for the normal match... */
bool matcher_get_out_interfaces::match(const filter_t & filter, tree_t tree, interface_t ifx) {
    if(ifx==i)
        return false;
    while(__sync_lock_test_and_set(&lock,1));
    //the tree should be alredy checked!
    ifxs.insert(ifx);
    __sync_lock_release(&lock);
    return false;
}

//this fucntion merges two predicate delta minimizing the list of filters 
void predicate_delta::merge(const predicate_delta & d) {
	//merge the removal list
	for(list<filter_t>::const_iterator it = d.removals.begin(); it!=d.removals.end(); it++) 
		add_removal_filter(*it);

	//merge additional list
	for(list<filter_t>::const_iterator it = d.additions.begin(); it!=d.additions.end(); it++) 
		add_additional_filter(*it);
}

// add a filter (f) to the set of removal filters.  If the removal set
// already contains f, then it simply ignores this request.  We assume
// that f is NOT among the additions in this delta.
//
void predicate_delta::add_removal_filter(const filter_t & f) {
#ifdef PARANOIA_CHECKS
	set<filter_t>::iterator it = additions.find(f);
	if (it!=additions.end()) {
		//erase the fitler from additions (see is_needed_add below
		//for explanations)
		additions.erase(it);
		return;
	}
#endif        

	for(list<filter_t>::iterator it = removals.begin(); it != removals.end();){
		if(it->subset_of(f))
			return;
		if (f.subset_of(*it)) {
			removals.erase(it++);
		} else {
			++it;
		}
	}
	removals.push_back(f);
}
    
void predicate_delta::add_additional_filter(const filter_t & f) {
#ifdef PARANOIA_CHECKS
	//check if the filter is in the removals set
	//if it is there we remove the filter from the removals 
	//set and we return false
	list<filter_t>::iterator it = removals.find(f);
	if(it != removals.end()) {
		removals.erase(it);
		return;
	}
#endif
	//here we need to remove all the possibile supersets of the filter that
	//we are introducing
	for(list<filter_t>::iterator it = additions.begin(); it != additions.end();) {
		if(it->subset_of(f))
			return;
		if (f.subset_of(*it)) {
			additions.erase(it++);
		}
		else {
			++it;
		}
	}
	additions.push_back(f);
}



void predicate_delta::create_minimal_delta(const filter_t & remove, 
										   map<interface_t,vector<filter_t>> & add, const interface_t i) {
	//we need to remove the filter remove and so we try to add it to removals
	add_removal_filter(remove);        
	map<interface_t,vector<filter_t>>::iterator it_map;
	for(it_map = add.begin(); it_map!=add.end(); it_map++) {
		//for each inetrface which is not i we add the uncovered filters, 
		//mining supersets of remove, to the additions set  
		if(it_map->first != i){
			vector<filter_t>::iterator it_set;
			for(it_set=it_map->second.begin(); it_set!=it_map->second.end(); it_set++){
				add_additional_filter(*it_set);
			}                
		}
	}
}


/** adds a new filter in predicate without any check. 
**/
void router::add_filter_without_check (const filter_t & x, tree_t t, interface_t i){  
    P.add(x,t,i);
}


/** add an interface to a tree **/
void router::add_ifx_to_tree(tree_t t, interface_t i){
     map<tree_t,vector<interface_t>>::iterator it = interfaces.find(t);
    if(it==interfaces.end()){
        vector<interface_t> s;
        s.push_back(i);
        interfaces.insert(pair<tree_t,vector<interface_t>>(t,s));
    }else{
        it->second.push_back(i);
    }
}

/** removes a filter from predicate without any check **/
void router::remove_filter_without_check (const filter_t & x, tree_t t, interface_t i){
    P.remove(x,t,i);
}

/** add a new filter if needed, removing all supersets. returns true if we need to 
broadcast the update to all the interfaces **/
bool router::add_filter (const filter_t & x, tree_t t, interface_t i, synch_filter_vector & to_add, synch_filter_vector & to_rm){
    //if the filter exists we discard the filter and return 0
    if(P.exists_filter(x,t,i))
        return 0;
    //same if exists a subset of x
    matcher_exists m;    
    P.exists_subset(x,t,i,m);
    if(m.get_match())
        return 0;
    matcher_collect_supersets m_super(to_rm,i);
    P.find_supersets_on_ifx(x,t,i,m_super);
    to_add.add(x);
    return 1;
}


void router::remove_filter (const filter_t & x, tree_t t, interface_t i, synch_ifx_delta_map & out_delta, synch_filter_vector & to_rm){
    if(P.exists_filter(x,t,i)) {
        //remove the filter from intreface i on tree t
        to_rm.add(x); 
        //for each interface we need to compute a new delta with - and +
        //
        //given an interface j we send the filter x (that we want to remove) 
        //if the union of all the filter on the other interfaces (j and i excluded)
        //does not conteins a subset of the filter x
        //
        //if we have to propagate filter x we have to check if we need to send additional
        //updates becuase we may uncover some filters. we send a + on interface j if 
        //there are supersets of x on interfaces different from j. the + is the sum
        //(minimal) of the supersets that we found.
        //
        matcher_count_subsets_by_ifx m_subs;

        P.count_subsets_by_ifx(x,t,m_subs);
        matcher_collect_supersets m_super(to_rm, i);
        bool superset_executed = false;
        map<tree_t,vector<interface_t>>::iterator map_it = interfaces.find(t);
        for (vector<interface_t>::iterator it=map_it->second.begin(); it!=map_it->second.end(); ++it){
            if(*it != i){
                if(!m_subs.exists_subsets_on_other_ifx(*it,i)){
                    if(!superset_executed){
                        superset_executed=true;
                        P.find_supersets(x,t,m_super);
                    }
                    predicate_delta pd;
                    pd.create_minimal_delta(x,*(m_super.get_supersets()),*it);
                    out_delta.add(*it,pd);
                }
            }
        }
    }
}

void router::start_threads(){
    for(unsigned int i = 0; i < THREAD_COUNT; ++i)
	    thread_pool[i] = new thread(&router::thread_loop, this, i);
}

void router::stop_threads(){
    for(unsigned int i = 0; i < THREAD_COUNT; ++i)
        job_enqueue(0);
    for(unsigned int i = 0; i < THREAD_COUNT; ++i)
        thread_pool[i]->join();
    for(unsigned int i = 0; i < THREAD_COUNT; ++i)
        delete(thread_pool[i]);   
}

void router::apply_delta(map<interface_t,predicate_delta> & output, 
						 const predicate_delta & d, interface_t i, tree_t t) {

    //map_it is the pointer to the set of interfaces use on tree d.tree.
	vector<interface_t> & tree_ifx = interfaces[t];
    //first part: add filters
    vector<thread> ts;
    synch_filter_vector to_add;
    synch_filter_vector to_rm;
    
    synch_ifx_delta_map out_delta(output);

   for(list<filter_t>::const_iterator it=d.additions.begin(); it!=d.additions.end(); it++){
        job_enqueue(new r_params(true, *it, t, i, 0, &to_add, &to_rm));
        inc_synch_queue();
    }
    
    for(list<filter_t>::const_iterator rm_it=d.removals.begin(); rm_it!=d.removals.end(); rm_it++) {
        job_enqueue(new r_params(false,*rm_it,t,i,&out_delta,0,&to_rm));
        inc_synch_queue();
    }


    //wait until the end of all the jobs
    //the wait condition goes in a while loop
    //spurious wakeups may occur. 
    std::unique_lock<std::mutex> lock(synch_queue_lock);
    while(synch_queue!=0){
        synch_queue_done.wait(lock);
    }
        
    for(vector<filter_t>::iterator add_it=to_add.filters.begin(); add_it!=to_add.filters.end(); ++add_it){
        P.add(*add_it,t,i);
        for(vector<interface_t>::iterator if_it = tree_ifx.begin(); if_it!=tree_ifx.end(); if_it++){
            if(*if_it != i) {
				output[*if_it].add_additional_filter(*add_it);
            }
        }
    }
    
    for(vector<filter_t>::iterator rm_it=to_rm.filters.begin(); rm_it!=to_rm.filters.end(); ++rm_it){
        P.remove(*rm_it,t,i);
    }
    
}
  