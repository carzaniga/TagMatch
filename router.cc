#include <iostream>
#include "router.hh"

using namespace std;

#define TEST 1

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
    std::map<interface_t,set<filter_t>>::iterator it;
    it=supersets.find(ifx);
    if(it==supersets.end()){
        set<filter_t> s;
        s.insert(filter);
        supersets.insert(pair<interface_t,set<filter_t>>(ifx,s));
    }else{
        it->second.insert(filter);
    }
	return false;
}

/** returns always false because we want to visit the whole trie. counts the number
of subsets on each interface **/
bool matcher_count_subsets_by_ifx::match(const filter_t & filter, tree_t tree, interface_t ifx) {
    if(ifx==i)
        return false;
    std::map<interface_t,unsigned int>::iterator it;
    it=subsets.find(ifx);
    if(it==subsets.end()){
        subsets.insert(pair<interface_t,unsigned int>(ifx,1));
    }else{
        it->second++;    
    }
	return false;
}


/** adds a new filter in predicate without any check. the initialization of the map
interfaces should be done be the routing protocol itself! **/
void router::add_filter_without_check (const filter_t & x, tree_t t, interface_t i){  
    map<tree_t,set<interface_t>>::iterator it = interfaces.find(t);
    if(it==interfaces.end()){
        set<interface_t> s;
        s.insert(i);
        interfaces.insert(pair<tree_t,set<interface_t>>(t,s));
    }else{
        it->second.insert(i);
    }
    P.add(x,t,i);
}

/** removes a filter from predicate without any check **/
void router::remove_filter_without_check (const filter_t & x, tree_t t, interface_t i){
    P.remove(x,t,i);
}

/** add a new filter if needed, removing all supersets. returns true if we need to 
broadcast the update to all the interfaces **/
bool router::add_filter (const filter_t & x, tree_t t, interface_t i){
    matcher_exists m;
    P.exists_subset(x,t,i,m);
    if(m.get_match()){
#if TEST
        cout<<"the filter (or a subsets) already exists" << endl;
#endif
        return 0;
    }else{
#if TEST
        cout<<"add filter"<<endl;
#endif
        matcher_collect_supersets m_super;
        P.find_supersets_on_ifx(x,t,i,m_super);
        map<interface_t,set<filter_t>> * sup = m_super.get_supersets();
        map<interface_t,set<filter_t>>::iterator it_sup = sup->find(i);
        if(it_sup!=sup->end()){
            for (set<filter_t>::iterator it=it_sup->second.begin(); it!=it_sup->second.end(); ++it){
#if TEST
                cout<<"remove a superset"<<endl;
#endif
                P.remove(*it,t,i);
            }
        }
        P.add(x,t,i);
        return 1;
    }
}


void router::remove_filter (set<predicate_delta> & output, const filter_t & x, tree_t t, interface_t i){
    if(P.exists_filter(x,t,i)){
#if TEST
        cout <<"remove filter" << endl;
#endif
        //remove the filter from intreface i on tree t
        P.remove(x,t,i);
        //for each interface we need to compute a new delta with - and +
        //
        //give a certain interface j we sent the filter x (that we want to remove) 
        //if the union of all the filter on the other interfaces (j and i excluded)
        //does not conteins a subset of the filter x
        //
        //if we have to propagate filter x we have to check if we need to send additional
        //updates becuase we may uncover some filters. we send a + on interface j if 
        //there are supersets of x on interfaces different from j. the + is the sum
        //(minimal) of the supersets that we found.
        //
        //we can make the sum minimal in the apply_delta function since we need to remove
        //all the filters covered by the + in the delta
    
        matcher_count_subsets_by_ifx m_subs;
        P.count_subsets_by_ifx(x,t,m_subs);
        //find all the supersets...(always needed??) 
        matcher_collect_supersets m_super;
        P.find_supersets(x,t,m_super);
        map<tree_t,set<interface_t>>::iterator map_it = interfaces.find(t);
        for (set<interface_t>::iterator it=map_it->second.begin(); it!=map_it->second.end(); ++it){
            if(*it != i){
                if(!m_subs.exists_subsets_on_other_ifx(*it)){
                    predicate_delta pd(t,*it);
                    pd.create_minimal_delta(x,*(m_super.get_supersets()),*it);
                    output.insert(pd);
                }else{
                }
            }
        }
  
    }else{
#if TEST
    cout <<"filter does not exist" << endl;
#endif

        return;
    }
}

void router::apply_delta(set<predicate_delta> & output, const predicate_delta & d) {
    // produces a set of predicate deltas as a result of applying d to p

    //this map stores temporally the set pf predicate_delta that we need to output
    map<interface_t,predicate_delta> tmp;
    //map_it is the pointer to the set of interfaces use on tree d.tree.
    map<tree_t,set<interface_t>>::iterator map_it = interfaces.find(d.tree);

    //add_filters
    for(set<filter_t>::iterator add_it=d.additions.begin(); add_it!=d.additions.end(); add_it++){
        bool add = add_filter(*add_it,d.tree,d.ifx);
        if(add){
            //if add we need to broadcast the filter to all the interfaces except to interface 
            //d.ifx
            for(set<interface_t>::iterator if_it = map_it->second.begin(); if_it!=map_it->second.end();if_it++){
                if(*if_it!=d.ifx){
                    map<interface_t,predicate_delta>::iterator it = tmp.find(*if_it);
                    if(it==tmp.end()){
                        predicate_delta pd(*if_it,d.tree);
                        pd.add_additional_filter(*add_it);
                        tmp.insert(pair<interface_t,predicate_delta>(*if_it,pd));
                    }else{
                        it->second.add_additional_filter(*add_it);
                    }
                }
            }
        }
    }
    
    //this si a temporary set used to store the outputs of remove_fitler
    set<predicate_delta> out;
    //remove_fitler
    for(set<filter_t>::iterator rm_it=d.removals.begin(); rm_it!=d.removals.end(); rm_it++){
        out.clear();
        remove_filter(out,*rm_it,d.tree,d.ifx);
        for(set<predicate_delta>::iterator out_it=out.begin(); out_it!=out.end(); out_it++){
            //for each predicate_delta in out we need to store it in tmp (each predicate_delta
            //has to be minimal)
            map<interface_t,predicate_delta>::iterator it = tmp.find(out_it->ifx);
            if(it==tmp.end()){
                tmp.insert(pair<interface_t,predicate_delta>(out_it->ifx,*out_it));
            }else{
                it->second.merge_deltas(*out_it);
            }
        }
    }
    
    //store the results in output
    for(map<interface_t,predicate_delta>::iterator it_tmp=tmp.begin(); it_tmp!=tmp.end(); it_tmp++)
        if(it_tmp->second.additions.size()!=0 || it_tmp->second.removals.size()!=0)
            output.insert(it_tmp->second);
}
  
