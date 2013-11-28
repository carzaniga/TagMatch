#include <iostream>
#include "router.hh"

using namespace std;

#define TEST 1

/** match always returns true becuase is called only when we alredy matched the filter,
the tree and the interface. **/ 
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


void router::add_filter_without_check (const filter_t & x, tree_t t, interface_t i){  
    interfaces.insert(i);
    P.add(x,t,i);
}

void router::remove_filter_without_check (const filter_t & x, tree_t t, interface_t i){
    P.remove(x,t,i);
}

//TO DO: better to return void and pass an object like in the case of remove_filter
//the best way to do this may be to have a map (or a vector) of delta_predicate.
//basically what we do do here (and in remove_filter) is to compute new filter to add
//to the delta and so we can simpli add them to an existing one
//schetck of the code:
//if the map does not contain the interface 
//add a new entry with a new predicate_delta object and add the filter to the
//additional filter
//else if it exist add the filter taking the additions/removals minimal

bool router::add_filter (const filter_t & x, tree_t t, interface_t i){
    matcher_exists m;
    P.exists_subset(x,t,i,m);
    if(m.get_match()){
#if TEST
        cout << "discard\n"; 
#endif        
        return 0;
    }else{
#if TEST
        std::cout << "add\n";
#endif        
        matcher_collect_supersets m_super;
        P.find_supersets_on_ifx(x,t,i,m_super);
        map<interface_t,set<filter_t>> * sup = m_super.get_supersets();
        map<interface_t,set<filter_t>>::iterator it_sup = sup->find(i);
        if(it_sup!=sup->end()){
            for (set<filter_t>::iterator it=it_sup->second.begin(); it!=it_sup->second.end(); ++it){
#if TEST
                cout << "remove\n"; 
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
        cout << "exists\n";
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
        for (set<interface_t>::iterator it=interfaces.begin(); it!=interfaces.end(); ++it){
            if(*it != i){
#if TEST
                cout << "checking interface " << *it << endl;
#endif
                if(!m_subs.exists_subsets_on_other_ifx(*it)){
#if TEST
                    cout << "send update - \n";
                    cout << "compute update + \n";
#endif
                    predicate_delta pd(t,*it);
                    pd.create_minimal_delta(x,*(m_super.get_supersets()),*it);
                    output.insert(pd);
                }else{
#if TEST
                    cout << "send nothing\n"; 
#endif
                }
            }
        }
  
    }else{
#if TEST
       cout << "don't exists\n";
#endif
        return;
    }
}

void router::apply_delta(set<predicate_delta> & output, predicate & p, const predicate_delta & d) {
    // produces a set of predicate deltas as a result of applying d to p

}
  
