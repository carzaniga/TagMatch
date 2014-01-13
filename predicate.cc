#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <math.h>
#include <thread>

#include "predicate.hh"

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
		}
		external_pairs[pairs_count].tree = t;
		external_pairs[pairs_count].interface = i;
		pairs_count += 1;
	}
}

void predicate::node::remove_pair(tree_t t,interface_t i) {
	for (tree_interface_pair * ti=ti_begin(); ti!=ti_end(); ti++) {
		if (ti->equals(t,i)) {
			if (pairs_count > 1) {
				ti->tree=(ti_end()-1)->tree;
				ti->interface=(ti_end() - 1)->interface;
			}
			remove_last_pair();
			break;
		}
	}
}

void predicate::node::remove_last_pair() {
	if(pairs_count<=LOCAL_PAIRS_CAPACITY)
		pairs_count--;
	else if(pairs_count==(LOCAL_PAIRS_CAPACITY+1)){
		tree_interface_pair * t2= ti_begin();
		for(uint16_t i=0;i<LOCAL_PAIRS_CAPACITY;i++){
			local_pairs[i].tree=t2->tree;
			local_pairs[i].interface=t2->interface;
			t2++ ;
		}
		pairs_count--;
		delete [] (t2-LOCAL_PAIRS_CAPACITY);
	}
	else if (pairs_count%EXT_PAIRS_ALLOCATION_UNIT==1){
		pairs_count--;
		size_t byte_pos = (pairs_count) * sizeof(tree_interface_pair);
		external_pairs=(tree_interface_pair *)realloc(external_pairs,byte_pos);
	}
	else
		pairs_count--;
}

void predicate::destroy() {
    for(int i=0;i<192;i++){
        for(int j=0; i< roots[i].size; j++){
            node r = roots[i].tries[j];
            if (r.pos <= r.left->pos)
                return;

            node * S[filter_t::WIDTH];
            unsigned int head = 0;
            S[0] = r.left;
        
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
        }
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
	virtual bool handle_filter(const filter_t & filter, const predicate::node & n);
private:
	const tree_t tree;
	match_handler & matcher;
};

bool tree_matcher::handle_filter(const filter_t & filter, const predicate::node & n) {
	for(const tree_interface_pair * ti = n.ti_begin(); ti != n.ti_end(); ++ti)
		if (ti->tree == tree)
			if (matcher.match(filter, tree, ti->interface))
				return true;
	return false;
}

// this is the handler we use to perform the tree and interface matching.  The
// predicate subset search finds subsets of the given filter, and
// this handler does the tree matching and the interface on the corresponding
// tree_interface pairs.
// 
class tree_ifx_matcher : public filter_const_handler {
public:
	tree_ifx_matcher(tree_t t, interface_t i, match_handler & mh): tree(t), ifx(i), matcher(mh) {}
	virtual bool handle_filter(const filter_t & filter, const predicate::node & n);
private:
	const tree_t tree;
    const interface_t ifx;
	match_handler & matcher;
};

bool tree_ifx_matcher::handle_filter(const filter_t & filter, const predicate::node & n) {
	for(const tree_interface_pair * ti = n.ti_begin(); ti != n.ti_end(); ++ti)
		if (ti->tree == tree && ti->interface == ifx)
			if (matcher.match(filter, tree, ti->interface))
				return true;
	return false;
}




#define TREE_MASK 0
#define APP 0
#define TAG_SET 0

#define SUPERSET_CUT 1

#if SUPERSET_CUT
struct stack_t {
	const predicate::node * n;
	filter_t::pos_t branch;

	void assign(const predicate::node * nx, filter_t::pos_t bx) {
		n = nx;
		branch = bx;
	}
};
#endif

void predicate::find_supersets(const filter_t & x, tree_t t, match_handler & h){

    tree_matcher matcher(t,h);

#if TAG_SET
    unsigned int start = x.popcount();
    if(start%7!=0)
        start = (start/7 + 1)*7;
#else
	filter_t::pos_t start = x.popcount();
#endif
    std::vector<std::thread> ts;
    for(filter_t::pos_t hw = start; hw < filter_t::WIDTH; hw++){  
        if(roots[hw].size==1)
            ts.push_back(std::thread(&predicate::find_supersets_of, this, x, std::ref(roots[hw].tries[0]), matcher));
            //find_supersets_of(x, roots[hw].tries[0], matcher);
        else if(roots[hw].size>1){
            //std::vector<std::thread> ts; 
            for(filter_t::pos_t i=0; i<roots[hw].size; i++)
                ts.push_back(std::thread(&predicate::find_supersets_of, this, x, std::ref(roots[hw].tries[i]), matcher));
            //for(auto& t : ts)
    	    //    t.join();
        }
    }
    for(auto& t : ts)
        t.join();
}


void predicate::find_supersets_on_ifx(const filter_t & x, tree_t t, interface_t i, match_handler & h) const {
    tree_ifx_matcher matcher(t,i,h);
    
#if TAG_SET
    filter_t::pos_t start = x.popcount();
    if(start%7!=0)
        start = (start/7 + 1)*7;
#else
	filter_t::pos_t start = x.popcount();
#endif
    std::vector<std::thread> ts;
    for(filter_t::pos_t hw = start; hw < filter_t::WIDTH; hw++){ 
        if(roots[hw].size==1)
            ts.push_back(std::thread(&predicate::find_supersets_of, this, x, std::ref(roots[hw].tries[0]), matcher));
            //find_supersets_of(x, roots[hw].tries[0], matcher);
        else if(roots[hw].size>1){
            //std::vector<std::thread> ts; 
            for(filter_t::pos_t i=0; i<roots[hw].size; i++)
                ts.push_back(std::thread(&predicate::find_supersets_of, this, x, std::ref(roots[hw].tries[i]), matcher));
            //for(auto& t : ts)
    	        //t.join();
        }
    }
    for(auto& t : ts)
        t.join();
}

void predicate::exists_subset(const filter_t & x, tree_t t, interface_t i, match_handler & h) const {
    tree_ifx_matcher matcher(t,i,h);

#if TAG_SET
    filter_t::pos_t stop = x.popcount();
    if(stop%7==0)
        stop-=8;
    else
        stop=(stop/7)*7-1;
#else
    filter_t::pos_t stop = x.popcount()-2;
#endif
    std::vector<std::thread> ts;
    for(filter_t::pos_t hw=0; hw<=stop; hw++){
        if(roots[hw].size==1)
             ts.push_back(std::thread(&predicate::find_subsets_of, this, x, std::ref(roots[hw].tries[0]), matcher));
            //find_subsets_of(x, roots[hw].tries[0], matcher);
        else if(roots[hw].size>1){
            //std::vector<std::thread> ts; 
            for(filter_t::pos_t i=0; i<roots[hw].size; i++)
                ts.push_back(std::thread(&predicate::find_subsets_of, this, x, std::ref(roots[hw].tries[i]), matcher));
            //for(auto& t : ts)
    	    //    t.join();
        }
    }
    for(auto& t : ts)
        t.join();
}

bool predicate::exists_filter(const filter_t & x, tree_t t, interface_t i) const {
    filter_t::pos_t hw = x.popcount()-1;
    if(roots[hw].size==0){
        std::cout<<"size 0" << std::endl;
        return false;
    }
    //this can be parallelized, for now we go sequentially 
    for(filter_t::pos_t i=0; i<roots[hw].size; i++){
        node * n =find(x,roots[hw].tries[i]);
        if(n!=0){
            std::cout<<"node found"<<std::endl;
            for(const tree_interface_pair * ti = n->ti_begin(); ti != n->ti_end(); ++ti)
                if (ti->tree == t && ti->interface == i){
                    std::cout << "match" << std::endl;
                    return true;
                }
            std::cout << "no match" << std::endl;
            return false;
        }
    }
    return false;
}

void predicate::count_subsets_by_ifx(const filter_t & x, tree_t t, match_handler & h) const {
    tree_matcher matcher(t,h);
    //first we look if the filter exist, becuase the filter is a subset of itself
    //than we look at the subsets

    //exact match
    filter_t::pos_t hw = x.popcount()-1;
    if(roots[hw].size!=0){
        for(filter_t::pos_t i=0; i<roots[hw].size; i++){
            node * n =find(x,roots[hw].tries[i]);
            if(n!=0){
                matcher.handle_filter(x,*n);
                break;
            }    
        }
    }   
    
    //subsets
#if TAG_SET
    filter_t::pos_t stop =hw+1;
    if(stop%7==0)
        stop-=8;
    else
        stop=(stop/7)*7-1;
#else
    filter_t::pos_t stop = hw-1;
#endif
    std::vector<std::thread> ts;
    for(filter_t::pos_t hw=0; hw<=stop; hw++){
        if(roots[hw].size==1)
           ts.push_back(std::thread(&predicate::find_subsets_of, this, x, std::ref(roots[hw].tries[0]), matcher)); 
            //find_subsets_of(x, roots[hw].tries[0], matcher);
        else if(roots[hw].size>1){
            //std::vector<std::thread> ts; 
            for(filter_t::pos_t i=0; i<roots[hw].size; i++)
                ts.push_back(std::thread(&predicate::find_subsets_of, this, x, std::ref(roots[hw].tries[i]), matcher));
            //for(auto& t : ts)
    	    //    t.join();
        }
    }
    for(auto& t : ts)
        t.join();
}



void predicate::match(const filter_t & x, tree_t t, match_handler & h) const {
	//
	// this is the modular matching function that uses the above match
	// handler through the modular find_subset_of function
	//
   std::cout << "match" << std::endl;
    tree_matcher matcher(t,h);

//exaxt match
    filter_t::pos_t hw = x.popcount()-1;
    if(roots[hw].size!=0){
        for(filter_t::pos_t i=0; i<roots[hw].size; i++){
            node * n =find(x,roots[hw].tries[i]);
            if(n!=0){
                matcher.handle_filter(x,*n);
                break;
            }    
        }
    }   
    
    //subsets
#if TAG_SET
    filter_t::pos_t stop =hw+1;
    if(stop%7==0)
        stop-=8;
    else
        stop=(stop/7)*7-1;
#else
    filter_t::pos_t stop = hw-1;
#endif
    std::vector<std::thread> ts;
    for(filter_t::pos_t hw=0; hw<=stop; hw++){
        if(roots[hw].size==1)
             ts.push_back(std::thread(&predicate::find_subsets_of, this, x, std::ref(roots[hw].tries[0]), matcher));
            //find_subsets_of(x, roots[hw].tries[0], matcher);
        else if(roots[hw].size>1){
            //std::vector<std::thread> ts; 
            for(filter_t::pos_t i=0; i<roots[hw].size; i++)
                ts.push_back(std::thread(&predicate::find_subsets_of, this, x, std::ref(roots[hw].tries[i]), matcher));
            //for(auto& t : ts)
    	    //    t.join();
        }
    }
    for(auto& t : ts)
        t.join();

}

predicate::node * predicate::add(const filter_t & x, tree_t t, interface_t i) {
    filter_t::pos_t hw = x.popcount()-1;
    //if there are multiple tries we add in a round-robin way to keep the load balanced
    if(roots[hw].size!=0){
        roots[hw].last_add = ( roots[hw].last_add + 1) %  roots[hw].size;
        node * n = add(x,roots[hw].tries[roots[hw].last_add]);
        n->add_pair(t, i);
        return n;
    }
    return 0;
}

#if 0
void predicate::set_mask(const filter_t & x, tree_t t) {
    node root = roots[x.popcount()-1];
	node * prev = &root;
	node * curr = root.left;

	while(prev->pos > curr->pos) {
		prev->add_tree_to_mask(t);
        if(prev->left!=&root && prev->left->pos < prev->pos)
            prev->init_tree_mask(prev->left);
        if(prev->right!=&root && prev->right->pos < prev->pos)
            prev->init_tree_mask(prev->right);
        prev = curr;
        curr = x[curr->pos] ? curr->right : curr->left;
	}
    curr->add_tree_to_mask(t);
    if(curr->left!=&root && curr->left->pos < curr->pos)
        curr->init_tree_mask(curr->left);
    if(curr->right!=&root && curr->right->pos < curr->pos)
        curr->init_tree_mask(curr->right);
}
#endif


predicate::node * predicate::add(const filter_t & x, node & root) {
  	node * prev = &(root);
	node * curr = root.left;
    while(prev->pos > curr->pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}
	if (x == curr->key)
		return curr;
	filter_t::pos_t pos = filter_t::most_significant_diff_pos(curr->key, x);

	prev = &(root);
	curr = root.left;
	
	while(prev->pos > curr->pos && curr->pos > pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}

	// now we insert the new node between prev and curr
	++filter_count;
	if (prev->pos < filter_t::NULL_POSITION && x[prev->pos]) {
		return prev->right = new node(pos, x, curr);
	} else {
		return prev->left = new node(pos, x, curr);
	}     
}

predicate::node * predicate::find(const filter_t & x, node & root) const {
    const node * prev = &root;
	node * curr = root.left;

   	while(prev->pos > curr->pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}
	return (x == curr->key) ? curr : 0;
}

#if 0
void predicate::find_subsets_of(const filter_t & x, tree_t t, filter_const_handler & h) const {
	//
	//  for datailt see the next implementaiotn of find_subset_of()
    //
#if TAG_SET
    unsigned int stop = x.popcount();
    unsigned int popcount =stop;
    if(stop%7==0)
        stop-=8;
    else
        stop=(stop/7)*7-1;
#else
    unsigned int popcount = x.popcount();
    unsigned int stop = popcount-2;
#endif
    for(unsigned int i=0; i<=stop; i++){
        node root = roots[i];
#if SUPERSET_CUT
        stack_t S[filter_t::WIDTH];
#else
        const node * S[filter_t::WIDTH];
#endif
        unsigned int head = 0;
 
        if (root.pos > root.left->pos)
#if SUPERSET_CUT
           	S[head++].assign(root.left, ((popcount - 1)-i)); 
#else
            S[head++] = root.left;
#endif

        while(head != 0) {
            assert(head <= filter_t::WIDTH);
#if SUPERSET_CUT
            --head;
            const node * n = S[head].n;
            filter_t::pos_t branch = S[head].branch;
            
#else
            const node * n = S[--head];		// for each visited node n...
#endif            
            //if (n->key.suffix_subset_of(x, n->pos)) 
            if(n->key.subset_of(x))
                if (h.handle_filter(n->key, *n))
                    return;

            if (n->left->pos == n->pos - 1 
                || (n->pos > n->left->pos
                && n->left->key.prefix_subset_of(x, n->pos, n->left->pos + 1)
                && n->left->match_tree(t))){ 
#if SUPERSET_CUT
                    if (x[n->pos]) {
					    if (branch > 0)
						    S[head++].assign(n->left, branch - 1);
				    } else {
					    S[head++].assign(n->left, branch);
				    }
#else
                    S[head++] = n->left;
#endif
            }

     
            if (x[n->pos]) {
                if (n->right->pos == n->pos - 1
                    || (n->pos > n->right->pos 
                    && n->right->key.prefix_subset_of(x, n->pos, n->right->pos + 1)
                    &&  n->right->match_tree(t)))
#if SUPERSET_CUT
                        S[head++].assign(n->right, branch);
#else
                        S[head++] = n->right;
#endif
            }

        }
    }
}
#endif


void predicate::find_subsets_of(const filter_t & x, node & root, filter_const_handler & h) const {
	//
	// this is a non-recursive (i.e., iterative) exploration of the
	// PATRICIA trie that looks for subsets.  The pattern is almost
	// exactly the same for supersets (see below).  We use a stack S
	// to keep track of the visited nodes, and we visit new nodes
	// along a subset (resp. superset) prefix.
	// 
#if SUPERSET_CUT
        stack_t S[filter_t::WIDTH];
#else
        const node * S[filter_t::WIDTH];
#endif        
        unsigned int head = 0;
        
        // if the trie is not empty we push the root node onto the stack.
        // The true root is root.left, not root, which is a sentinel node.
        //
        if (root.pos > root.left->pos)
#if SUPERSET_CUT
           	S[head++].assign(root.left, ((x.popcount())-root.left->key.popcount())); 
#else
            S[head++] = root.left;
#endif


        //
        // INVARIANT: root.left->pos is the position of the leftmost 1-bit
        // in root.left.  Therefore root.left is always a subset of the
        // input filter x up to position n->pos + 1 (i.e., excluding
        // position n->pos itself)
        // 
        while(head != 0) {
            assert(head <= filter_t::WIDTH);
#if SUPERSET_CUT
            --head;
            const node * n = S[head].n;
            filter_t::pos_t branch = S[head].branch;
            
#else
            const node * n = S[--head];		// for each visited node n...
#endif 
            //
            // INVARIANT: n is a subset of x up to position n->pos + 1
            // (i.e., excluding position n->pos itself)
            // 
            //if (n->key.suffix_subset_of(x, n->pos)) 
            if(n->key.subset_of(x))
                if (h.handle_filter(n->key, *n))
                    return;          
            // push n->left on the stack only when the bits of
            // n->left->key in positions between n->pos and
            // n->left->pos, excluding n->left->pos, are a subset of x
            // 
            if (n->left->pos == n->pos - 1 
                || (n->pos > n->left->pos
                && n->left->key.prefix_subset_of(x, n->pos, n->left->pos + 1))){ 
#if SUPERSET_CUT
                    if (x[n->pos]) {
					    if (branch > 0)
						    S[head++].assign(n->left, branch - 1);
				    } else {
					    S[head++].assign(n->left, branch);
				    }

#else
                    S[head++] = n->left;
#endif
            }    
            // push n->right on the stack only when x has a 1 in n->pos,
            // and then when the bits of n->right->key in positions
            // between n->pos and n->right->pos, excluding n->right->pos,
            // are a subset of x
            // 
            if (x[n->pos]) {
                if (n->right->pos == n->pos - 1
                    || (n->pos > n->right->pos 
                    && n->right->key.prefix_subset_of(x, n->pos, n->right->pos + 1)))
#if SUPERSET_CUT
                        S[head++].assign(n->right, branch);
#else
                        S[head++] = n->right;
#endif

            }
        }
}

#if 0
void predicate::find_subsets_of(const filter_t & x, filter_handler & h) {

	//
	// See the above "const" find_subsets_of for technical details.
	//
#if TAG_SET
    unsigned int stop = x.popcount();
    if(stop%7==0)
        stop-=8;
    else
        stop=(stop/7)*7-1;
#else
    unsigned int stop = x.popcount()-2;
#endif
    for(unsigned int i=0; i<=stop; i++){ 
        node root = roots[i];
        node * S[filter_t::WIDTH];
        unsigned int head = 0;

        if (root.pos > root.left->pos)
            S[head++] = root.left;

        while(head != 0) {
            assert(head <= filter_t::WIDTH);
            node * n = S[--head];

            if (n->key.suffix_subset_of(x, n->pos)) 
                if (h.handle_filter(n->key, *n))
                    return;

            if (n->left->pos == n->pos - 1 
                || (n->pos > n->left->pos
                    && n->left->key.prefix_subset_of(x, n->pos, n->left->pos + 1))) 
                    S[head++] = n->left;

            if (x[n->pos]) {
                if (n->right->pos == n->pos - 1
                    || (n->pos > n->right->pos 
                        && n->right->key.prefix_subset_of(x, n->pos, n->right->pos + 1)))
                    S[head++] = n->right;
            }
        }
    }
}
#endif

void predicate::find_supersets_of(const filter_t & x, node & root, filter_const_handler & h) const {
	//
	// See also the above find_subsets_of for technical details.
	//

    
#if SUPERSET_CUT
        stack_t S[filter_t::WIDTH];
#else
        const node * S[filter_t::WIDTH];
#endif
        unsigned int head = 0;

        // if the trie is not empty we push the root node onto the stack.
        // The true root is root.left, not root, which is a sentinel node.
        //
        if (root.pos > root.left->pos
            //
            // INVARIANT: root.left->pos is the position of the leftmost
            // 1-bit in root.left.  Therefore root.left should be
            // considered only if x's most significant bit it to the right
            // (i.e., lower position) of root.left->pos.
            // 
            && x.most_significant_one_pos() <= root.left->pos){
#if SUPERSET_CUT
			S[head++].assign(root.left, (root.left->key.popcount() - (x.popcount())));
#else
            S[head++] = root.left;
#endif
        }
        while(head != 0) {
            assert(head <= filter_t::WIDTH);
#if SUPERSET_CUT
            --head;
            const node * n = S[head].n;
            filter_t::pos_t branch = S[head].branch;
#else
            const node * n = S[--head];		// for each visited node n...
#endif
            //
            // INVARIANT: n is a superset of x up to position n->pos + 1
            // (i.e., excluding position n->pos itself)
            // 
            if (x.suffix_subset_of(n->key, n->pos)) 
                if (h.handle_filter(n->key, *n))
                    return;

            // push n->right on the stack only when the bits of
            // n->right->key in positions between n->pos and
            // n->right->pos, excluding n->right->pos, are a subset of x
            // 
            if (n->right->pos == n->pos - 1 
                || (n->pos > n->right->pos
                    && x.prefix_subset_of(n->right->key, n->pos, n->right->pos + 1))) { 
#if SUPERSET_CUT
				if (!x[n->pos]) {
					if (branch > 0)
						S[head++].assign(n->right, branch - 1);
				} else {
					S[head++].assign(n->right, branch);
				}
#else
				S[head++] = n->right;
#endif
            }
            // push n->left on the stack only when x has a 0 in n->pos,
            // and then when the bits of n->right->key in positions
            // between n->pos and n->left->pos, excluding n->left->pos,
            // are a superset of x
            // 
            if (!x[n->pos]) {
                if (n->left->pos == n->pos - 1
                    || (n->pos > n->left->pos 
                        && x.prefix_subset_of(n->left->key, n->pos, n->left->pos + 1))){
#if SUPERSET_CUT
                    S[head++].assign(n->left, branch);
#else
                    S[head++] = n->left;
#endif
                }
            }
        }
    
}

#if 0
void predicate::find_supersets_of(const filter_t & x, filter_handler & h) {
	//
	// See the above "const" find_superset_of for technical details.
	//
#if TAG_SET
    unsigned int start = x.popcount();
    if(start%7!=0)
        start=(start/7 +1)*7;
#else
    unsigned int start = x.popcount();
#endif 
    for(unsigned int i=start; i<192; i++){
        node root = roots[i];        
        node * S[filter_t::WIDTH];
        unsigned int head = 0;

        if (root.pos > root.left->pos
            && x.most_significant_one_pos() <= root.left->pos)
            S[head++] = root.left;

        while(head != 0) {
            assert(head <= filter_t::WIDTH);
            node * n = S[--head];

            if (x.suffix_subset_of(n->key, n->pos)) 
                if (h.handle_filter(n->key, *n))
                    return;

            if (n->right->pos == n->pos - 1 
                || (n->pos > n->right->pos
                    && x.prefix_subset_of(n->right->key, n->pos, n->right->pos + 1))) 
                    S[head++] = n->right;

            if (!x[n->pos]) {
                if (n->left->pos == n->pos - 1
                    || (n->pos > n->left->pos 
                        && x.prefix_subset_of(n->left->key, n->pos, n->left->pos + 1)))
                    S[head++] = n->left;
            }
        }
    }
}
#endif

void predicate::remove(const filter_t & x , tree_t t, interface_t i){
    filter_t::pos_t hw = x.popcount()-1;
    if(roots[hw].size==0)
        return;
    //this can be parallelized, for now we go sequentially 
    for(filter_t::pos_t i=0; i<roots[hw].size; i++){
        node * n =find(x,roots[hw].tries[i]);
        if(n!=0)
            n->remove_pair(t,i);
            return;
    }
}

