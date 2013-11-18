#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cstring>

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

void predicate::destroy() {
	if (root.pos <= root.left->pos)
		return;

	node * S[filter_t::WIDTH];
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

#define TREE_MASK 1
#define APP 1
void predicate::match(const filter_t & x, tree_t t, match_handler & h) const {
	//
	// this is the modular matching function that uses the above match
	// handler through the modular find_subset_of function
	//
#if TREE_MASK
    tree_matcher matcher(t,h);
    find_subsets_of(x, t, matcher);
#else
	tree_matcher matcher(t,h);
	find_subsets_of(x, matcher);
#endif
}

predicate::node * predicate::add(const filter_t & x, tree_t t, interface_t i) {
   	node * n = add(x);
#if TREE_MASK
    set_mask(x,t);       
#endif
	n->add_pair(t, i);
	return n;
}

void predicate::set_mask(const filter_t & x, tree_t t) {
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



predicate::node * predicate::add(const filter_t & x) {
	node * prev = &root;
	node * curr = root.left;

	while(prev->pos > curr->pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}
	if (x == curr->key)
		return curr;

	filter_t::pos_t pos = filter_t::most_significant_diff_pos(curr->key, x);

	prev = &root;
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

predicate::node * predicate::find(const filter_t & x) const {
	const node * prev = &root;
	node * curr = root.left;

	while(prev->pos > curr->pos) {
		prev = curr;
		curr = x[curr->pos] ? curr->right : curr->left;
	}
	return (x == curr->key) ? curr : 0;
}


void predicate::find_subsets_of(const filter_t & x, tree_t t, filter_const_handler & h) const {
	//
	//  for datailt see the next implementaiotn of find_subset_of()
    // 
	const node * S[filter_t::WIDTH];
	unsigned int head = 0;

	unsigned int count =0;
    
	if (root.pos > root.left->pos)
		S[head++] = root.left;

    //
    // in this implementation we also use the cut 
    // that exploits the application tags
    //
 
#if APP   
    filter_t app;
    
    if(predicate::t.yt.subset_of(x))
        app=predicate::t.yt;
    else if(predicate::t.tw.subset_of(x))
        app=predicate::t.tw;
    else if(predicate::t.blog.subset_of(x))
        app=predicate::t.blog;
    else if(predicate::t.del.subset_of(x))
        app=predicate::t.del;
    else if(predicate::t.bt.subset_of(x))
        app=predicate::t.bt;
    else
        return;
#endif

	while(head != 0) {
		assert(head <= filter_t::WIDTH);
		const node * n = S[--head];		// for each visited node n...
		
        //if (n->key.suffix_subset_of(x, n->pos)) 
        if(n->key.subset_of(x))
            if (h.handle_filter(n->key, *n))
			    return;

		if (n->pos > n->left->pos){
#if APP
           if (n->left->key.prefix_subset_of(x, n->pos, n->left->pos + 1) &&
                //app.prefix_subset_of(n->left->key, n->left->pos + 1) && 
                app.prefix_subset_of(n->left->key, n->pos, n->left->pos + 1) &&
                n->left->match_tree(t))  
				S[head++] = n->left;
#else
           if (n->left->key.prefix_subset_of(x, n->pos, n->left->pos + 1) &&
               n->left->match_tree(t))  
				S[head++] = n->left;

#endif
 
        }

		if (n->pos > n->right->pos && x[n->pos]){ 
#if APP
            if (n->right->key.prefix_subset_of(x, n->pos, n->right->pos + 1) &&
                //app.prefix_subset_of(n->right->key, n->right->pos + 1) &&
                app.prefix_subset_of(n->right->key, n->pos, n->right->pos + 1) && 
                n->right->match_tree(t)) 
				S[head++] = n->right;
#else
             if (n->right->key.prefix_subset_of(x, n->pos, n->right->pos + 1) &&
                 n->right->match_tree(t)) 
				S[head++] = n->right;

#endif
        }
	}
}



void predicate::find_subsets_of(const filter_t & x, filter_const_handler & h) const {
	//
	// this is a non-recursive (i.e., iterative) exploration of the
	// PATRICIA trie that looks for subsets.  The pattern is almost
	// exactly the same for supersets (see below).  We use a stack S
	// to keep track of the visited nodes, and we visit new nodes
	// along a subset (resp. superset) prefix.
	// 
	const node * S[filter_t::WIDTH];
	unsigned int head = 0;
	
	// if the trie is not empty we push the root node onto the stack.
	// The true root is root.left, not root, which is a sentinel node.
	//
	if (root.pos > root.left->pos)
		S[head++] = root.left;
#if APP
    // we need to chck wich application tag is conteined in the message
    // to do this we can simpli do a subset check between the message and 
    // the application tags in predicate::t. if no tag is in the
    // message we can discard it
    
    filter_t app;
    /*int count =0;
    if(t.yt.subset_of(x))
        count++;
    if(t.tw.subset_of(x))
        count++;
    if(t.blog.subset_of(x))
        count++;
    if(t.del.subset_of(x))
        count++;
    if(t.bt.subset_of(x))
        count++;

    if(count>1)
        std::cout << "msg with multiple application tags" << std::endl;
    else if(count==0)
        std::cout << "msg with no application tag" << std::endl;*/

    
    if(t.yt.subset_of(x))
        app=t.yt;
    else if(t.tw.subset_of(x))
        app=t.tw;
    else if(t.blog.subset_of(x))
        app=t.blog;
    else if(t.del.subset_of(x))
        app=t.del;
    else if(t.bt.subset_of(x))
        app=t.bt;
    else
        return;
#endif


	//
	// INVARIANT: root.left->pos is the position of the leftmost 1-bit
	// in root.left.  Therefore root.left is always a subset of the
	// input filter x up to position n->pos + 1 (i.e., excluding
	// position n->pos itself)
	// 
	while(head != 0) {
		assert(head <= filter_t::WIDTH);
		const node * n = S[--head];		// for each visited node n...
		//
		// INVARIANT: n is a subset of x up to position n->pos + 1
		// (i.e., excluding position n->pos itself)
		// 
		//if (n->key.suffix_subset_of(x, n->pos)) 
        if(n->key.subset_of(x))
			if (h.handle_filter(n->key, *n))
			    return;

		if (n->pos > n->left->pos){
#if APP
           if (n->left->key.prefix_subset_of(x, n->pos, n->left->pos + 1) &&
                //app.prefix_subset_of(n->left->key, n->left->pos + 1))  
                app.prefix_subset_of(n->left->key, n->pos, n->left->pos + 1)) 
				S[head++] = n->left;
 
            
#else
			// push n->left on the stack only when the bits of
			// n->left->key in positions between n->pos and
			// n->left->pos, excluding n->left->pos, are a subset of x
			// i
			if (n->left->key.prefix_subset_of(x, n->pos, n->left->pos + 1)) 
				S[head++] = n->left;
#endif
        }

		if (n->pos > n->right->pos && x[n->pos]){ 

#if APP
            if (n->right->key.prefix_subset_of(x, n->pos, n->right->pos + 1) &&
                //app.prefix_subset_of(n->right->key, n->right->pos + 1)) 
                app.prefix_subset_of(n->right->key, n->pos, n->right->pos + 1)) 
				S[head++] = n->right;


#else
			// push n->right on the stack only when the bits of
			// n->right->key in positions between n->pos and
			// n->right->pos, excluding n->right->pos, are a subset of
			// x
			// 
			if (n->right->key.prefix_subset_of(x, n->pos, n->right->pos + 1)) 
				S[head++] = n->right;
#endif
        }
	}
}

void predicate::find_subsets_of(const filter_t & x, filter_handler & h) {
	//
	// See the above "const" find_subsets_of for technical details.
	// 
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

		if (n->pos > n->left->pos)
			if (n->left->key.prefix_subset_of(x, n->pos, n->left->pos + 1)) 
				S[head++] = n->left;

		if (n->pos > n->right->pos && x[n->pos]) 
			if (n->right->key.prefix_subset_of(x, n->pos, n->right->pos + 1)) 
				S[head++] = n->right;
	}
}

void predicate::find_supersets_of(const filter_t & x, filter_const_handler & h) const {
	//
	// See also the above find_subsets_of for technical details.
	// 
	const node * S[filter_t::WIDTH];
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
		&& x.most_significant_one_pos() <= root.left->pos)
		S[head++] = root.left;

	while(head != 0) {
		assert(head <= filter_t::WIDTH);
		const node * n = S[--head];		// for each visited node n...
		//
		// INVARIANT: n is a superset of x up to position n->pos + 1
		// (i.e., excluding position n->pos itself)
		// 
		if (x.suffix_subset_of(n->key, n->pos)) 
			if (h.handle_filter(n->key, *n))
				return;

		if (n->pos > n->right->pos)
			// push n->right on the stack only when the bits of
			// n->right->key in positions between n->pos and
			// n->right->pos, excluding n->right->pos, are a superset of x
			// 
			if (x.prefix_subset_of(n->right->key, n->pos, n->right->pos + 1)) 
				S[head++] = n->right;

		if (n->pos > n->left->pos && !x[n->pos]) 
			// push n->left on the stack only when the bits of
			// n->left->key in positions between n->pos and
			// n->left->pos, excluding n->left->pos, are a subset of
			// x
			// 
			if (x.prefix_subset_of(n->left->key, n->pos, n->left->pos + 1)) 
				S[head++] = n->left;
	}
}

void predicate::find_supersets_of(const filter_t & x, filter_handler & h) {
	//
	// See the above "const" find_superset_of for technical details.
	// 
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

		if (n->pos > n->right->pos)
			if (x.prefix_subset_of(n->right->key, n->pos, n->right->pos + 1)) 
				S[head++] = n->right;

		if (n->pos > n->left->pos && !x[n->pos]) 
			if (x.prefix_subset_of(n->left->key, n->pos, n->left->pos + 1)) 
				S[head++] = n->left;
	}
}
