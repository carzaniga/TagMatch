#ifndef PREDICATE_HH_INCLUDED
#define PREDICATE_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "filter.hh"
#include "matcher.hh"
#include "routing.hh"

class predicate {      
public:
    predicate();
    ~predicate();

    void clear();
	void match(const filter_t & x, match_handler & h) const;

	void add(const filter_t & x, const tree_interface_pair * b, const tree_interface_pair * e);
	void add(const filter_t & x, tree_t t, interface_t i);

	class node;
private:
	node * add(const filter_t & x);
	
    node * root;    
};

#endif
