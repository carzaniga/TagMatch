#ifndef MATCHER_HH_INCLUDED
#define MATCHER_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "filter.hh"
#include "routing.hh"

class match_handler {
public:
	// this will be called by predicate::match().  The return value
	// indicates whether the search for matching filters should stop.
	// So, if this function returns TRUE, match() will terminate
	// immediately.
	// 
	virtual bool match(const filter_t & filter, tree_t tree, interface_t ifx) = 0;
};

#endif // header guard
