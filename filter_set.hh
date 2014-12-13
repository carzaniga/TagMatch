#ifndef FILTER_SET_HH_INCLUDED
#define FILTER_SET_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "bitvector.hh"

typedef bitvector<192> filter_t;

class filter_set {
public:
	static void clear();
	static void add(const filter_t & x);
	static bool find(const filter_t & x);
	static size_t count_subsets_of(const filter_t & x);
	static size_t count_supersets_of(const filter_t & x);
};

#endif
