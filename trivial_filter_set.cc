#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <vector>
#include <algorithm>

#include "filter_set.hh"
#include "packet.hh"

static std::vector<filter_t> fib;

void filter_set::clear() {
	fib.clear();
}

void filter_set::add(const filter_t & f) {
	fib.emplace_back(f);
}

bool filter_set::find(const filter_t & f) {
	return (std::find(fib.begin(), fib.end(), f) != fib.end());
}

size_t filter_set::count_subsets_of(const filter_t & x) {
	size_t result = 0;
	for(auto & f : fib) 
		if(f.subset_of(x))
			++result;
	return result;
}

size_t filter_set::count_supersets_of(const filter_t & x) {
	size_t result = 0;
	for(auto & f : fib) 
		if(x.subset_of(f))
			++result;
	return result;
}
