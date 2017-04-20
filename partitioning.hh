#ifndef PARTITIONING_HH_INCLUDED
#define PARTITIONING_HH_INCLUDED

#include <vector>

#include "tagmatch.hh"
#include "fib.hh"

class partitioning {
public:
	static void balanced_partitioning(std::vector<partition_fib_entry *> & fib,
									  std::vector<partition_prefix> & masks);

	static void set_maxp(unsigned int size);
	static unsigned int get_maxp();

	static void set_cpu_threads(unsigned int t);
	static unsigned int get_cpu_threads();
};

#endif
