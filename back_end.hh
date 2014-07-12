#ifndef BACK_END_HH_INCLUDED
#define BACK_END_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <vector>

#include "packet.hh"

/// add a prefix f of length n to the front end FIB
/// 
class back_end {
public:
	static void process_batch(unsigned int part, packet ** batch, unsigned int batch_size);
	static void add_filter(unsigned int partition, const filter_t & f, 
						   std::vector<tree_interface_pair>::const_iterator begin,
						   std::vector<tree_interface_pair>::const_iterator end);

	static void start();
	static void shutdown();
};

#endif // BACK_END_HH_INCLUDED
