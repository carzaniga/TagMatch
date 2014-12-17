#ifndef BACK_END_HH_INCLUDED
#define BACK_END_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdlib>
#include <vector>
#include <map>

#include "packet.hh"

/// add a prefix f of length n to the front end FIB
/// 
class back_end {
public:
	static void add_filter(unsigned int partition, const filter_t & f, 
						   std::vector<tree_interface_pair>::const_iterator begin,
						   std::vector<tree_interface_pair>::const_iterator end);
	static void start(std::map<unsigned int, unsigned char> * prefix_lengths );
	static void process_batch(unsigned int part, packet ** batch, unsigned int batch_size);
	static void stop();
	static void clear();

	static size_t bytesize();

	static void analyze_fibs();
};

#endif // BACK_END_HH_INCLUDED
