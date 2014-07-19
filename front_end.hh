#ifndef FRONT_END_HH_INCLUDED
#define FRONT_END_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef WITH_FRONTEND_STATISTICS
#include <iostream>
#endif

#include "packet.hh"

/// add a prefix f of length n to the front end FIB
/// 
class front_end {
public:
	static void add_prefix(unsigned int id, const prefix<192> & f, unsigned int n);
	static void start(unsigned int n);
	static void match(packet * p);
	static void stop();
	static void clear();
#ifdef WITH_FRONTEND_STATISTICS
	static std::ostream & print_statistics(std::ostream &);
#endif
};

#endif // FRONT_END_HH_INCLUDED
