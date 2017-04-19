#ifndef QUERY_HH_INCLUDED
#define QUERY_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#include <climits>
#include <cstring>
#include <cassert>
#include <string>
#include <atomic>
#include <vector>
#include <algorithm>
#include <mutex>

#include "filter.hh"
#include "key.hh"

// We process a stream of queries.  Each query comes into the system
// in the form of a "query".  The original idea of TagMatch derives
// from TagNet and is to literally process network querys.  However,
// what we call a query here is more generally an incoming query that
// at a minimum contains a tag set (filter) and possibly additional
// information.


// This class represents a "naked" query.
//
class basic_query {
public:
	filter_t filter;
	bool match_unique;

	basic_query() : filter(), match_unique(false) {};
	basic_query(const filter_t & f) : filter(f), match_unique(false) {};
	basic_query(const std::string & f) : filter(f), match_unique(false) {};
	basic_query(const basic_query & q) : filter(q.filter), match_unique(q.match_unique) {};

	std::ostream & write_binary(std::ostream & output) const;
	std::istream & read_binary(std::istream & input);
	std::ostream & write_ascii(std::ostream & output) const;
	std::istream & read_ascii(std::istream & input);
};

// This class represents a query plus its matching keys.
//
class query : public basic_query {
public:
	std::vector<tagmatch_key_t> output_keys;

	query() : basic_query() {};
	query(const filter_t & f) : basic_query(f) {};
	query(const std::string & f) : basic_query(f) {};
	query(const query & q) : basic_query(q.filter), output_keys(q.output_keys) {};
	query(query && q) : basic_query(q), output_keys(std::move(q.output_keys)) {};
};

#endif // QUERY_HH_INCLUDED
