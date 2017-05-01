#ifndef FILTER_HH_INCLUDED
#define FILTER_HH_INCLUDED

#include <cstdint>
#include "bitvector.hh"

/** A Bloom filter representing a tag set.
 */
typedef bitvector<192> filter_t;
typedef uint8_t filter_pos_t;

#endif
