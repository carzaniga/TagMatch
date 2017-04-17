#ifndef TIP_ARRAY_HH_INCLUDED
#define TIP_ARRAY_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#include <cstring>

#include "routing.hh"
#include "filter.hh"
#include "patricia_predicate.hh"

//
// tip_array implements a compact, dynamic array of
// tree_interface_pair values.  At this point it only support growing
// arrays, so only addition, but it could be easily extended to
// support removal.
//
// DESIGN: store up to LOCAL_CAPACITY tree_interface_pair objects
// locally, and when more are needed, move them to an external,
// dynamically allocated array.
//
class tip_array {
private:
	// LOCAL_CAPACITY should be at least the maximal value that yields
	// the minimal sizeof(tip_array).  On a 64-bit system, that is
	// typically 4, assuming tree_interface_pair are 16-bit values.
	//
	static const unsigned int LOCAL_CAPACITY = 4;

	uint16_t size;
	union {
		tree_interface_pair local_tips[LOCAL_CAPACITY];
		tree_interface_pair * external_tips;
	};

public:
	tip_array() : size(0) {};

	~tip_array() {
		if (size > LOCAL_CAPACITY)
			delete[](external_tips);
	}

	// copy constructor
	tip_array(const tip_array & other);

	// move constructor (C++ 11)
	tip_array(tip_array && other) : size(other.size) {
		if (size > LOCAL_CAPACITY)
			external_tips = other.external_tips;
		else
			memcpy(local_tips, other.local_tips, size*sizeof(tree_interface_pair));
		other.size = 0;
	}

	// move assignment (C++ 11)
	tip_array & operator = (tip_array && other) {
		size = other.size;
		if (size > LOCAL_CAPACITY)
			external_tips = other.external_tips;
		else
			memcpy(local_tips, other.local_tips, size*sizeof(tree_interface_pair));
		other.size = 0;
		return *this;
	}

	// copy assignment, with ownership transfer
	tip_array & operator = (tip_array & other) {
		size = other.size;
		if (size > LOCAL_CAPACITY)
			external_tips = other.external_tips;
		else
			memcpy(local_tips, other.local_tips, size*sizeof(tree_interface_pair));
		other.size = 0;
		return *this;
	}

	void add(tree_interface_pair tip);

	const tree_interface_pair * begin() const {
		return (size > LOCAL_CAPACITY) ? external_tips : local_tips;
	}

	const tree_interface_pair * end() const {
		return begin() + size;
	}
};

#endif
