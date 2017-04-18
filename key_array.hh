#ifndef KEY_ARRAY_HH_INCLUDED
#define KEY_ARRAY_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#include <cstring>

#include "tagmatch.hh"

//
// key_array implements a compact, dynamic array of tagmatch_key_t values.  At
// this point it only support growing arrays, so only addition, but it
// could be easily extended to support removal.
//
// DESIGN: store up to LOCAL_CAPACITY tagmatch_key_t objects locally, and when
// more are needed, move them to an external, dynamically allocated
// array.
//
class key_array {
private:
	// LOCAL_CAPACITY should be at least the maximal value that yields
	// the minimal sizeof(key_array).  On a 64-bit system, that is
	// typically 2, assuming tagmatch_key_t are 32-bit values.
	//
	static const unsigned int LOCAL_CAPACITY = 4;

	uint16_t size;
	union {
		tagmatch_key_t local_keys[LOCAL_CAPACITY];
		tagmatch_key_t * external_keys;
	};

public:
	key_array() : size(0) {};

	~key_array() {
		if (size > LOCAL_CAPACITY)
			delete[](external_keys);
	}

	// copy constructor
	key_array(const key_array & other);

	// move constructor (C++ 11)
	key_array(key_array && other) : size(other.size) {
		if (size > LOCAL_CAPACITY)
			external_keys = other.external_keys;
		else
			memcpy(local_keys, other.local_keys, size*sizeof(tagmatch_key_t));
		other.size = 0;
	}

	// move assignment (C++ 11)
	key_array & operator = (key_array && other) {
		size = other.size;
		if (size > LOCAL_CAPACITY)
			external_keys = other.external_keys;
		else
			memcpy(local_keys, other.local_keys, size*sizeof(tagmatch_key_t));
		other.size = 0;
		return *this;
	}

	// copy assignment, with ownership transfer
	key_array & operator = (key_array & other) {
		size = other.size;
		if (size > LOCAL_CAPACITY)
			external_keys = other.external_keys;
		else
			memcpy(local_keys, other.local_keys, size*sizeof(tagmatch_key_t));
		other.size = 0;
		return *this;
	}

	void add(tagmatch_key_t key);

	const tagmatch_key_t * begin() const {
		return (size > LOCAL_CAPACITY) ? external_keys : local_keys;
	}

	const tagmatch_key_t * end() const {
		return begin() + size;
	}
};

#endif
