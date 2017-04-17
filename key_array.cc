#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#include <cstring>

#include "tagmatch.hh"
#include "key_array.hh"

// DESIGN: we do not store the allocated size, because we know it to
// be the smallest power of two that is greater or equal to the
// current size.
//
// In order for this to work, we must use an initial capacity that is
// a power of two (and must also be greater than LOCAL_CAPACITY).
//
static const unsigned int INITIAL_EXTERNAL_CAPACITY = 8;

static bool is_power_of_two(uint16_t x) {
	return ((x) & ((x) - 1)) == 0;
}

key_array::key_array(const key_array & other) : size(other.size) {
	if (size > LOCAL_CAPACITY) {
		external_keys = other.external_keys;
		unsigned int allocated_size = INITIAL_EXTERNAL_CAPACITY;
		while (allocated_size < size)
			allocated_size *= 2;
		external_keys = new tagmatch_key_t[allocated_size];
		memcpy(external_keys, other.external_keys, size*sizeof(tagmatch_key_t));
	} else {
		memcpy(local_keys, other.local_keys, size*sizeof(tagmatch_key_t));
	}
}

void key_array::add(tagmatch_key_t key) {
    if (size < LOCAL_CAPACITY) {
		local_keys[size++] = key;
		return;
	}
	if (size == LOCAL_CAPACITY) {
		// allocate external keys for the first time
		tagmatch_key_t * new_key_vector = new tagmatch_key_t[INITIAL_EXTERNAL_CAPACITY];
		memcpy(new_key_vector, local_keys, sizeof(local_keys));
		external_keys = new_key_vector;
    } else if (is_power_of_two(size)) {
		// size is a power of two, which means that we reached the
		// maximum capacity of the current allocated buffer, so we
		// proceed to reallocate it.  But only if the size variable
		// itself would not overflow.  In that case we silently ignore
		// the add request.
		//
		if (size > UINT32_MAX/2)
			return;
		tagmatch_key_t * new_key_vector = new tagmatch_key_t[size*2];
		memcpy(new_key_vector, external_keys, size*sizeof(tagmatch_key_t));
		delete[](external_keys);
		external_keys = new_key_vector;
    }
    external_keys[size++] = key;
}
