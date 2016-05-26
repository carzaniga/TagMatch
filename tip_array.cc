#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#include <cstring>

#include "routing.hh"
#include "filter.hh"
#include "tip_array.hh"

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

void tip_array::add(tree_interface_pair tip) {
    if (size < LOCAL_CAPACITY) {
		local_tips[size++] = tip;
		return;
	}
	if (size == LOCAL_CAPACITY) {
		// allocate external tips for the first time
		tree_interface_pair * new_tip_vector = new tree_interface_pair[INITIAL_EXTERNAL_CAPACITY];
		memcpy(new_tip_vector, local_tips, sizeof(local_tips));
		external_tips = new_tip_vector;
    } else if (is_power_of_two(size)) {
		// size is a power of two, which means that we reached the
		// maximum capacity of the current allocated buffer, so we
		// proceed to reallocate it.  But only if the size variable
		// itself would not overflow.  In that case we silently ignore
		// the add request.
		//
		if (size > UINT32_MAX/2)
			return;
		tree_interface_pair * new_tip_vector = new tree_interface_pair[size*2];
		memcpy(new_tip_vector, local_tips, size*sizeof(tree_interface_pair));
		delete[](external_tips);
		external_tips = new_tip_vector;
    }
    external_tips[size++] = tip;
}
