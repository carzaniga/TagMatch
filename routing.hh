#ifndef ROUTING_HH_INCLUDED
#define ROUTING_HH_INCLUDED

#include <cstdint>

//
// This defines the main entities and concepts in the TagNet muti-tree
// routing scheme.  At this point, these are pretty basic concepts
// (router interfaces and trees).
//
// ASSUMPTIONS:
//   1. a router has at most 2^13 = 8192 interfaces
//   2. the are at most 2^3 = 8 trees
//

/** interface identifier */ 
typedef uint16_t interface_t;

/** tree identifier */ 
typedef uint16_t tree_t;

/** tree--interface pair */ 
typedef uint16_t tree_interface_pair;

static const unsigned int TIP_TREE_OFFSET = 13;
static const uint16_t TIP_IFX_MASK = (0xFFFF >> (16 - TIP_TREE_OFFSET));

inline tree_interface_pair tip_value(tree_t t, interface_t ifx) {
	return ((t << TIP_TREE_OFFSET) | (ifx & TIP_IFX_MASK));
}

inline tree_t tip_tree(tree_interface_pair value) {
	return value >> TIP_TREE_OFFSET;
}

inline interface_t tip_interface(tree_interface_pair value) {
	return value & TIP_IFX_MASK;
}

inline tree_t tip_uint16_value(tree_interface_pair value) {
	return value;
}

#endif
