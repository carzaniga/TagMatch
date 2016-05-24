#ifndef ROUTING_HH_INCLUDED
#define ROUTING_HH_INCLUDED

//
// This defines the main entities and concepts in the TagNet muti-tree
// routing scheme.  At this point, these are pretty basic concepts
// (router interfaces and trees).
//

/** interface identifier */ 
typedef uint16_t interface_t;

/** tree identifier */ 
typedef uint16_t tree_t;

/** tree--interface pair */ 
class tree_interface_pair {
// 
// ASSUMPTIONS: 
//   1. a router has at most 2^13 = 8192 interfaces
//   2. the are at most 2^3 = 8 trees
//
protected:
	uint16_t value;

public:
	static const unsigned int TREE_OFFSET = 13;
	static const uint16_t IFX_MASK = (0xFFFF >> (16 - TREE_OFFSET));

	tree_interface_pair() {};
	
	tree_interface_pair(tree_t t, interface_t ifx)
		: value((t << TREE_OFFSET) | (ifx & IFX_MASK)) {};
	
	tree_interface_pair & operator = (const tree_interface_pair & p) {
		value = p.value;
		return *this;
	}

	bool operator < (const tree_interface_pair &x) const {
		return value < x.value;
	}
	bool operator == (const tree_interface_pair & rhs) const {
		return value == rhs.value;
	}
	bool equals(tree_t t, interface_t ifx) const {
		return (value == ((t << TREE_OFFSET) | (ifx & IFX_MASK)));
	}
	uint16_t get_uint16_value() const {
		return value;
	}

	void assign_uint16_value(uint16_t v) {
		value = v;
	}

	void assign(tree_t t, interface_t ifx) {
		value = ((t << TREE_OFFSET) | (ifx & IFX_MASK));
	}

	uint16_t tree() const {
		return value >> TREE_OFFSET;;
	}

	uint16_t interface() const {
		return value & IFX_MASK;
	}

	static uint16_t tree(uint16_t value) {
		return value >> TREE_OFFSET;;
	}

	static uint16_t interface(uint16_t value) {
		return value & IFX_MASK;
	}
};

#endif
