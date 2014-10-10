#ifndef PACKET_HH_INCLUDED
#define PACKET_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#include <climits>
#include <cstring>
#include <cassert>
#include <string>
#include <atomic>

#include "parameters.hh"

/** interface identifier */ 
typedef uint16_t interface_t;

/** tree identifier */ 
typedef uint16_t tree_t;

/** tree--interface pair */ 
class tree_interface_pair {
// 
// ASSUMPTIONS: 
//   1. the router has at most 2^13 = 8192 interfaces
//   2. the are at most 2^3 = 8 trees
//
	uint16_t value;

public:
	static const unsigned int TREE_OFFSET = 13;
	static const uint16_t IFX_MASK = (0xFFFF >> (16 - TREE_OFFSET));

	tree_interface_pair(tree_t t, interface_t ifx)
		: value((t << TREE_OFFSET) | (ifx & IFX_MASK)) {};
	
	tree_interface_pair(const tree_interface_pair & p)
		: value(p.value) {};
	
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

//
// A packet is whatever we get from the network.  At this point, the
// only important components are the filter, which is a 192-bit Bloom
// filter, the tree that the packet is routed on, and the interface it
// comes from.
//
// We assume that the 192-bit Bloom filter is 64-bit aligned quantity
// that we can access as an array of uint64_t, or an array of uint32_t
// values.
//

// 
// We use blocks of 64 bits to represent and process filters and their
// prefixes.
//
typedef uint64_t block_t;
static_assert(sizeof(block_t)*CHAR_BIT == 64, "uint64_t must be a 64-bit word");
static const int BLOCK_SIZE = 64;
static const block_t BLOCK_ONE = 0x1;

//
// Main representation of a prefix.  Essentially we will instantiate
// this template with Size=64, Size=128, and Size=192.
//

template <unsigned int Size>
class prefix {
	static_assert(sizeof(uint64_t)*CHAR_BIT == 64, "uint64_t must be a 64-bit word");
    static_assert((Size % 64) == 0, "filter width must be a multiple of 64");

	static const int BLOCK_SIZE = 64;
	static const block_t BLOCK_ONE = 0x1;

    static const int BLOCK_COUNT = Size / BLOCK_SIZE;
    // 
    // BIT LAYOUT: a prefix is represented with the bit pattern in
    // reverse order.  That is, the first bit is the least significant
    // bit, and the pattern goes from left-to-right from the least
    // significant bit towards the most significant bit.  Notice that
    // we do not store the length of a prefix.  So, a prefix of one
    // bit will still be stored as a 64-bit quantity with all the
    // trailing bits set to 0.
    // 
    // EXAMPLE:
    // prefix "000101" is represented by the three blocks:
    // b[0] = (101000)binary, b[1] = 0, b[2] = 0
    //
    block_t b[BLOCK_COUNT];
    
public:
    const block_t * begin() const {
		return b;
    }

    const block_t * end() const {
		return b + BLOCK_COUNT;
    }

    const uint64_t * begin64() const {
		return b;
    }

    const uint64_t * end64() const {
		return b + BLOCK_COUNT;
    }

	// WARNING: we want to be able to access the bit pattern of the
	// prefix/filter as an array of 32-bit integer.  In this case we
	// simply look at the array of 64-bit unsigned integer as an array
	// of 32-bit integer.  This, as far as I understand, is undefined
	// behavior?
	// 
	const uint32_t * unsafe_begin32() const {
		return reinterpret_cast<const uint32_t *>(b);
	}

	const uint32_t * unsafe_end32() const {
		return reinterpret_cast<const uint32_t *>(b) + (Size / 32);
	}

	const uint32_t uint32_value(unsigned int i) const {
		return (i % 2 == 1) ? (b[i/2] >> 32) : (b[i/2] & 0xffffffffULL);
	}

	const void copy_into_uint32_array(uint32_t * x) const {
		for(int i = 0; i < BLOCK_SIZE; ++i) {
			*x++ = b[i] & 0xffffffff;
			*x++ = b[i] >> 32;
		}
	}

    bool subset_of(const block_t * p) const {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			if ((b[i] & ~p[i]) != 0)
				return false;

		return true;
    }

    prefix(const std::string & p) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = 0;

		assert(p.size() <= Size);

		// see the layout specification above
		//
		block_t mask = BLOCK_ONE;
		int i = 0;
		for(std::string::const_iterator c = p.begin(); c != p.end(); ++c) {
			if (*c == '1')
				b[i] |= mask;

			mask <<= 1;
			if (mask == 0) {
				mask = BLOCK_ONE;
				if (++i == BLOCK_COUNT)
					return;
			}
		}
    }

	prefix() {};

    prefix(const block_t * p) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = p[i];
    }

    prefix(const prefix & p) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = p.b[i];
    }

    void assign(const block_t * p) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = p[i];
    }

    prefix & operator = (const prefix & p) {
		assign(p.b);
		return *this;
    }

	void clear() {
		for(int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = 0;
	}

	void set_bit(unsigned int pos) {
		b[pos/BLOCK_SIZE] |= (BLOCK_ONE << (pos % BLOCK_SIZE));
	}
};

typedef prefix<192> filter_t;

// this class represents the raw data read from the network.
// 
class network_packet {
public:
	filter_t filter;
	tree_interface_pair ti_pair;

	network_packet(const filter_t f, tree_t t, interface_t i)
		: filter(f), ti_pair(t,i) {};

	network_packet(const std::string & f, tree_t t, interface_t i)
		: filter(f), ti_pair(t,i) {};

	network_packet(const block_t * f, tree_t t, interface_t i)
		: filter(f), ti_pair(t,i) {};

	network_packet(const network_packet & p) 
		: filter(p.filter), ti_pair(p.ti_pair) {};
};

// this is the class that defines and stores a packet and the related
// meta-data used during the whole matching and forwarding process.
// In essence, this is a wrapper/descriptor for a network packet.  We
// use it to keep track of a packet during the matching process.  In
// particular, we dispatch each message to the back-end a number of
// times, once for each prefix/partition matched by that message as
// computed by the front end.  So, we must fetch and record the
// partial results of the matching, and we must also realize that the
// whole matching process is done.
//
class packet : public network_packet {
private:
    // the packet is in "frontend" state when we are still working with the prefix
    // matching within the frontend, then it goes into BackEnd state.
	// 
    enum matching_state {
		FrontEnd = 0,
		BackEnd = 1
	};

	matching_state state;

	// we count the number of queues, each associated with a
	// partition, to which we pass this message for matching by the
	// backend.  We use this counter to determine when we are done
	// with the matching in all the partitions.
	//
	std::atomic<unsigned int> pending_partitions;

	// Array of flags (0/1) when a flag is set then we have a match on
	// the corresponding interface.  We could use a bit vector but
	// this should be faster
	// 
	std::atomic<unsigned char> output[INTERFACES]; 
    
public:
	packet(const filter_t f, uint16_t t, uint16_t i)
		: network_packet(f, t, i), state(FrontEnd), pending_partitions(0) {
	};

	packet(const std::string & f, uint16_t t, uint16_t i)
		: network_packet(f, t, i), state(FrontEnd), pending_partitions(0) {
	};

	packet(const block_t * f, uint16_t t, uint16_t i)
		: network_packet(f, t, i), state(FrontEnd), pending_partitions(0) {
	};

	packet(const packet & p) 
		: network_packet(p), state(FrontEnd), pending_partitions(0) {
	};

    void add_partition() {
		++pending_partitions;
    }

    void partition_done() {
		--pending_partitions;
    }

    void reset() {
        state = FrontEnd;
		pending_partitions = 0;
		reset_output();
    }

    void reset_output() {
		for(unsigned int i = 0; i < INTERFACES; ++i)
			output[i].store(0);
    }

    void frontend_done() {
        state = BackEnd;
    }

    bool is_matching_complete() const {
        return (state != FrontEnd) && (pending_partitions == 0);
    }

	bool get_output(unsigned int ifx) const {
		return (output[ifx].load() == 1);
	}

	void set_output(unsigned int ifx) {
		output[ifx].store(1);
	}

	void reset_output(unsigned int ifx) {
		output[ifx].store(0);
	}
};

#endif // PACKET_HH_INCLUDED
