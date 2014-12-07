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
#include "bitvector.hh"
#include "io_util.hh"

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

	tree_interface_pair()
		: value(0) {};
	
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

	std::ostream & write_binary(std::ostream & output) const {
		return io_util_write_binary(output, value);
	}

	std::istream & read_binary(std::istream & input) {
		return io_util_read_binary(input, value);
	}

	std::ostream & write_ascii(std::ostream & output) const {
		return output << tree() << ' ' << interface();
	}

	std::istream & read_ascii(std::istream & input) {
		uint16_t t;
		uint16_t i;
		if (input >> t >> i)
			value = ((t << TREE_OFFSET) | (i & IFX_MASK));
		return input;
	}
};

//
// A packet is whatever we get from the network.  At this point, the
// only important components are the filter, which is a 192-bit Bloom
// filter, the tree that the packet is routed on, and the interface it
// comes from.

typedef bitvector<192> filter_t;

// this class represents the raw data read from the network.
// 
class network_packet {
public:
	filter_t filter;
	tree_interface_pair ti_pair;

	network_packet() : filter(), ti_pair() {};

	network_packet(const filter_t f, tree_t t, interface_t i)
		: filter(f), ti_pair(t,i) {};

	network_packet(const std::string & f, tree_t t, interface_t i)
		: filter(f), ti_pair(t,i) {};

	network_packet(const network_packet & p) 
		: filter(p.filter), ti_pair(p.ti_pair) {};

	std::ostream & write_binary(std::ostream & output) const;
	std::istream & read_binary(std::istream & input);
	std::ostream & write_ascii(std::ostream & output) const;
	std::istream & read_ascii(std::istream & input);
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
#ifdef WITH_ATOMIC_OUTPUT
	std::atomic<unsigned char> output[INTERFACES]; 
#else    
	unsigned char output[INTERFACES];
#endif
public:
	packet(const filter_t f, uint16_t t, uint16_t i)
		: network_packet(f, t, i), state(FrontEnd), pending_partitions(0) {
	};

	packet(const std::string & f, uint16_t t, uint16_t i)
		: network_packet(f, t, i), state(FrontEnd), pending_partitions(0) {
	};

	packet(const packet & p) 
		: network_packet(p), state(FrontEnd), pending_partitions(0) {
	};

    void add_partition() {
		++pending_partitions;
    }

    void add_partitions(unsigned int p) {
		if (p > 0)
			pending_partitions += p;
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
#ifdef WITH_ATOMIC_OUTPUT
		for(unsigned int i = 0; i < INTERFACES; ++i)
			output[i].store(0);
#else
		memset(output, 0, sizeof(output));
#endif
    }

    void frontend_done() {
        state = BackEnd;
    }

    bool is_matching_complete() const {
        return (state != FrontEnd) && (pending_partitions == 0);
    }

	bool get_output(unsigned int ifx) const {
#ifdef WITH_ATOMIC_OUTPUT		
		return (output[ifx].load() == 1);
#else
		return (output[ifx] == 1);
#endif
	}

	void set_output(unsigned int ifx) {
#ifdef WITH_ATOMIC_OUTPUT		
		output[ifx].store(1);
#else
		output[ifx] = 1;
#endif
	}

	void reset_output(unsigned int ifx) {
#ifdef WITH_ATOMIC_OUTPUT		
		output[ifx].store(0);
#else
		output[ifx] = 0;
#endif
	}
};

#endif // PACKET_HH_INCLUDED
