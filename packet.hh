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

typedef uint32_t interface_t;
/** tree--interface pair */ 
class tree_interface_pair {
// 
// ASSUMPTIONS: 
//   1. the router has at most 2^13 = 8192 interfaces
//   2. the are at most 2^3 = 8 trees
//
	uint32_t value;

public:

	tree_interface_pair()
		: value(0) {};
	
	tree_interface_pair(const tree_interface_pair & p)
		: value(p.value) {};
	tree_interface_pair(interface_t ifx) {
		value =ifx  ;
	}

	bool equals(interface_t ifx) const {
		return value == ifx ;
	}
	uint32_t get_uint32_value() const {
		return value;
	}
	//is this cast correct?
	uint32_t interface() const {
		return value;
	}
	//     static uint16_t tree(uint16_t value) {
	//     +//             return value >> TREE_OFFSET;;
	//     +//     }
	//     +//
	//     +//     static uint16_t interface(uint16_t value) {
	//     +//             return value & IFX_MASK;
	//     +//     }
	//
	
	bool operator < (const tree_interface_pair &x) const {
		return value < x.value;
	}
	bool operator == (const tree_interface_pair & rhs) const {
		return value == rhs.value;
	}

	std::ostream & write_binary(std::ostream & output) const {
		return io_util_write_binary(output, value);
	}

	std::istream & read_binary(std::istream & input) {
		return io_util_read_binary(input, value);
	}

	std::ostream & write_ascii(std::ostream & output) const {
		return output << interface();
	}

	std::istream & read_ascii(std::istream & input) {
		uint32_t i;
		if (input >> i) {
			value = i ;
		}
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

	network_packet(const filter_t f, interface_t i)
		: filter(f), ti_pair(i) {};

	network_packet(const std::string & f, interface_t i)
		: filter(f), ti_pair(i) {};

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
	int mcount = 0;
	uint32_t matched_subscriptions[MAX_MATCHES_PP];
#endif
public:
	packet(const filter_t f, uint32_t i)
		: network_packet(f, i), state(FrontEnd), pending_partitions(0) {
	};

	packet(const std::string & f, uint32_t i)
		: network_packet(f, i), state(FrontEnd), pending_partitions(0) {
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

	void set_matched_subscription(uint32_t sub) {
		if (mcount < MAX_MATCHES_PP)
			matched_subscriptions[mcount++] = sub;
	}

	uint32_t get_matches_count() {
		return mcount;
	}

	uint32_t get_match(uint32_t i) {
		return matched_subscriptions[i];
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
