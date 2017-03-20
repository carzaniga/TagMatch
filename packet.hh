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
#include <vector>
#include <algorithm>

#include "parameters.hh"
#include "bitvector.hh"
#include "io_util.hh"
#include <mutex> 

typedef uint32_t interface_t;
typedef uint32_t tagmatch_key_t;

#if 0
/** key class */ 
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
#endif

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
	// TODO: is this thing here still useful at all? Does it state the origin of the packet?
	tagmatch_key_t key;

	network_packet() : filter(), key() {};

	network_packet(const filter_t f, interface_t i)
		: filter(f), key(i) {};

	network_packet(const std::string & f, interface_t i)
		: filter(f), key(i) {};

	network_packet(const network_packet & p) 
		: filter(p.filter), key(p.key) {};

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
#else   
	std::mutex mtx;
	std::vector<uint32_t> output_users;
#ifdef WITH_MATCH_STATISTICS
	std::atomic<uint32_t> pre,post;
#endif
	std::atomic<bool> finalized;
#endif
public:
	packet(const filter_t f, uint32_t i)
		: network_packet(f, i), state(FrontEnd), pending_partitions(0) {
			finalized = false;
	};

	packet(const std::string & f, uint32_t i)
		: network_packet(f, i), state(FrontEnd), pending_partitions(0) {
			finalized = false;
	};

	packet(const packet & p) 
		: network_packet(p), state(FrontEnd), pending_partitions(0) {
#ifdef WITH_MATCH_STATISTICS
		pre = 0;
		post = 0;
#endif
		output_users = p.output_users;
		finalized = false;
	};

    void add_partition() {
		++pending_partitions;
    }

    void add_partition(uint32_t id) {
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
		finalized = 0;
    }

    void frontend_done() {
        state = BackEnd;
    }

    bool is_matching_complete() const {
        return (state != FrontEnd) && (pending_partitions == 0);
    }

	void lock_mtx() {
		mtx.lock();
	}

	void unlock_mtx() {
		mtx.unlock();
	}

	void add_output_user(uint32_t user) {
		//Warning: you should lock the mutex before calling this method! 
		output_users.push_back(user);
	}

	std::vector<uint32_t> get_users() {
		return output_users;
	}

	bool finalize_matching() {
		// This is where we should implement the actual forwarding code.
		// In this experimental code, instead, we simply collect statistics
		// about the matching.  In order to do that, we still perform a "merge"
		// over the set of output "users".
		if (!atomic_exchange(&finalized, true)) {
#ifdef WITH_MATCH_STATISTICS
			pre = output_users.size();
			assert(pre > 0);
#endif
#if 1
			std::sort( output_users.begin(), output_users.end() );
			output_users.erase( unique( output_users.begin(), output_users.end() ), output_users.end() );
#endif
#ifdef WITH_MATCH_STATISTICS
			post = output_users.size();
#endif
			//Flush the output vector and release its memory
			output_users.clear();
			std::vector<uint32_t>().swap(output_users);
			// For some reason output.resize(0) doesn't really free the memory
			// output_users.resize(0);
			return true;
		}
		else {
			//Nothing to do
			return false;
		}
	}

#ifdef WITH_MATCH_STATISTICS
	uint32_t getpre() {
		return pre;
	}

	uint32_t getpost() {
		return post;
	}
#endif
};

#endif // PACKET_HH_INCLUDED
