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
#include <mutex>

#include "tagmatch.hh"
#include "io_util.hh"

// We process a stream of queries.  Each query comes into the system
// in the form of a "packet".  The original idea of TagMatch derives
// from TagNet and is to literally process network packets.  However,
// what we call a packet here is more generally an incoming query that
// at a minimum contains a tag set (filter) and possibly additional
// information.


// This class represents the raw data read from the network.
//
class network_packet {
public:
	filter_t filter;

	network_packet() : filter() {};

	network_packet(const filter_t f)
		: filter(f) {};

	network_packet(const std::string & f)
		: filter(f) {};

	network_packet(const network_packet & p)
		: filter(p.filter) {};

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

	std::mutex mtx;
	std::vector<uint32_t> output_keys;
	std::atomic<bool> finalized;
	bool match_unique;
	match_handler * handler;

#ifdef WITH_MATCH_STATISTICS
	std::atomic<uint32_t> pre;
	std::atomic<uint32_t> post;
#endif

public:
	packet(const filter_t f)
		: network_packet(f), state(FrontEnd), pending_partitions(0),
		  finalized(false), match_unique(false), handler(nullptr)
#ifdef WITH_MATCH_STATISTICS
		, pre(0), post(0)
#endif
		{ };

	packet(const std::string & f)
		: network_packet(f), state(FrontEnd), pending_partitions(0),
		  finalized(false), match_unique(false), handler(nullptr)
#ifdef WITH_MATCH_STATISTICS
		, pre(0), post(0)
#endif
		{ };

	packet(const packet & p)
		: network_packet(p.filter), state(FrontEnd), pending_partitions(0),
		  output_keys(p.output_keys),
		  finalized(false), match_unique(false), handler(nullptr)
#ifdef WITH_MATCH_STATISTICS
		, pre(0), post(0)
#endif
		{ };

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
		finalized = false;
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

	void add_output_key(tagmatch_key_t key) {
		// Warning: you should lock the mutex before calling this method!
		output_keys.push_back(key);
	}

	std::vector<uint32_t> get_output_keys() {
		return output_keys;
	}

    void configure_match(bool unique, match_handler * h) {
		match_unique = unique;
		handler = h;
    }

	bool finalize_matching() {
		if (!atomic_exchange(&finalized, true)) {
#ifdef WITH_MATCH_STATISTICS
			pre = output_keys.size();
			assert(pre > 0);
#endif
			if (match_unique) {
				// Delete duplicates from the list of output keys
				std::sort( output_keys.begin(), output_keys.end() );
				output_keys.erase( unique(output_keys.begin(), output_keys.end() ), output_keys.end());
			}
#ifdef WITH_MATCH_STATISTICS
			post = output_keys.size();
#endif
			if (handler)
				handler->match_done(this);
			return true;
		}
		else {
			// Nothing to do
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
