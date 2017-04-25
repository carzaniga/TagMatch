#ifndef TAGMATCH_QUERY_HH_INCLUDED
#define TAGMATCH_QUERY_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <atomic>
#include <vector>
#include <mutex>
#include <cassert>

#include "filter.hh"
#include "key.hh"
#include "query.hh"
#include "match_handler.hh"

// this is the class that defines and stores a query and the related
// meta-data used during the whole matching and forwarding process by
// the CPU/GPU TagNet matcher.  In essence, this is a wrapper for a
// query (plus results) that keeps track of whether the query is in
// the front-end or back-end, and how many partitions needs
// processing.
//
class tagmatch_query : public query {
private:
	enum {
		FRONT_END = 0,
		BACK_END = 1,
		FINALIZED = 2,
	};
	std::atomic<unsigned char> state;
	// We count the number of queues, each associated with a
	// partition, to which we pass this message for matching by the
	// backend.  We use this counter to determine when we are done
	// with the matching in all the partitions.
	//
	std::atomic<unsigned int> pending_partitions;
	match_handler * handler;
	std::mutex output_mtx;

public:
	tagmatch_query()
		: query(), state(FRONT_END), pending_partitions(0), handler(nullptr) { };

	tagmatch_query(const filter_t & f)
		: query(f), state(FRONT_END), pending_partitions(0), handler(nullptr) { };

	tagmatch_query(const std::string & f)
		: query(f), state(FRONT_END), pending_partitions(0), handler(nullptr) { };

	tagmatch_query(tagmatch_query && x)
		: query(std::move(x)),
		  state(x.state.load()), pending_partitions(x.pending_partitions.load()), handler(x.handler), output_mtx() {
		x.handler = nullptr;
	};

	tagmatch_query(const tagmatch_query & x)
		: query(x),
		  state(x.state.load()), pending_partitions(x.pending_partitions.load()), handler(x.handler), output_mtx() { };

    void partition_enqueue() {
		++pending_partitions;
    }

    void partition_done() {
		--pending_partitions;
    }

    void reset() {
        state = FRONT_END;
		pending_partitions = 0;
    }

    void frontend_done() {
		unsigned char s = FRONT_END;
		state.compare_exchange_strong(s, BACK_END);
    }

	void add_output(tagmatch_key_t key) {
		// Warning: you should lock the mutex before calling this method!
		output_mtx.lock();
		output_keys.push_back(key);
		output_mtx.unlock();
	}

	void add_output(std::vector<tagmatch_key_t>::const_iterator k,
					std::vector<tagmatch_key_t>::const_iterator end) {
		output_mtx.lock();
		for (; k != end; ++k)
			output_keys.push_back(*k);
		output_mtx.unlock();
	}

    void set_match_handler(match_handler * h) {
		handler = h;
    }

	bool finalize_matching() {
		if (pending_partitions > 0)
			return false;

		unsigned char s = BACK_END;
		if (state.compare_exchange_strong(s, FINALIZED)) {
			if (match_unique) {
				// Delete duplicates from the list of output keys
				std::sort(output_keys.begin(), output_keys.end());
				output_keys.erase(std::unique(output_keys.begin(), output_keys.end()),
								  output_keys.end());
			}
			if (handler)
				handler->match_done(this);
			return true;
		}
		else {
			// Nothing to do
			return false;
		}
	}
};

#endif // QUERY_HH_INCLUDED
