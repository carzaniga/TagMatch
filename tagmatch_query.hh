#ifndef TAGMATCH_QUERY_HH_INCLUDED
#define TAGMATCH_QUERY_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <atomic>
#include <vector>
#include <mutex>

#include "filter.hh"
#include "key.hh"
#include "query.hh"

// this is the class that defines and stores a query and the related
// meta-data used during the whole matching and forwarding process by
// the CPU/GPU TagNet matcher.  In essence, this is a wrapper for a
// query (plus results) that keeps track of whether the query is in
// the front-end or back-end, and how many partitions needs
// processing.
//
class tagmatch_query : public query {
private:
    // the query is in "frontend" state when we are still working with the prefix
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
	std::atomic<bool> finalized;
	match_handler * handler;
	std::mutex output_mtx;

#ifdef WITH_MATCH_STATISTICS
	std::atomic<unsigned int> pre;
	std::atomic<unsigned int> post;
#endif

public:
	tagmatch_query()
		: query(),
		  state(FrontEnd), pending_partitions(0), finalized(false), handler(nullptr)
#ifdef WITH_MATCH_STATISTICS
		, pre(0), post(0)
#endif
		{ };

	tagmatch_query(const filter_t & f)
		: query(f),
		  state(FrontEnd), pending_partitions(0), finalized(false), handler(nullptr)
#ifdef WITH_MATCH_STATISTICS
		, pre(0), post(0)
#endif
		{ };

	tagmatch_query(const std::string & f)
		: query(f),
		  state(FrontEnd), pending_partitions(0), finalized(false), handler(nullptr)
#ifdef WITH_MATCH_STATISTICS
		, pre(0), post(0)
#endif
		{ };

	tagmatch_query(tagmatch_query && q)
		: query(std::move(q)),
		  state(q.state), pending_partitions(q.pending_partitions.load()),
		  finalized(q.finalized.load()), handler(nullptr),
		  output_mtx()
#ifdef WITH_MATCH_STATISTICS
		, pre(q.pre.load()), post(q.post.load())
#endif
		{ };

	tagmatch_query(const tagmatch_query & q)
		: query(q),
		  state(q.state), pending_partitions(q.pending_partitions.load()),
		  finalized(q.finalized.load()), handler(nullptr),
		  output_mtx()
#ifdef WITH_MATCH_STATISTICS
		, pre(q.pre.load()), post(q.post.load())
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
	unsigned int getpre() {
		return pre;
	}

	unsigned int getpost() {
		return post;
	}
#endif
};

#endif // QUERY_HH_INCLUDED
