#ifndef SYNCHRONOUS_MATCH_HANDLER_HH_INCLUDED
#define SYNCHRONOUS_MATCH_HANDLER_HH_INCLUDED

#include <cassert>
#include <mutex>
#include <condition_variable>

#include "query.hh"
#include "match_handler.hh"

// Handler for synchronous match operations.
//
// This is an abstract class.  At a minimum, an application must
// define the callback method process_results().
//
// The processing in this case is synchronous in the sense that it is
// a thread that calls synchronize_and_process() that executes
// process_results() as soon as the results are ready.
//
class synchronous_match_handler : public match_handler {
private:
	std::mutex mtx;
	std::condition_variable cv;
	query * ready_query;

public:
	synchronous_match_handler()
		: mtx(), cv(), ready_query(nullptr) {};

	virtual void process_results(query * q) = 0;

	virtual void match_done(query * q) {
		assert(q);
		std::unique_lock<std::mutex> lock(mtx);
		ready_query = q;
		cv.notify_all();
	};

	virtual void synchronize_and_process() {
		query * q;
		do {
			std::unique_lock<std::mutex> lock(mtx);
			while (! ready_query)
				cv.wait(lock);
			q = ready_query;
			ready_query = nullptr;
		} while (0);
		process_results(q);
	};
};

#endif
