#ifndef MATCH_HANDLER_HH_INCLUDED
#define MATCH_HANDLER_HH_INCLUDED

#include "query.hh"

// Handler for asynchronous match operations
//
// This is an abstract class.  At a minimum, an application must
// define the callback method process_results().
//
// This handler is asynchronous in the sense that it is a thread of
// the matcher that processes the results by running match_done(),
// which immediately calls process_results().
//
class match_handler {
public:
	virtual void match_done(query * q) {
		process_results(q);
	}
	virtual void process_results(query * q) = 0;
	virtual ~match_handler() { };
};

#endif
