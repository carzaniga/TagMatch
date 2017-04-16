#ifndef MATCH_HANDLER_HH_INCLUDED
#define MATCH_HANDLER_HH_INCLUDED

#include <condition_variable>
#include <mutex>

class packet;

// Handler for asynchronous match operations
//
// This is an abstract class.  An application must define the callback
// method match_done()
//
class match_handler {
public:
	virtual void match_hold() {};
	virtual void match_done(packet * p) = 0;
};

// Handler for synchronous match operations
//
class synchronous_match_handler : public match_handler {
private:
	std::mutex mtx;
	std::condition_variable cv;
	bool done;
	
public:
	synchronous_match_handler()
		: done(false) {};
	synchronous_match_handler(const synchronous_match_handler & other)
		: done(other.done) {};
		
	virtual void match_done(packet * p) {
		std::unique_lock<std::mutex> lock(mtx);
		done = true;
		cv.notify_all(); 
	};

	virtual void match_hold() {
		std::unique_lock<std::mutex> lock(mtx);
		while (! done)
			cv.wait(lock);
	};
};

#endif
