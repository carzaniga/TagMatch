#ifndef MATCH_HANDLER_HH_INCLUDED
#define MATCH_HANDLER_HH_INCLUDED

#include <condition_variable>
#include "packet.hh"

// handler for asynchronous match operations
//
// this is an abstract class.  An application must define the callback method match_done()
//
class match_handler {
	public:
		packet * p;
		bool match_unique;

		virtual void match_hold() {};
		virtual void match_done() = 0;
};

// handler for synchronous match operations
class synchronous_match_handler : public match_handler {
	private:
		std::mutex mtx;
		std::condition_variable cv;
		bool done;
	
	public:
		synchronous_match_handler() : done(false) {};
		
		synchronous_match_handler(packet * pkt) : done(false) {
			p = pkt;
		};

		synchronous_match_handler(const synchronous_match_handler & smh) {
			done = smh.done;
			p = smh.p;
		};
		
		virtual void match_done() {
			std::unique_lock<std::mutex> lock(mtx);
			done = true;
			cv.notify_all(); // TODO: or one... which is better?
		};

		virtual void match_hold() {
			std::unique_lock<std::mutex> lock(mtx);
			while (! done)
				cv.wait(lock);
		};
};

#endif
