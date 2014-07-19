#ifdef HAVE_CONFIG_H
#include "config.h"
#else
#define HAVE_BUILTIN_CTZL
#endif

#include <cassert>
#include <cstring>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>

#ifdef WITH_FRONTEND_STATISTICS
#include <iostream>
#endif

#include "parameters.hh"
#include "packet.hh"
#include "front_end.hh"
#include "back_end.hh"

using std::vector;
using std::thread;
using std::atomic;
using std::mutex;
using std::unique_lock;
using std::condition_variable;

#ifdef WITH_FRONTEND_STATISTICS
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_values;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::ostream;
#endif

// ***DESIGN OF THE FORWARDING TABLE***
//
// The FIB maps a filter prefix to a queue.
// 
// FIB: prefix -> queue
// 
// The queues are used to batch packets that match the corresponding
// prefix.  When the queue is full, we ship the whole batch to
// back-end.
//
// Filters are 192-bit wide.  Thus a prefix is at most 192-bit wide.
//
//
// MAIN IDEA: we partition the set of prefixes by the position X of
// their leftmost 1-bit.  This is the leftmost bit equal to 1.  Thus
// we have a map Prefixes: {0...191} -> (prefix -> queue).
// 
// Since we represent filters using 64-bit blocks, we further split
// the map into three vectors of 64 positions each.  The first vector
// pp1 holds all the prefixes whose leftmost 1-bit is in position
// 0<=X<64; the second vector pp2 holds all the prefixes whose
// leftmost 1-bit is in position 64<=X<128; and the third vector pp3
// holds all the prefixes whose leftmost 1-bit is in position X>=128.
// We then further split each vector as follows: pp1.p64 contains
// prefixes of total lenght <= 64; pp1.p128 contains prefixes of total
// lenght between 64 and 128; pp1.p192 contains prefixes of total
// lenght > 128; similarly, pp2.p64 and pp3.p64 contains prefixes of
// total lenght <= 64; and pp2.p128 contains prefixes of total length
// between 128 and 192.
//
// This way we can use specialized subset checks for each combination
// of block-length.

//
// IMPORTANT:
// BIT LAYOUT: we represent a prefix with the bit pattern in reverse
// order.  That is, the first bit is the least significant bit, and
// the pattern goes from left-to-right from the least significant bit
// towards the most significant bit.
//

//
// leftmost 1-bit position 
//
#ifdef HAVE_BUILTIN_CTZL
static inline int leftmost_bit(const uint64_t x) {
    // Since we represent the leftmost bit in the least-significant
    // position, the leftmost bit corresponds to the count of trailing
    // zeroes (see the layout specification above).
    return __builtin_ctzl(x);
} 
#else
static inline int leftmost_bit(uint64_t x) {
    int n = 0;
	if ((x & 0xFFFFFFFF) == 0) {
		n += 32;
		x >>= 32;
	}
	if ((x & 0xFFFF) == 0) {
		n += 16;
		x >>= 16;
	}
	if ((x & 0xFF) == 0) {
		n += 8;
		x >>= 8;
	}
	if ((x & 0xF) == 0) {
		n += 4;
		x >>= 4;
	}
	if ((x & 0x3) == 0) {
		n += 2;
		x >>= 2;
	}
	if ((x & 0x1) == 0) {
		n += 1;
	}
    return n;
}
#endif

// We implement the queues asso using a floating batch of packets.  We
// allocate and recycle these batches using a pool based on a free
// list that we access using a lock-free algorithm.
// 
class batch_pool;

union batch {
private:
	batch * next;

	friend batch_pool;

public:
	packet * packets[PACKETS_BATCH_SIZE];
};

// This is a free-list allocator of batches. 
// 
class batch_pool {
private:
	static vector<batch *> pool;
	static atomic<batch *> head;

	static batch * allocate_one() {
		batch * res = new batch();
		pool.push_back(res);
		return res;
	}

public:
	static void preallocate(unsigned int n) {
		while(n-- > 0) {
			batch * b = allocate_one();
			b->next = head;
			head = b;
		}
	}

	static void clear() {
		for(vector<batch*>::iterator i = pool.begin(); i != pool.end(); ++i)
			delete(*i);
		pool.clear();
		head = 0;
	}

	static unsigned int allocated_size() {
		return pool.size();
	}

	static batch * get() {
		batch * b = head;
		do {
			if (!b) 
				return allocate_one();
		} while (!head.compare_exchange_weak(b, b->next));

		return b;
	}

	static void put(batch * b) {
		b->next = head;
		do {
		} while(!head.compare_exchange_weak(b->next, b));
	}
};

vector<batch *> batch_pool::pool;
atomic<batch *> batch_pool::head(0);

class partition_queue {
	unsigned int partition_id;
	atomic<unsigned int> tail; //one-past the last element
	batch * b;
#ifdef WITH_FRONTEND_STATISTICS
	unsigned int flush_count;
	unsigned int enqueue_count;
	high_resolution_clock::time_point first_enqueue_time;
    milliseconds max_latency;
#endif

public:
	partition_queue() : partition_id(0), tail(0), b(0)
#ifdef WITH_FRONTEND_STATISTICS
		, flush_count(0), enqueue_count(0)
#endif
		{};
	partition_queue(const partition_queue & pq) 
		: partition_id(pq.partition_id), b(pq.b)
#ifdef WITH_FRONTEND_STATISTICS
		, flush_count(pq.flush_count), enqueue_count(pq.enqueue_count), 
		  max_latency(pq.max_latency)
#endif
		{
		unsigned int tmp = pq.tail;
		tail = tmp;
	};

    void initialize(unsigned int id) {
		partition_id = id; 
		tail = 0;
		b = batch_pool::get();
#ifdef WITH_FRONTEND_STATISTICS
		flush_count = 0;
		enqueue_count = 0;
		max_latency = std::chrono::milliseconds(0);
#endif
	}

	void enqueue(packet * p);
	void flush();

#ifdef WITH_FRONTEND_STATISTICS
	unsigned int get_flush_count() const {
		return flush_count;
	}

	unsigned int get_enqueue_count() const {
		return enqueue_count;
	}

	unsigned int get_partition_id() const {
		return partition_id;
	}

	unsigned int get_max_latency_ms() const {
		return max_latency.count();
	}

	ostream & print_statistics(ostream & os) const {
		os << "part=" << partition_id
		   << " enqueue_count=" << enqueue_count
		   << " flush_count=" << flush_count
		   << " max_latency=" << max_latency.count() << "ms"
		   << std::endl;
		return os;
	}

	void update_flush_statistics() {
		flush_count += 1;
		milliseconds latency 
			= duration_cast<milliseconds>(high_resolution_clock::now() - first_enqueue_time);
		if (latency > max_latency)
			max_latency = latency;
	}
#endif
};

void partition_queue::enqueue(packet * p) {
	assert(tail <= PACKETS_BATCH_SIZE);

	unsigned int t = tail;
	
	do {
		while (t == PACKETS_BATCH_SIZE)
			t = tail;
	} while(!tail.compare_exchange_weak(t, PACKETS_BATCH_SIZE));

#ifdef WITH_FRONTEND_STATISTICS
	enqueue_count += 1;
	if (t == 0)
		first_enqueue_time = high_resolution_clock::now();
#endif
	b->packets[t] = p;
	++t;

	if (t == PACKETS_BATCH_SIZE) {
		batch * bx = b;
		b = batch_pool::get();
#ifdef WITH_FRONTEND_STATISTICS
		update_flush_statistics();
#endif
		tail = 0;
		back_end::process_batch(partition_id, bx->packets, PACKETS_BATCH_SIZE);
		batch_pool::put(bx);
	} else {
		tail = t;
	}
}

void partition_queue::flush() {
	unsigned int bx_size = tail;
	do {
		while (bx_size == PACKETS_BATCH_SIZE)
			bx_size = tail;

		if (bx_size == 0)
			return;
	} while(!tail.compare_exchange_weak(bx_size, PACKETS_BATCH_SIZE));

	batch * bx = b;
	b = batch_pool::get();
#ifdef WITH_FRONTEND_STATISTICS
	update_flush_statistics();
#endif
	tail = 0;
	back_end::process_batch(partition_id, bx->packets, bx_size);
	batch_pool::put(bx);
}

template<unsigned int Size>
class prefix_queue_pair : public partition_queue {
public:
	prefix<Size> p;

public:
	prefix_queue_pair(): partition_queue(), p() {};
	prefix_queue_pair(const prefix_queue_pair &pqp)
		: partition_queue(pqp), p(pqp.p) {};

	void initialize(unsigned int id, const block_t * pb) {
		partition_queue::initialize(id);
		p.assign(pb);
	}
};

typedef prefix_queue_pair<64> queue64;
typedef prefix_queue_pair<128> queue128;
typedef prefix_queue_pair<192> queue192;

// 
// container of prefixes whose leftmost 1-bit is in the 3rd 64-bit
// block.
// 
class p3_container {
public:
    // Since this is the third of three blocks, it may only contain
    // prefixes of up to 64 bits.
    //
	vector<queue64> p64;

    void add64(unsigned int id, const block_t * p) {
		p64.emplace_back();
		p64.back().initialize(id, p);
    }

	void clear() {
		p64.clear();
	}
#ifdef WITH_FRONTEND_STATISTICS
	ostream & print_statistics(ostream & os) {
		for(vector<queue64>::const_iterator i = p64.begin(); i != p64.end(); ++i)
			i->print_statistics(os);
		return os;
	}
#endif
};

// 
// container of prefixes whose leftmost 1-bit is in the 2nd 64-bit
// block.
// 
class p2_container : public p3_container {
public:
    // Since this is the second of three blocks, it may contain
    // prefixes of up to 64 bits (inherited from p3_container) and
    // prefixes of up to 128 bits.
    //
	vector<queue128> p128;

    void add128(unsigned int id, const block_t * p) {
		p128.emplace_back();
		p128.back().initialize(id, p);
    }

	void clear() {
		p128.clear();
		p3_container::clear();
	}

#ifdef WITH_FRONTEND_STATISTICS
	ostream & print_statistics(ostream & os) {
		for(vector<queue128>::const_iterator i = p128.begin(); i != p128.end(); ++i)
			i->print_statistics(os);
		return p3_container::print_statistics(os);
	}
#endif
};

// 
// container of prefixes whose leftmost 1-bit is in the 1st 64-bit
// block.
// 
class p1_container : public p2_container {
public:
    // Since this is the second of three blocks, it may contain
    // prefixes of up to 64 and 128 bits (inherited from p3_container
    // and p2_container) plus prefixes of up to 192 bits.
    //
	vector<queue192> p192;

    void add192(unsigned int id, const block_t * p) {
		p192.emplace_back();
		p192.back().initialize(id, p);
    }
	void clear() {
		p192.clear();
		p2_container::clear();
	}
#ifdef WITH_FRONTEND_STATISTICS
	ostream & print_statistics(ostream & os) {
		for(vector<queue192>::const_iterator i = p192.begin(); i != p192.end(); ++i)
			i->print_statistics(os);
		return p2_container::print_statistics(os);
	}
#endif
};

static p1_container pp1[64];
static p2_container pp2[64];
static p3_container pp3[64];

// This is how we compile the FIB
// 
void front_end::add_prefix(unsigned int id, const filter_t & f, unsigned int n) {
    const block_t * b = f.begin();

    if (*b) {
		if (n <= 64) {
			pp1[leftmost_bit(*b)].add64(id, b);
		} else if (n <= 128) {
			pp1[leftmost_bit(*b)].add128(id, b);
		} else {
			pp1[leftmost_bit(*b)].add192(id, b);
		}
    } else if (*(++b)) {
		if (n <= 64) {
			pp2[leftmost_bit(*b)].add64(id, b);
		} else {
			pp2[leftmost_bit(*b)].add128(id, b);
		}
    } else if (*(++b)) {
		pp3[leftmost_bit(*b)].add64(id, b);
    }
}

// This is the main matching function
// 
static void match(packet * pkt) {

	const block_t * b = pkt->filter.begin();

    if (*b) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			p1_container & c = pp1[m];

			for(vector<queue64>::iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b)) 
					i->enqueue(pkt);

			for(vector<queue128>::iterator i = c.p128.begin(); i != c.p128.end(); ++i) 
				if (i->p.subset_of(b))
					i->enqueue(pkt);

			for(vector<queue192>::iterator i = c.p192.begin(); i != c.p192.end(); ++i) 
				if (i->p.subset_of(b))
					i->enqueue(pkt);

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);
			
    } else if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			p2_container & c = pp2[m];

			for(vector<queue64>::iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b))
					i->enqueue(pkt);

			for(vector<queue128>::iterator i = c.p128.begin(); i != c.p128.end(); ++i) 
				if (i->p.subset_of(b))
					i->enqueue(pkt);

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);

    } else if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			p3_container & c = pp3[m];

			for(vector<queue64>::iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b))
					i->enqueue(pkt);

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);
    }
    pkt->frontend_done();
}

// FRONT-END EXECUTION THREADS
// 
// The front-end matcher runs with a pool of threads.
//
// We maintain a queue of jobs for the front-end threads.  There are
// two kinds of jobs.  When the front-end is in the MATCHING state,
// then the threads execute matching jobs, suspending (spinning) if
// the job queue is empty.  
// 
// When the front end is in the FLUSHING state, then the threads
// execute FLUSHING jobs, and then they terminate when the job queue is empty.
// 
// In the transition state (FINALIZE_MATCHING), the threads execute
// matching jobs until the matching job queue is empty.  When it is,
// then they transition to the flushing loop.
// 
enum front_end_state {
	FE_INITIAL = 0,
	FE_MATCHING = 1,
	FE_FINALIZE_MATCHING = 2,
	FE_FLUSHING = 3,
	FE_FINALIZE_FLUSHING = 4,
	FE_FINAL = 5,
};

static atomic<front_end_state> processing_state(FE_INITIAL);
static mutex processing_mtx;
static condition_variable processing_cv;

// We actually maintain a single queue, where we insert both matching
// and flushing jobs.  We can do that because the two phases are
// mutually exclusive.
// 
union job {
	packet * p;
	partition_queue * q;
};

static const size_t JOB_QUEUE_SIZE = 1024; // must be a power of 2 for efficiency
static job job_queue[JOB_QUEUE_SIZE];

// However, we must still maintain two separate pairs of indexes
// (head,tail), one used by the matching loop, and the other used by
// the flushing loop.

// matching job queue indexes
// 
static atomic<unsigned int> matching_job_queue_head(0);	// position of the first element 
static unsigned int matching_job_queue_tail = 0;		// one-past position of the last element

// flushing job queue indexes
// 
static atomic<unsigned int> flushing_job_queue_head(0);	// position of the first element 
static unsigned int flushing_job_queue_tail = 0;		// one-past position of the last element

static unsigned int matching_threads;

// 
// This is used to enqueue a matching job.  It suspends in a spin loop
// as long as the queue is full.  
// 
// ** WARNING: THIS IS TO BE USED BY A SINGLE THREAD. ** 
// 
void front_end::match(packet * pkt) {
	assert(processing_state == FE_INITIAL || processing_state == FE_MATCHING);
	unsigned int tail_plus_one = (matching_job_queue_tail + 1) % JOB_QUEUE_SIZE;

	while (tail_plus_one == matching_job_queue_head) {
		;		// full queue => spin
	}
	job_queue[matching_job_queue_tail].p = pkt;
	matching_job_queue_tail = tail_plus_one;
}

static void match_loop() {
	unsigned int head, head_plus_one;
	packet * p;

	for(;;) {
		head = matching_job_queue_head;
		do {
			while (head == matching_job_queue_tail) {  
				// empty queue => spin
				if (processing_state == FE_MATCHING) {
					// if we are still matching, then we loop
					head = matching_job_queue_head;
					continue;		  
				} else {
					// if we are stopped, then we switch to flushing
					unique_lock<mutex> lock(processing_mtx);
					if (processing_state == FE_FINALIZE_MATCHING) {
						if (--matching_threads == 0) {
							processing_state = FE_FLUSHING;
							processing_cv.notify_all();
						}
					}
					return;
				}
			}
			p = job_queue[head].p;
			head_plus_one = (head + 1) % JOB_QUEUE_SIZE;
		} while (!matching_job_queue_head.compare_exchange_weak(head, head_plus_one));
		match(p);
	}
}

static void flush_loop() {
	unsigned int head, head_plus_one;
	partition_queue * q;

	for(;;) {
		head = flushing_job_queue_head;
		do {
			while (head == flushing_job_queue_tail) {
				// empty queue => spin
				if (processing_state == FE_FLUSHING) {
					// if we are still flushing, then we loop
					head = flushing_job_queue_head;
					continue;		  
				} else {
					// if we are done, then we terminate
					unique_lock<mutex> lock(processing_mtx);
					if (processing_state == FE_FINALIZE_FLUSHING) {
						processing_state = FE_FINAL;
						processing_cv.notify_all();
					}
					return;
				}
			}
			q = job_queue[head].q;
			head_plus_one = (head + 1) % JOB_QUEUE_SIZE;
		} while (!flushing_job_queue_head.compare_exchange_weak(head, head_plus_one));
		q->flush();
	}
}

// this simply enqueues a packet in the queue of matching jobs.
//
// WARNING: THIS IS TO BE USED BY A SINGLE THREAD.
// 
static void enqueue_flush_job(partition_queue * q) {
	unsigned int tail_plus_one = (flushing_job_queue_tail + 1) % JOB_QUEUE_SIZE;

	while (tail_plus_one == flushing_job_queue_head) {
		// full queue => spin
		;
	}
	job_queue[flushing_job_queue_tail].q = q;
	flushing_job_queue_tail = tail_plus_one;
	
}

static void processing_thread() {
	match_loop();
	flush_loop();
}

static vector<thread *> thread_pool;

void front_end::start(unsigned int threads) {
	if (processing_state == FE_INITIAL && threads > 0) {
		processing_state = FE_MATCHING;
		matching_threads = threads;
		do {
			thread_pool.push_back(new thread(processing_thread));
		} while(--threads > 0);
	}
}

void front_end::stop() {
	do {
		unique_lock<mutex> lock(processing_mtx);
		processing_state = FE_FINALIZE_MATCHING;
		while(processing_state != FE_FLUSHING)
			processing_cv.wait(lock);
	} while(0);

	for(unsigned int j = 0; j < 64; ++j) {
		for(vector<queue64>::iterator i = pp1[j].p64.begin(); i != pp1[j].p64.end(); ++i)
			enqueue_flush_job(&(*i));

		for(vector<queue128>::iterator i = pp1[j].p128.begin(); i != pp1[j].p128.end(); ++i) 
			enqueue_flush_job(&(*i));

		for(vector<queue192>::iterator i = pp1[j].p192.begin(); i != pp1[j].p192.end(); ++i) 
			enqueue_flush_job(&(*i));

		for(vector<queue64>::iterator i = pp2[j].p64.begin(); i != pp2[j].p64.end(); ++i) 
			enqueue_flush_job(&(*i));

		for(vector<queue128>::iterator i = pp2[j].p128.begin(); i != pp2[j].p128.end(); ++i) 
			enqueue_flush_job(&(*i));

		for(vector<queue64>::iterator i = pp3[j].p64.begin(); i != pp3[j].p64.end(); ++i) 
			enqueue_flush_job(&(*i));
	}
	do {
		unique_lock<mutex> lock(processing_mtx);
		processing_state = FE_FINALIZE_FLUSHING;
		while(processing_state != FE_FINAL)
			processing_cv.wait(lock);
	} while(0);

	for(vector<thread *>::iterator i = thread_pool.begin(); i != thread_pool.end(); ++i)
		(*i)->join();

	thread_pool.clear();

	processing_state = FE_INITIAL;;
}

void front_end::clear() {
	for(int i = 0; i < 64; ++i) {
		pp1[i].clear();
		pp2[i].clear();
		pp3[i].clear();
	}
	batch_pool::clear();
}

#ifdef WITH_FRONTEND_STATISTICS
ostream & front_end::print_statistics(ostream & os) {
	for(int i = 0; i < 64; ++i) {
		pp1[i].print_statistics(os);
		pp2[i].print_statistics(os);
		pp3[i].print_statistics(os);
	}
	return os;
}
#endif
