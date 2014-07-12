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

using std::atomic;
using std::vector;

#include "parameters.hh"

#include "packet.hh"
#include "front_end.hh"
#include "back_end.hh"
#include "free_list.hh"

// ***DESIGN OF THE FORWARDING TABLE***
//
// The FIB maps a filter prefix to a queue.
// 
// FIB: prefix -> queue
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

union batch;

typedef free_list_static_spinning_allocator<batch> batch_allocator_t;

union batch {
private:
	batch * next;

	friend batch_allocator_t;

public:
	packet * packets[PACKETS_BATCH_SIZE];
};

static batch_allocator_t batch_allocator;

class queue {
	unsigned int partition_id;
	unsigned int tail; //one-past the last element
	batch * b;
	std::mutex mtx;

public:
    queue(unsigned int id)
		: partition_id(id), 
		  tail(0), 
		  b(batch_allocator.allocate())
		{ };

	void enqueue(packet * p);
	void flush();

	~queue() { batch_allocator.recycle(b); }
};

void queue::enqueue(packet * p) {
	assert(tail < PACKETS_BATCH_SIZE);
	mtx.lock();
	b->packets[tail++] = p;
	if (tail == PACKETS_BATCH_SIZE) {
		batch * bx = b;
		b = batch_allocator.allocate();
		tail = 0;
		mtx.unlock();
		back_end::process_batch(partition_id, bx->packets, PACKETS_BATCH_SIZE);
		batch_allocator.recycle(bx);
	} else {
		mtx.unlock();
	}
}

void queue::flush() {
	mtx.lock();
	batch * bx = b;
	unsigned int bx_size = tail;
	b = batch_allocator.allocate();
	tail = 0;
	mtx.unlock();
	back_end::process_batch(partition_id, bx->packets, bx_size);
	batch_allocator.recycle(bx);
}

template<unsigned int Size>
class prefix_queue_pair {
public:
	const prefix<Size> p;
	queue * q;

	prefix_queue_pair(unsigned int id, const block_t * pb): p(pb), q(new queue(id)) {};
};

typedef prefix_queue_pair<64> queue64;
typedef prefix_queue_pair<128> queue128;
typedef prefix_queue_pair<192> queue192;

// 
// container of prefixes whose leftmost bit is the 3rd 64-bit block.
// 
class p3_container {
public:
    // Since this is the third of three blocks, it may only contain
    // prefixes of up to 64 bits.
    //
    vector<queue64> p64;

    void add64(unsigned int id, const block_t * p) {
		p64.push_back(queue64(id, p));
    }
};

// 
// container of prefixes whose leftmost bit is the 2nd 64-bit block.
// 
class p2_container : public p3_container {
public:
    // Since this is the second of three blocks, it may contain
    // prefixes of up to 64 bits (inherited from p3_container) and
    // prefixes of up to 128 bits.
    //
    vector<queue128> p128;

    void add128(unsigned int id, const block_t * p) {
		p128.push_back(queue128(id, p));
    }
};

// 
// container of prefixes whose leftmost bit is the 1st 64-bit block.
// 
class p1_container : public p2_container {
public:
    // Since this is the second of three blocks, it may contain
    // prefixes of up to 64 and 128 bits (inherited from p3_container
    // and p2_container) plus prefixes of up to 192 bits.
    //
    vector<queue192> p192;

    void add192(unsigned int id, const block_t * p) {
		p192.push_back(queue192(id, p));
    }
};

static p1_container pp1[64];
static p2_container pp2[64];
static p3_container pp3[64];

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

static void match(packet * pkt) {

	const block_t * b = pkt->filter.begin();

    if (*b) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p1_container & c = pp1[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b)) 
					i->q->enqueue(pkt);

			for(vector<queue128>::const_iterator i = c.p128.begin(); i != c.p128.end(); ++i) 
				if (i->p.subset_of(b))
					i->q->enqueue(pkt);

			for(vector<queue192>::const_iterator i = c.p192.begin(); i != c.p192.end(); ++i) 
				if (i->p.subset_of(b))
					i->q->enqueue(pkt);

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);
			
    } else if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p2_container & c = pp2[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b))
					i->q->enqueue(pkt);

			for(vector<queue128>::const_iterator i = c.p128.begin(); i != c.p128.end(); ++i) 
				if (i->p.subset_of(b))
					i->q->enqueue(pkt);

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);

    } else if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p3_container & c = pp3[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b))
					i->q->enqueue(pkt);

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);
    }
    pkt->frontend_done();
}

void front_end::clear() {
    for(int j = 0; j< 64; j++) {
        for(vector<queue64>::const_iterator i = pp1[j].p64.begin(); i != pp1[j].p64.end(); ++i)
            if (i->q)
				delete(i->q);
		pp1[j].p64.clear();

        for(vector<queue128>::const_iterator i = pp1[j].p128.begin(); i != pp1[j].p128.end(); ++i) 
            if (i->q)
				delete(i->q);
		pp1[j].p128.clear();

        for(vector<queue192>::const_iterator i = pp1[j].p192.begin(); i != pp1[j].p192.end(); ++i) 
            if (i->q)
				delete(i->q);
		pp1[j].p192.clear();

        for(vector<queue64>::const_iterator i = pp2[j].p64.begin(); i != pp2[j].p64.end(); ++i) 
            if (i->q)
				delete(i->q);
		pp2[j].p64.clear();

        for(vector<queue128>::const_iterator i = pp2[j].p128.begin(); i != pp2[j].p128.end(); ++i) 
            if (i->q)
				delete(i->q);
		pp2[j].p128.clear();

        for(vector<queue64>::const_iterator i = pp3[j].p64.begin(); i != pp3[j].p64.end(); ++i) 
            if (i->q)
				delete(i->q);
		pp3[j].p64.clear();

    }
}

static mutex flush_mtx;

/*******************************************************************************/
/******************************THREAD POOL**************************************/
/*******************************************************************************/

union job {
	packet * p;
	queue * q;
};

static const size_t JOB_QUEUE_SIZE = 1024; // must be a power of 2 for efficiency
static job job_queue[JOB_QUEUE_SIZE];

static volatile unsigned int job_queue_head = 0;	// position of the first element 
static volatile unsigned int job_queue_tail = 0;	// one-past position of the last element

enum front_end_state {
	FE_INITIAL = 0,
	FE_MATCHING = 1,
	FE_FINALIZE_MATCHING = 2,
	FE_FLUSHING = 3
};

static atomic<front_end_state> processing_state(FE_INITIAL);
static mutex processing_mtx;
static condition_variable processing_cv;

// This is used to enqueue a job.  It suspends in a busy loop if the
// queue is full.  Otherwise, it returns an available position and
// shifts the tail of the queue.
// 
// WARNING: This is INTENDED TO BE USED BY A SINGLE THREAD.
// 
static unsigned int get_enqueuing_pos() {
	unsigned int tail_plus_one;
	unsigned int tail;

	tail = job_queue_tail;
	tail_plus_one = (tail + 1) % JOB_QUEUE_SIZE;

	while (tail_plus_one == job_queue_head) {
		// full queue => busy loop
		;
	}
	job_queue_tail = tail_plus_one;
	return tail;
}

// this simply enqueues a packet in the queue of matching jobs.
// 
void front_end::match(packet * pkt) {
//	assert(front_end_state == FE_INITIAL || front_end_state == FE_MATCHING);
	job_queue[get_enqueuing_pos()].p = pkt;
}

static void match_loop() {
	assert(processing_state == FE_MATCHING);

	unsigned int head, head_plus_one;

	for(;;) { 				   
		head = job_queue_head;

		if (head == job_queue_tail) {			
			if (processing_state == FE_MATCHING) {
				continue;		   // empty queue => busy loop
			} else {
				unique_lock<std::mutex> lock(processing_mtx);
				processing_state = FE_FLUSHING;
				processing_cv.notify_all();
				return;
			}
		}
		head_plus_one = (head + 1) % JOB_QUEUE_SIZE;

		if (__sync_bool_compare_and_swap(&job_queue_head, head, head_plus_one))
			match(job_queue[head].p);
	}
}

static void flush_loop() {
	unsigned int head, head_plus_one;

	for(;;) { 				   
		head = job_queue_head;

		if (head == job_queue_tail)
			return;

		head_plus_one = (head + 1) % JOB_QUEUE_SIZE;

		if (__sync_bool_compare_and_swap(&job_queue_head, head, head_plus_one))
			job_queue[head].q->flush();
	}
}

static void processing_thread() {
	match_loop();
	flush_loop();
}

static std::vector<std::thread *> thread_pool;

void front_end::start(unsigned int threads) {
	if (processing_state == FE_INITIAL && threads > 0) {
		processing_state = FE_MATCHING;

		do {
			thread_pool.push_back(new std::thread(processing_thread));
		} while(--threads > 0);
	}
}

// this simply enqueues a packet in the queue of matching jobs.
// 
static void enqueue_flush_job(queue * q) {
	job_queue[get_enqueuing_pos()].q = q;
}

void front_end::shutdown() {
	do {
		unique_lock<std::mutex> lock(processing_mtx);
		processing_state = FE_FINALIZE_MATCHING;
		while(processing_state != FE_FLUSHING)
			processing_cv.wait(lock);
	} while(0);

	for(unsigned int j = 0; j < 64; ++j) {
		for(vector<queue64>::const_iterator i = pp1[j].p64.begin(); i != pp1[j].p64.end(); ++i)
			if (i->q) 
				enqueue_flush_job(i->q);

		for(vector<queue128>::const_iterator i = pp1[j].p128.begin(); i != pp1[j].p128.end(); ++i) 
			if (i->q)
				enqueue_flush_job(i->q);

		for(vector<queue192>::const_iterator i = pp1[j].p192.begin(); i != pp1[j].p192.end(); ++i) 
			if (i->q)
				enqueue_flush_job(i->q);

		for(vector<queue64>::const_iterator i = pp2[j].p64.begin(); i != pp2[j].p64.end(); ++i) 
			if (i->q)
				enqueue_flush_job(i->q);

		for(vector<queue128>::const_iterator i = pp2[j].p128.begin(); i != pp2[j].p128.end(); ++i) 
			if (i->q)
				enqueue_flush_job(i->q);

		for(vector<queue64>::const_iterator i = pp3[j].p64.begin(); i != pp3[j].p64.end(); ++i) 
			if (i->q)
				enqueue_flush_job(i->q);
	}
	for(std::vector<std::thread *>::iterator i = thread_pool.begin(); i != thread_pool.end(); ++i)
		(*i)->join();

	thread_pool.clear();

	processing_state = FE_INITIAL;;
}

