#ifdef HAVE_CONFIG_H
#include "config.h"
#else
#define HAVE_BUILTIN_CTZL
#endif
#include <iostream>
#include <climits>
#include <cstdlib>
#include <cstdio>				// sscanf
#include <cstring>				// strcmp
#include <vector>
#include <cstdint>
#include <chrono>
#include <cassert>
#include <thread>
#ifdef WITH_MUTEX_THREADPOOL
#include <mutex>
#include <condition_variable>
#endif
#ifndef WITH_BUILTIN_CAS
#include <atomic>
#endif

using namespace std;
using namespace std::chrono;

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
// We use blocks of 64 bits...
//
typedef uint64_t block_t;
static_assert(sizeof(block_t)*CHAR_BIT == 64, "uint64_t must be a 64-bit word");
static const int BLOCK_SIZE = 64;
static const block_t BLOCK_ONE = 0x1;

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

//
// Main representation of a prefix.  Essentially we will instantiate
// this template with Size=64, Size=128, and Size=192.
//
template <unsigned int Size>
class prefix {
    static_assert((Size % 64) == 0, "prefix width must be a multiple of 64");
    static const int BLOCK_COUNT = Size / BLOCK_SIZE;
    // 
    // BIT LAYOUT: a prefix is represented with the bit pattern in
    // reverse order.  That is, the first bit is the least significant
    // bit, and the pattern goes from left-to-right from the least
    // significant bit towards the most significant bit.  Notice that
    // we do not store the length of a prefix.  So, a prefix of one
    // bit will still be stored as a 64-bit quantity with all the
    // trailing bits set to 0.
    // 
    // EXAMPLE:
    // prefix "000101" is represented by the three blocks:
    // b[0] = (101000)binary, b[1] = 0, b[2] = 0
    //
    block_t b[BLOCK_COUNT];
    
public:
    const block_t * begin() const {
		return b;
    }

    const block_t * end() const {
		return b + BLOCK_COUNT;
    }

    bool subset_of(const block_t * p) const {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			if ((b[i] & ~p[i]) != 0)
				return false;

		return true;
    }

    prefix(const string & p) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = 0;

		assert(p.size() <= Size);

		// see the layout specification above
		//
		block_t mask = BLOCK_ONE;
		int i = 0;
		for(string::const_iterator c = p.begin(); c != p.end(); ++c) {
			if (*c == '1')
				b[i] |= mask;

			mask <<= 1;
			if (mask == 0) {
				mask = BLOCK_ONE;
				if (++i == BLOCK_COUNT)
					return;
			}
		}
    }

    prefix(const block_t * p) {
		for (int i = 0; i < BLOCK_COUNT; ++i)
			b[i] = p[i];
    }
};

typedef prefix<192> filter_t;

class queue {
public:

	static const unsigned int MAX_SIZE = 1024;

#ifdef WITH_BUILTIN_CAS
	volatile unsigned int tail;
#else
	std::atomic<unsigned int> tail; // one-past the last element
#endif
	unsigned int q[MAX_SIZE]; 
	queue(): tail(0) {}; 

	void enqueue (unsigned int n) {
		unsigned int old_t;
		unsigned int new_t;

#ifdef WITH_BUILTIN_CAS

	try_push:
		old_t = tail;
		new_t = old_t+1;

		if (new_t > MAX_SIZE)
			goto try_push;

		if (!__sync_bool_compare_and_swap (&tail, old_t, new_t))
			goto try_push;
#else
		old_t = tail.load(std::memory_order_acquire);

	try_push:
		new_t = old_t+1;

		if (new_t > MAX_SIZE) {
			// queue full => busy loop
			old_t = tail.load(std::memory_order_acquire);
			goto try_push;
		}

		if (!std::atomic_compare_exchange_weak(&tail, &old_t, new_t))
			goto try_push;
#endif

		q[old_t]=n;

		if (new_t == MAX_SIZE) {
			flush();
		}
	}

	void flush () {
		//send stuff to gpu!
		//is it better to have two queues? 
		//in this way you can copy one queue to the gpu
		//and use the other one for the prefiltering
#ifdef WITH_BUILTIN_CAS
		tail = 0;
#else
		tail.store(0, std::memory_order_release);
#endif
	}
};

template<unsigned int Size>
class prefix_queue_pair {
public:
	const prefix<Size> p;
	queue * q;

    prefix_queue_pair(const block_t * pb, queue * qq): p(pb), q(qq) {};
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

    void add64(const block_t * p, queue * q) {
		p64.push_back(queue64(p,q));
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

    void add128(const block_t * p, queue * q) {
		p128.push_back(queue128(p,q));
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

    void add192(const block_t * p, queue * q) {
		p192.push_back(queue192(p,q));
    }
};

static p1_container pp1[64];
static p2_container pp2[64];
static p3_container pp3[64];

void fib_add_prefix(const filter_t & f, unsigned int n, queue * q) {
    const block_t * b = f.begin();

    if (*b) {
		if (n <= 64) {
			pp1[leftmost_bit(*b)].add64(b,q);
		} else if (n <= 128) {
			pp1[leftmost_bit(*b)].add128(b,q);
		} else {
			pp1[leftmost_bit(*b)].add192(b,q);
		}
    } else if (*(++b)) {
		if (n <= 64) {
			pp2[leftmost_bit(*b)].add64(b,q);
		} else {
			pp2[leftmost_bit(*b)].add128(b,q);
		}
    } else if (*(++b)) {
		pp3[leftmost_bit(*b)].add64(b,q);
    }
}

void fib_match(const filter_t * q) {
    const block_t * b = q->begin();

    if (*b) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p1_container & c = pp1[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b))
					i->q->enqueue(1);

			for(vector<queue128>::const_iterator i = c.p128.begin(); i != c.p128.end(); ++i) 
				if (i->p.subset_of(b))
					i->q->enqueue(1);

			for(vector<queue192>::const_iterator i = c.p192.begin(); i != c.p192.end(); ++i) 
				if (i->p.subset_of(b))
					i->q->enqueue(1);

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);
			
    } else if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p2_container & c = pp2[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b))
					i->q->enqueue(1);

			for(vector<queue128>::const_iterator i = c.p128.begin(); i != c.p128.end(); ++i) 
				if (i->p.subset_of(b))
					i->q->enqueue(1);

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);

    } else if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p3_container & c = pp3[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b))
					i->q->enqueue(1);

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);
    }
}

static unsigned long total_matches() {
    unsigned long c = 0;
    for(int j = 0; j< 64; j++){
        for(vector<queue64>::const_iterator i = pp1[j].p64.begin(); i != pp1[j].p64.end(); ++i){ 
            c+=i->q->tail;
        }
        for(vector<queue128>::const_iterator i = pp1[j].p128.begin(); i != pp1[j].p128.end(); ++i) 
            c+=i->q->tail;
        for(vector<queue192>::const_iterator i = pp1[j].p192.begin(); i != pp1[j].p192.end(); ++i) 
            c+=i->q->tail;
        for(vector<queue64>::const_iterator i = pp2[j].p64.begin(); i != pp2[j].p64.end(); ++i) 
            c+=i->q->tail;
        for(vector<queue128>::const_iterator i = pp2[j].p128.begin(); i != pp2[j].p128.end(); ++i) 
            c+=i->q->tail;
        for(vector<queue64>::const_iterator i = pp3[j].p64.begin(); i != pp3[j].p64.end(); ++i) 
            c+=i->q->tail;
    }
    return c;
}

static void destroy_fib() {
    for(int j = 0; j< 64; j++){
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

static const size_t JOB_QUEUE_SIZE = 1024; // must be a power of 2 for efficiency
static const filter_t * job_queue[JOB_QUEUE_SIZE];

#ifdef WITH_MUTEX_THREADPOOL

static size_t job_queue_head = 0;		// position of the first element in the queue
static size_t job_queue_tail = 0;		// one-past position of the last element in the queue

static std::mutex job_queue_mtx;
static std::condition_variable job_queue_producer_cv;
static std::condition_variable job_queue_consumers_cv;

static void match_job_enqueue(const filter_t * f) {
	size_t tail_plus_one;
	std::unique_lock<std::mutex> lock(job_queue_mtx);
 try_enqueue:
	tail_plus_one = (job_queue_tail + 1) % JOB_QUEUE_SIZE;

	if (tail_plus_one == job_queue_head) { // full queue 
		job_queue_producer_cv.wait(lock);
		goto try_enqueue;
	}
	job_queue[job_queue_tail] = f;
	job_queue_tail = tail_plus_one;

	job_queue_consumers_cv.notify_all();
}

static const filter_t * match_job_dequeue() {
	std::unique_lock<std::mutex> lock(job_queue_mtx);

 try_dequeue:
	if (job_queue_head == job_queue_tail) { // empty queue 
		job_queue_consumers_cv.wait(lock);
		goto try_dequeue;
	}

    const filter_t * f = job_queue[job_queue_head];
	job_queue_head = (job_queue_head + 1) % JOB_QUEUE_SIZE;

	job_queue_producer_cv.notify_one();
	return f;
}

#else

#ifdef WITH_BUILTIN_CAS

static volatile unsigned int job_queue_head = 0;	// position of the first element 
static volatile unsigned int job_queue_tail = 0;	// one-past position of the last element

static void match_job_enqueue(const filter_t * f) {
	unsigned int tail_plus_one;
	unsigned int my_tail;

	my_tail = job_queue_tail;
	tail_plus_one = (my_tail + 1) % JOB_QUEUE_SIZE;

	while (tail_plus_one == job_queue_head) {
		// full queue => busy loop
		;
	}

	job_queue[my_tail] = f;
	job_queue_tail = tail_plus_one;
}

static const filter_t * match_job_dequeue() {
	unsigned int my_head, head_plus_one;

 try_dequeue:
	my_head = job_queue_head;

	if (my_head == job_queue_tail)
		goto try_dequeue;		   // empty queue => busy loop

	head_plus_one = (my_head + 1) % JOB_QUEUE_SIZE;

	const filter_t * result = job_queue[my_head];

	if (!__sync_bool_compare_and_swap(&job_queue_head, my_head, head_plus_one))
		goto try_dequeue;

	return result;
}

#else

static std::atomic<unsigned int> job_queue_head(0);	// position of the first element 
static std::atomic<unsigned int> job_queue_tail(0);	// one-past position of the last element

static void match_job_enqueue(const filter_t * f) {
	unsigned int tail_plus_one;
	unsigned int my_tail;

	my_tail = job_queue_tail.load(std::memory_order_acquire);
	tail_plus_one = (my_tail + 1) % JOB_QUEUE_SIZE;

	while (tail_plus_one == job_queue_head.load(std::memory_order_acquire)) {
		// full queue => busy loop
		;
	}

	job_queue[my_tail] = f;
	job_queue_tail.store(tail_plus_one, std::memory_order_release);
}

static const filter_t * match_job_dequeue() {
	unsigned int my_head, head_plus_one;

	my_head = job_queue_head.load(std::memory_order_acquire);
 try_dequeue:

	if (my_head == job_queue_tail.load(std::memory_order_acquire))
		goto try_dequeue;		   // empty queue => busy loop

	head_plus_one = (my_head + 1) % JOB_QUEUE_SIZE;

	const filter_t * result = job_queue[my_head];

	if (!std::atomic_compare_exchange_weak(&job_queue_head, &my_head, head_plus_one))
		goto try_dequeue;

	return result;
}
#endif // HAVE_BUILTIN_CAS
#endif

#ifndef THREAD_COUNT
#define THREAD_COUNT 4
#endif

std::thread * thread_pool[THREAD_COUNT];

void thread_loop() {
	const filter_t * f;
	while((f = match_job_dequeue()))
		fib_match(f);
}

int main(int argc, const char * argv[]) {
    unsigned int N = 1;			// how many cycles throug the queries?

	for(int i = 1; i < argc; ++i) {
		if (sscanf(argv[i], "n=%u", &N) || sscanf(argv[i], "N=%u", &N))
			continue;

		if (strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0) {
			std::cout << "usage: " << argv[0] << " [n=<number-of-rounds>]\n"
					  << std::endl;
			return 1;
		}
	}

    vector<filter_t> queries;	// we store all the queries here
    string command, filter_string;

    while(std::cin >> command >> filter_string) {
		if (command == "+") {
			filter_t f(filter_string);
			unsigned int n = filter_string.size();
			fib_add_prefix(f,n,new queue());
		} else if (command=="!") {
			filter_t f(filter_string);
			queries.push_back(f);
		} else if (command == "match") {
			break;
		}
    }

	for(size_t i = 0; i < THREAD_COUNT; ++i)
		thread_pool[i] = new thread(thread_loop);

    high_resolution_clock::time_point start = high_resolution_clock::now();

	for(unsigned int round = 0; round < N; ++round) 
		for(vector<filter_t>::const_iterator i = queries.begin(); i != queries.end(); ++i)
			match_job_enqueue(&(*i));

	for(size_t i = 0; i < THREAD_COUNT; ++i)
		match_job_enqueue(0);

	for(size_t i = 0; i < THREAD_COUNT; ++i)
		thread_pool[i]->join();

    high_resolution_clock::time_point stop = high_resolution_clock::now();

	for(size_t i = 0; i < THREAD_COUNT; ++i)
		delete(thread_pool[i]);

    nanoseconds ns = duration_cast<nanoseconds>(stop - start);
    cout << "Average matching time: " << ns.count()/queries.size()/N
		 << "ns" << endl
		 << "queries: " << queries.size() << endl
		 << "total matches: " << total_matches() << endl;

	destroy_fib();
    return 0;
}
