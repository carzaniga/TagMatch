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
#include <climits>

#include <iomanip>
#include <iostream>

#include "parameters.hh"
#include "front_end.hh"
#include "fib.hh"
#include "filter.hh"
#include "tagmatch.hh"
#include "tagmatch_query.hh"
#include "gpu.hh"
#include "back_end.hh"

using std::vector;
using std::thread;
using std::atomic;
using std::mutex;
using std::unique_lock;
using std::condition_variable;

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_values;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::ostream;


// ***DESIGN OF THE FORWARDING TABLE***
//
// The FIB maps a filter prefix to a queue.
//
// FIB: prefix -> queue
//
// The queues are used to batch queries that match the corresponding
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
// We then further split each vector as follows: pp1 contains
// prefixes of total lenght <= 64; pp1.p128 contains prefixes of total
// lenght between 64 and 128; pp1.p192 contains prefixes of total
// lenght > 128; similarly, pp2 and pp3 contains prefixes of
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

// We implement the partition queues using a floating batch of
// queries.  The queues remain associated with their partition, but
// the query batches (buffers) are independent object that we
// allocate and then recycle using a pool based on a free list that we
// access using a lock-free algorithm.
//
class batch_pool;

union batch {
public:
	struct {
		unsigned int bsize;
		tagmatch_query * queries[QUERIES_BATCH_SIZE];
	};
private:
	batch * next;

	friend batch_pool;
};

// A free-list allocator of query batches.
//
class batch_pool {
private:
	// we keep track of all the batch objects we allocate by storing
	// them (pointers) in the pool vector.  We then use that vector
	// exclusively to deallocate the batches.  Access to the pool is
	// synchronized (mutex).
	static std::mutex pool_mtx;
	static vector<batch *> pool;

	// once the batch objects are allocated, we either give them to
	// threads that use them, or we hold them in a singly-linked
	// free-list.
	static atomic<batch *> head;

public:
	// Deallocate all allocated batches, regardless of whether they
	// are in the free-list or not.  This means that clear() must be
	// called when none of the batches is in use.
	//
	static void clear() {
		std::lock_guard<std::mutex> lock(pool_mtx);
		head = nullptr;
		for(batch * i : pool)
			delete(i);
		pool.clear();
	}

	static batch * get() {
		batch * b = head.load(std::memory_order_acquire);
		do {
			if (!b) {
				b = new batch();
				if (b) {
					std::lock_guard<std::mutex> lock(pool_mtx);
					pool.push_back(b);
				}
				return b;
			}
		} while (!head.compare_exchange_weak(b, b->next));

		return b;
	}

	static void recycle(batch * b) noexcept {
		b->next = head.load(std::memory_order_acquire);
		do {
		} while(!head.compare_exchange_weak(b->next, b));
	}
};

mutex batch_pool::pool_mtx;
vector<batch *> batch_pool::pool;
atomic<batch *> batch_pool::head(nullptr);

// This class implements a central component of the front end, namely
// the queue of messages associated with each partition and used to
// buffer queries between the front end and the back end.
//
class partition_queue {
	// primary information:
	//
	partition_id_t partition_id;
	atomic<unsigned int> tail;		// one-past the last element
	atomic<unsigned int> written;	// number of elements actually written in the queue
	batch * b;						// query buffer

	// statistics and timing information
	//
	high_resolution_clock::time_point first_enqueue_time;
#ifdef WITH_FRONTEND_STATISTICS
    milliseconds max_latency;
	unsigned int flush_count;
	unsigned int enqueue_count;
#endif

	// We also maintain a global list of pending queues.  These are
	// queues with at least one element, that therefore need to be
	// flushed to the back-end at some point.  We add each queue to
	// the back of the list when we enqueue the first query, and we
	// remove the queue from the list whenever we flush the queue.  We
	// use these two pointers, plus a static (global) "sentinel" to
	// implement the list in the most efficient and simple way.
	//
	partition_queue * prev_pending;
	partition_queue * next_pending;
	static partition_queue pending_list; // list sentinel
	static mutex pending_list_mtx;		 // mutex to access list

	void add_to_pending_list() noexcept {
		// append this queue to the pending list
		std::lock_guard<std::mutex> lock(pending_list_mtx);
		next_pending = &pending_list;
		prev_pending = pending_list.prev_pending;
		pending_list.prev_pending->next_pending = this;
		pending_list.prev_pending = this;
	}

	void remove_from_pending_list() noexcept {
		std::lock_guard<std::mutex> lock(pending_list_mtx);
		next_pending->prev_pending = prev_pending;
		prev_pending->next_pending = next_pending;
	}

	void do_flush(unsigned int size) noexcept;

public:
	partition_queue() noexcept
		: partition_id(0), tail(0), written(0), b(nullptr),
#ifdef WITH_FRONTEND_STATISTICS
		  max_latency(std::chrono::milliseconds(0)),
		  flush_count(0), enqueue_count(0),
#endif
		  prev_pending(this), next_pending(this)
		{};

    void initialize(unsigned int id) noexcept {
		partition_id = id;
		tail = 0;
		written = 0;
		if (!b)
			b = batch_pool::get();

#ifdef WITH_FRONTEND_STATISTICS
		max_latency = std::chrono::milliseconds(0);
		flush_count = 0;
		enqueue_count = 0;
#endif
	}

	~partition_queue() noexcept {
		if (b) {
			batch_pool::recycle(b);
		}
	}

	void enqueue(tagmatch_query * p) noexcept;
	void flush() noexcept;

	// Get the first pending queue whose current latency is greater
	// than the given limit.  Returns false (nullptr) if no such
	// pending queue is found.
	//
	static partition_queue * first_pending(milliseconds latency_limit) noexcept {
		// we don't care about synchronizing this operation, since it is
		// the flush operation on the returned partition queue that really
		// matters, and that actually removes that partition queue from
		// the pending list.
		//
		// So, two threads might grab the same pending queue here, but
		// then only one will go into do_flush().
		//
		partition_queue * q = pending_list.next_pending;
		if (q == &pending_list)
			return nullptr;

		milliseconds current_latency
			= duration_cast<milliseconds>(high_resolution_clock::now() - q->first_enqueue_time);
		if (current_latency < latency_limit)
			return nullptr;
		return q;
	}

#ifdef WITH_FRONTEND_STATISTICS
	unsigned int get_max_latency_ms() const noexcept {
		return max_latency.count();
	}

	unsigned int get_flush_count() const noexcept {
		return flush_count;
	}

	unsigned int get_enqueue_count() const noexcept {
		return enqueue_count;
	}
#endif
	unsigned int get_partition_id() const noexcept {
		return partition_id;
	}

	ostream & print_statistics(ostream & os) const {
		os << std::setw(9) << partition_id
#ifdef WITH_FRONTEND_STATISTICS
		<< " " << std::setw(11) << max_latency.count()
		<< " " << std::setw(13) << enqueue_count
		<< " " << std::setw(11) << flush_count
#endif
		<< std::endl;
		return os;
	}

	high_resolution_clock::time_point get_first_enqueue_time() noexcept {
		return first_enqueue_time;
	}

#ifdef WITH_FRONTEND_STATISTICS
	void update_flush_statistics() {
		flush_count += 1;

		milliseconds latency
			= duration_cast<milliseconds>(high_resolution_clock::now() - first_enqueue_time);
		if (latency > max_latency)
			max_latency = latency;
	}
#endif
};

static void finalize_batch(batch * bx) {
	for (unsigned int r = 0; r < bx->bsize; r++) {
		tagmatch_query * q = bx->queries[r];
		q->partition_done();
		q->finalize_matching();
	}
	batch_pool::recycle(bx) ;
}

partition_queue partition_queue::pending_list;
mutex partition_queue::pending_list_mtx;

void partition_queue::do_flush(unsigned int size) noexcept {
	while(written.load(std::memory_order_relaxed) < size)
		; // we loop waiting until every thread that acquired some
	      // (good) position in the queue, even earlier position than
	      // this one, actually completes the writing in the queue
		  //
	      // The following fence synchronizes with the atomic
	      // increments in enqueue(), and therefore guarantees that
	      // this thread will see all values written into the batch
	      // (non-atomic).
	std::atomic_thread_fence(std::memory_order_acquire);
	batch * bx = b;
	b = batch_pool::get();
#ifdef WITH_FRONTEND_STATISTICS
	update_flush_statistics();
#endif
	remove_from_pending_list();
	written.store(0, std::memory_order_release);
	tail.store(0, std::memory_order_release);
	bx->bsize = size;
	bx = back_end::process_batch(partition_id, bx->queries, size, bx);
	if(bx)
		finalize_batch(bx);
}

void partition_queue::enqueue(tagmatch_query * q) noexcept {
	assert(tail <= QUERIES_BATCH_SIZE);
	q->partition_enqueue();
	unsigned int t = tail.load(std::memory_order_acquire);
	do {
		while (t == QUERIES_BATCH_SIZE)
			t = tail.load(std::memory_order_acquire);
	} while(!tail.compare_exchange_weak(t, t + 1));

#ifdef WITH_FRONTEND_STATISTICS
	enqueue_count += 1;
#endif
	b->queries[t] = q;
	// The following atomic (incremet) release on the "written"
	// counter is intended to synchonize with the atomic acquire fence
	// in do_flush.  This guarantees that all queries written in the
	// batch in the line above are indeed read by the flushing thread.
	written.fetch_add(1, std::memory_order_release);
	if (t + 1 == QUERIES_BATCH_SIZE) {
		do_flush(QUERIES_BATCH_SIZE);
	} else  {
		if (t == 0) {
			first_enqueue_time = high_resolution_clock::now();
			add_to_pending_list();
		}
	}
}

void partition_queue::flush() noexcept {
	unsigned int bx_size = tail.load(std::memory_order_acquire);
	do {
		if (bx_size == QUERIES_BATCH_SIZE)
			return;

		if (bx_size == 0)
			return;
	} while(!tail.compare_exchange_weak(bx_size, QUERIES_BATCH_SIZE));
	do_flush(bx_size);
}

class mask_queue_pair : public partition_queue {
public:
	filter_t mask;     // it holds common bits of all filters in this partition
public:
	void initialize(partition_id_t p, const filter_t & m) {
		partition_queue::initialize(p);
		mask = m;
	}
};

//
// We store all the queue_prefix pairs in three compact tables.  This
// should improve memory-access times.
//
static mask_queue_pair * ptable[filter_t::WIDTH + 1];

//
// this is the temporary front-end FIB
//
class tmp_part_descr {
public:
	partition_id_t id;
	const filter_t mask;

	tmp_part_descr(partition_id_t i, const filter_t & f) : id(i), mask(f) {};
};

static vector<tmp_part_descr> tmp_ptable[filter_t::WIDTH];

// This is how we compile the temporary FIB
//
void front_end::add_partition(unsigned int id, const filter_t & mask) {
	// Find the least crowded array in which to store the new
	// partition.
	//
	// ASSUME mask != all_zero
	//
	filter_pos_t min_i = mask.next_bit(0);
	unsigned int min = tmp_ptable[min_i].size();
	for (filter_pos_t i = mask.next_bit(min_i); i < filter_t::WIDTH; i = mask.next_bit(i + 1)) {
		if (tmp_ptable[i].size() < min) {
			min = tmp_ptable[i].size();
			min_i = i;
		}
	}
	tmp_ptable[min_i].emplace_back(id, mask);
}

// This is how we compile the real FIB from the temporary FIB
//
void front_end::consolidate() {
	unsigned int size = 0;
	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i)
		size += tmp_ptable[i].size();

	if (ptable[0]) {
		delete[](ptable[0]);
		ptable[0] = nullptr;
	}
	ptable[0] = new mask_queue_pair[size];

	mask_queue_pair * mqp = ptable[0];

	for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i) {
		for(const tmp_part_descr & p : tmp_ptable[i]) {
			mqp->initialize(p.id, p.mask);
			++mqp;
		}
		tmp_ptable[i].clear();
		ptable[i + 1] = mqp;
	}
}

// This is the main matching function
//
static void do_match(tagmatch_query * q) {
	const filter_t & f = q->filter;
	for (filter_pos_t i = f.next_bit(0); i < filter_t::WIDTH; i = f.next_bit(i + 1))
		for(mask_queue_pair * mqp = ptable[i]; mqp != ptable[i + 1]; ++mqp)
			if (mqp->mask.subset_of(f))
				mqp->enqueue(q);

	q->frontend_done();
	// Now we need to check whether any query has already been processed
	// and has already finished the match in the back_end
	//
	q->finalize_matching();
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
	FE_FINALIZE_FLUSHING = 4
};

static atomic<front_end_state> processing_state(FE_INITIAL);
static mutex processing_mtx;
static condition_variable processing_cv;

// We actually maintain a single queue, where we insert both matching
// and flushing jobs.  We can do that because the two phases are
// mutually exclusive.
//
union job {
	tagmatch_query * volatile p;
	partition_queue * volatile q;
};

static const size_t JOB_QUEUE_SIZE = 1024*1024; // must be a power of 2 for efficiency
static job job_queue[JOB_QUEUE_SIZE];

// However, we must still maintain two separate pairs of indexes
// (head,tail), one used by the matching loop, and the other used by
// the flushing loop.

// matching job queue indexes
//
static atomic<unsigned int> matching_job_queue_head(0);	// position of the first element
static atomic<unsigned int> matching_job_queue_tail(0);		// one-past position of the last element

// flushing job queue indexes
//
static atomic<unsigned int> flushing_job_queue_head(0);	// position of the first element
static unsigned int flushing_job_queue_tail = 0;		// one-past position of the last element

// number of threads that are still in the matching loop
static atomic<unsigned int> matching_threads(0);

//
// This is used to enqueue a matching job.  It suspends in a spin loop
// as long as the queue is full.
//
// ** WARNING: THIS IS TO BE USED BY A SINGLE THREAD. **
//
void front_end::match(tagmatch_query * p) noexcept {
	assert(processing_state == FE_INITIAL || processing_state == FE_MATCHING);
	unsigned int tail_plus_one = (matching_job_queue_tail + 1) % JOB_QUEUE_SIZE;

	while (tail_plus_one == matching_job_queue_head.load(std::memory_order_acquire))
		;		// full queue => spin

	job_queue[matching_job_queue_tail].p = p;
	matching_job_queue_tail.store(tail_plus_one, std::memory_order_release);
}

static milliseconds flush_limit(0);

unsigned int front_end::get_latency_limit_ms() {
	return flush_limit.count();
}

void front_end::set_latency_limit_ms(unsigned int l) {
	flush_limit = milliseconds(l);
}

static void match_loop() {
	static const unsigned int LATENCY_CHECK_SLACK_LIMIT = 10;
	unsigned int latency_check_slack = 0;
	unsigned int head, head_plus_one;
	tagmatch_query * p;
	for(;;) {
	main_loop:
		if (latency_check_slack < LATENCY_CHECK_SLACK_LIMIT) {
			++latency_check_slack;
		} else {
			latency_check_slack = 0;
			if (flush_limit > milliseconds(0)) {
				partition_queue * q;
				while((q = partition_queue::first_pending(flush_limit))) {
					q->flush();
				}
			}
		}
		head = matching_job_queue_head.load(std::memory_order_acquire);
		do {
			if (head == matching_job_queue_tail) {
                // empty queue
				if (processing_state.load(std::memory_order_acquire) == FE_MATCHING)
					// if we are still matching, then we spin
					goto main_loop;
				else
					// otherwise we get out of the match loop
					return;
			}
			p = job_queue[head].p;
			head_plus_one = (head + 1) % JOB_QUEUE_SIZE;
		} while (!matching_job_queue_head.compare_exchange_weak(head, head_plus_one));
        std::atomic_thread_fence(std::memory_order_acquire);
		do_match(p);
	}
}

static void flush_loop() {
	unsigned int head, head_plus_one;
	partition_queue * q;
	for(;;) {
	main_loop:
		head = flushing_job_queue_head.load(std::memory_order_acquire);
		do {
			if (head == flushing_job_queue_tail) {
			// empty queue
				if (processing_state.load(std::memory_order_acquire) == FE_FLUSHING) {
					// if we are still flushing, then we spin
					goto main_loop;
				} else {
					// otherwise we are done
					return;
				}
			}
			q = job_queue[head].q;
			head_plus_one = (head + 1) % JOB_QUEUE_SIZE;
		} while (!flushing_job_queue_head.compare_exchange_weak(head, head_plus_one));
        std::atomic_thread_fence(std::memory_order_acquire);
		q->flush();
	}
}

// this simply enqueues a query in the queue of matching jobs.
//
// WARNING: THIS IS TO BE USED BY A SINGLE THREAD.
//
static void enqueue_flush_job(partition_queue * q) {
	unsigned int tail_plus_one = (flushing_job_queue_tail + 1) % JOB_QUEUE_SIZE;

	while (tail_plus_one == flushing_job_queue_head.load(std::memory_order_acquire))
		; // full queue => spin

	job_queue[flushing_job_queue_tail].q = q;
	flushing_job_queue_tail = tail_plus_one;
}

static void processing_thread() {
	// Barrier: we hold until the state transition: FE_INITIAL ->
	// FE_MATCHING, at which point we enter the match loop
	do {
		unique_lock<mutex> lock(processing_mtx);
		while (processing_state == FE_INITIAL)
			processing_cv.wait(lock);
	} while(0);

	match_loop();

	// We got out of the match_loop, and if we are the last thread to
	// do so, we signal the stopping thread
	do {
		unique_lock<mutex> lock(processing_mtx);
		--matching_threads;
		processing_cv.notify_all();
	} while (0);
	// Barrier: we hold until the state transition:
	// FE_FINALIZE_MATCHING -> FE_FLUSHING, at which point we enter
	// the flush loop.
	do {
		unique_lock<mutex> lock(processing_mtx);
		while (processing_state == FE_FINALIZE_MATCHING)
			processing_cv.wait(lock);
	} while(0);

	flush_loop();
}

static vector<thread *> thread_pool;

void front_end::start(unsigned int threads) {
	if (threads == 0)
		return;

	unique_lock<mutex> lock(processing_mtx);

	// In order to make this method reentrant, we proceed only if the
	// front end is in the FE_INITIAL state.  Notice that this whole
	// method is synchronized, so only one start thread can proceed
	// beyond this barrier.
	//
	if (processing_state != FE_INITIAL)
		return;

	for (unsigned int i = 0; i < threads; ++i)
		thread_pool.push_back(new thread(processing_thread));

	matching_threads = threads;
	processing_state = FE_MATCHING;
	processing_cv.notify_all();
}

void front_end::stop() {
	// We proceed only when the front end is in the FE_MATCHING state.
	// This is because (1) FE_INITIAL means we are already stopped,
	// and (2) any other state beyond FE_MATCHING means that the
	// stopping process has already started.  This way, we make stop()
	// fully reentrant.
	do {
		unique_lock<mutex> lock(processing_mtx);
		if (processing_state != FE_MATCHING)
			return;

		// We switch from FE_MATCHING to FE_FINALIZE_MATCHING, which
		// causes the worker threads to get out of the matching loop.
		processing_state = FE_FINALIZE_MATCHING;
		processing_cv.notify_all();
	} while (0);

	// Barrier: we wait for all the worker threads to terminate the
	// matching loop, and then we switch to FE_FLUSHING
	do {
		unique_lock<mutex> lock(processing_mtx);
		while (matching_threads > 0)
			processing_cv.wait(lock);
		processing_state = FE_FLUSHING;
		processing_cv.notify_all();
	} while(0);

	// We then enqueue all pending batches for processing in the flush loop.
	if (ptable[0]) {
		for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i)
			for(mask_queue_pair * q = ptable[i]; q != ptable[i + 1]; ++q)
				enqueue_flush_job(q);
	}
	// When all pending batches are in the flush queue, we switch to
	// FE_FINALIZE_FLUSHING, which signals the worker threads that
	// they can get out of the flush loop as soon as they hit the end
	// of the flush queue.
	do {
		unique_lock<mutex> lock(processing_mtx);
		processing_state = FE_FINALIZE_FLUSHING;
		processing_cv.notify_all();
	} while(0);

	// Barrier: we wait for all the worker threads to terminate.
	//
	for(thread * t : thread_pool) {
		t->join();
		delete(t);
	}

	// Now there are no more threads (this is the only thread
	// operating on the front end), so we clean things up in the front
	// end
	thread_pool.clear();

	// And we then grab and process the left-over batches from the
	// back end.  We do that in two stages.
	//
	for (unsigned int s = 0; s < back_end::gpu_count() * GPU_STREAMS; s++) {
		batch * bx = back_end::flush_stream();
		if(bx)
			finalize_batch(bx);
	}

	back_end::release_stream_handles();

	for (unsigned int s = 0; s < back_end::gpu_count() * GPU_STREAMS; s++){
		batch * bx = back_end::second_flush_stream();
		if(bx)
			finalize_batch(bx);
	}
	back_end::release_stream_handles();

	do {
		unique_lock<mutex> lock(processing_mtx);
		processing_state = FE_INITIAL;
		processing_cv.notify_all();
	} while(0);
}

void front_end::clear() {
	if (ptable[0]) {
		delete[](ptable[0]);
		ptable[0] = nullptr;
	}
	batch_pool::clear();
}

ostream & front_end::print_statistics(ostream & os) {
	os << "partition  max latency (ms) enqueue count  flush count" << std::endl;
	if (ptable[0]) {
		for(filter_pos_t i = 0; i < filter_t::WIDTH; ++i)
			for(mask_queue_pair * q = ptable[i]; q != ptable[i + 1]; ++q)
				q->print_statistics(os);
	}
	return os;
}
