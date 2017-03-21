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
#include "packet.hh"
#include "front_end.hh"
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

// We implement the partition queues using a floating batch of
// packets.  The queues remain associated with their partition, but
// the packet batches (buffers) are independent object that we
// allocate and then recycle using a pool based on a free list that we
// access using a lock-free algorithm.
// 
class batch_pool;

union batch {
private:
	batch * next;

	friend batch_pool;

public:
	struct {
		unsigned int bsize;
		packet * packets[PACKETS_BATCH_SIZE];
	};
};

// A free-list allocator of packet batches.
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

	static void put(batch * b) {
		b->next = head.load(std::memory_order_acquire);
		do {
		} while(!head.compare_exchange_weak(b->next, b));
	}
};

#ifdef WITH_MATCH_STATISTICS
static std::mutex mtx;
std::vector<uint32_t> matches;
std::atomic<uint64_t> macthes_pre_merge, matches_post_merge;
#endif
mutex batch_pool::pool_mtx;
vector<batch *> batch_pool::pool;
atomic<batch *> batch_pool::head(nullptr);

// This class implements a central component of the front end, namely
// the queue of messages associated with each partition and used to
// buffer packets between the front end and the back end.
// 
class partition_queue {
	// primary information:
	// 
	 unsigned int partition_id;
	atomic<unsigned int> tail; // one-past the last element
	atomic<unsigned int> written; // number of elements actually written in the queue
	batch * b;				   // packet buffer

	// statistics and timing information
	// 
	high_resolution_clock::time_point first_enqueue_time;
#ifdef WITH_FRONTEND_STATISTICS
    milliseconds max_latency;
	unsigned int flush_count;
	unsigned int enqueue_count;
#endif

	// We also maintain a (global) list of pending queues.  These are
	// queues with at least one element, that therefore need to be
	// flushed to the back-end.  We add each queue to the back of the
	// list when we enqueue the first packet, and we remove the queue
	// from the list whenever we flush the queue.  We use these two
	// pointers, plus a static (global) "sentinel" to implement the
	// list in the most efficient and simple way.
	// 
	partition_queue * prev_pending;
	partition_queue * next_pending;

	void add_to_pending_pq_list() noexcept;
	void remove_from_pending_pq_list() noexcept;

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

	void enqueue(packet * p) noexcept;
	void flush() noexcept;

	// Flush the first pending queue.  Returns false when no pending
	// queue is found.
	// 
	static partition_queue * first_pending();

	// Flush the first pending queue whose current latency is greater
	// than the given limit.  Returns false if no such pending queue
	// is found.
	// 
	static partition_queue * first_pending(milliseconds latency_limit) noexcept;

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

// This is the sentinel of the list of pending partition queues.
// 
static partition_queue pending_pq_list;
static mutex pending_pq_list_mtx;

void partition_queue::add_to_pending_pq_list() noexcept {
    std::lock_guard<std::mutex> lock(pending_pq_list_mtx);
	next_pending = &pending_pq_list;
	prev_pending = pending_pq_list.prev_pending;
	pending_pq_list.prev_pending->next_pending = this;
	pending_pq_list.prev_pending = this;
}

void partition_queue::remove_from_pending_pq_list() noexcept {
    std::lock_guard<std::mutex> lock(pending_pq_list_mtx);
	next_pending->prev_pending = prev_pending;
	prev_pending->next_pending = next_pending;
}

partition_queue * partition_queue::first_pending(milliseconds latency_limit) noexcept {
	// we don't care about synchronizing this operation, since it is
	// the flush operation on q that really matters, and that removes
	// q from the pending list.
	// 
	// So, two threads might grab the same pending queue, but then
	// only one will succeed, and the other one will simply waste some
	// cycles.
	// 
	partition_queue * q = pending_pq_list.next_pending;
	if (q == &pending_pq_list)
		return nullptr;

	milliseconds current_latency 
		= duration_cast<milliseconds>(high_resolution_clock::now() - q->first_enqueue_time);
	if (current_latency < latency_limit)
		return nullptr;

	return q;
}

partition_queue * partition_queue::first_pending() {
	// We don't care to synchronize this operation; see comment above.
	// 
	partition_queue * q = pending_pq_list.next_pending;
	if (q == &pending_pq_list)
		return nullptr;

	return q;
}

void partition_queue::enqueue(packet * p) noexcept {
	assert(tail <= PACKETS_BATCH_SIZE);

	p->add_partition(partition_id);
	unsigned int t = tail.load(std::memory_order_acquire);
	
	do {
		while (t == PACKETS_BATCH_SIZE)
			t = tail.load(std::memory_order_acquire);
	} while(!tail.compare_exchange_weak(t, t + 1));

#ifdef WITH_FRONTEND_STATISTICS
	enqueue_count += 1;
#endif
	b->packets[t] = p;
	++written;
	if (t + 1 == PACKETS_BATCH_SIZE) {
		batch * bx = b;
		while(written.load(std::memory_order_acquire) < PACKETS_BATCH_SIZE)
			; // we loop waiting until every thread that acquired some
			  // (good) position in the queue, even earlier position
			  // than this one, actually completes the writing in the
			  // queue
			  //
		b = batch_pool::get();
#ifdef WITH_FRONTEND_STATISTICS
		update_flush_statistics();
#endif
		remove_from_pending_pq_list();
		written.store(0, std::memory_order_release);
		tail.store(0, std::memory_order_release);
		bx->bsize = PACKETS_BATCH_SIZE;
		bx = (batch *)back_end::process_batch(partition_id, bx->packets, PACKETS_BATCH_SIZE, (void *)bx);
		if (bx != NULL) {
			for (unsigned int r = 0; r < bx->bsize; r++) {
				packet * pkt = bx->packets[r];
		  		pkt->partition_done();
				if (pkt->is_matching_complete()) {
					if (pkt->finalize_matching()) {
#ifdef WITH_MATCH_STATISTICS
					macthes_pre_merge += pkt->getpre();
					matches_post_merge += pkt->getpost();
					mtx.lock();
					matches.push_back(pkt->getpost());
					mtx.unlock();
#endif
					}
				}
			}
			batch_pool::put(bx);
		}
	} else  {
		if (t == 0) {
			first_enqueue_time = high_resolution_clock::now();
			add_to_pending_pq_list();
		}
	}
}

void partition_queue::flush() noexcept {
	unsigned int bx_size = tail.load(std::memory_order_acquire);
	do {
		while (bx_size == PACKETS_BATCH_SIZE)
			bx_size = tail.load(std::memory_order_acquire);

		if (bx_size == 0)
			return;
	} while(!tail.compare_exchange_weak(bx_size, PACKETS_BATCH_SIZE));

	batch * bx = b;
	while(written.load(std::memory_order_acquire) < bx_size)
		; // we loop waiting until every thread that acquired some
	      // (good) position in the queue, even earlier position than
	      // this one, actually completes the writing in the queue
		  //
	b = batch_pool::get();
#ifdef WITH_FRONTEND_STATISTICS
	update_flush_statistics();
#endif
	remove_from_pending_pq_list();
	written.store(0, std::memory_order_release);
	tail.store(0, std::memory_order_release);
	bx->bsize = bx_size;
	bx = (batch *)back_end::process_batch(partition_id, bx->packets, bx_size, (void *)bx);
	if (bx != NULL) {
		for (unsigned int r = 0; r < bx->bsize; r++) {
			packet * pkt = bx->packets[r];
		  	pkt->partition_done();
			if (pkt->is_matching_complete()) {
				if (pkt->finalize_matching()) {
#ifdef WITH_MATCH_STATISTICS
					macthes_pre_merge += pkt->getpre();
					matches_post_merge += pkt->getpost();
					mtx.lock();
					matches.push_back(pkt->getpost());
					mtx.unlock();
#endif
				}
			}
		}
		batch_pool::put(bx);
	}
}

template<unsigned int Size>
class prefix_queue_pair : public partition_queue {
public:

	filter_t common_bits ;     // it holds common bits of all filters in this partition
public:
	void initialize(unsigned int id, const block_t * pb) {
		partition_queue::initialize(id);
		common_bits = back_end::get_cbits(id) ;
	}
};

typedef prefix_queue_pair<64> queue64;
// 
// container of prefixes whose leftmost 1-bit is in the 3rd 64-bit
// block.
// 
class p1_container {
public:
    // Since this is the third of three blocks, it may only contain
    // prefixes of up to 64 bits.  
    //
	// We store the prefixes in a contiguous section of the global
	// p64_table.
    //
	queue64 * p64_begin;
	queue64 * p64_end;

	ostream & print_statistics(ostream & os) {
		for(queue64 * i = p64_begin; i != p64_end; ++i)
			i->print_statistics(os);
		return os;
	}
};
//
// We store all the queue_prefix pairs in three compact tables.  This
// should improve memory-access times.
// 
static queue64 * p64_table = nullptr;

static p1_container pp1[64];
static p1_container pp2[64];
static p1_container pp3[64];

//
// this is the temporary front-end FIB 
// 
class tmp_prefix_descr {
public:
	unsigned int id;
	const filter_t filter;

	tmp_prefix_descr(unsigned int i, const filter_t f)
		: id(i), filter(f) {};
};

class tmp_prefix_pos_descr {
public:
	vector<tmp_prefix_descr> p64;

	tmp_prefix_pos_descr() : p64() {};
	
	void clear() {
		p64.clear();
	}
};

static tmp_prefix_pos_descr tmp_pp[192];

// This is how we compile the temporary FIB
// 
void front_end::add_prefix(unsigned int id, const filter_t & f, unsigned int n) {
	int index = -1;
	unsigned int min = UINT_MAX; 
	for (int i=0; i< 192; i++ ){
		if( f[i]==1){
			// Here I find the least crowded array to store the new one partition.
			//
			if(tmp_pp[i].p64.size() < min){
				min = tmp_pp[i].p64.size() ;
				index = i;
			}
		}
	}
	tmp_pp[index].p64.emplace_back(id,f) ;
}

// This is how we compile the real FIB from the temporary FIB
// 
static void compile_fib() {
	unsigned int p64_size = 0;

	for(unsigned int i = 0; i < 192; ++i) {
		p64_size += tmp_pp[i].p64.size();
	}

	p64_table = new queue64[p64_size];

	unsigned int p64_i = 0;

	for(unsigned int i = 0; i < 64; ++i) {
		pp1[i].p64_begin = p64_table + p64_i;
		for(const tmp_prefix_descr & d : tmp_pp[i].p64) 
			p64_table[p64_i++].initialize(d.id, d.filter.begin());
		pp1[i].p64_end = p64_table + p64_i;

		tmp_pp[i].clear();
	}

	for(unsigned int i = 64; i < 128; ++i) {
		pp2[i - 64].p64_begin = p64_table + p64_i;
		for(const tmp_prefix_descr & d : tmp_pp[i].p64) 
			p64_table[p64_i++].initialize(d.id, d.filter.begin() + 1);
		pp2[i - 64].p64_end = p64_table + p64_i;

		tmp_pp[i].clear();
	}

	for(unsigned int i = 128; i < 192; ++i) {
		pp3[i - 128].p64_begin = p64_table + p64_i;
		for(const tmp_prefix_descr & d : tmp_pp[i].p64) 
			p64_table[p64_i++].initialize(d.id, d.filter.begin() + 2);
		pp3[i - 128].p64_end = p64_table + p64_i;
		tmp_pp[i].clear();
	}
}

static bool use_identity_permutation = true;
static unsigned char bit_permutation[filter_t::WIDTH] = { 0 };

void front_end::set_identity_permutation() noexcept {
	use_identity_permutation = true;
}

void front_end::set_bit_permutation_pos(unsigned char old_pos, unsigned char new_pos) {
	use_identity_permutation = false;
	bit_permutation[old_pos] = new_pos;
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

			
			for(queue64 * q = c.p64_begin; q != c.p64_end; ++q) 
				if (q->common_bits.subset_of(pkt->filter)) 
					q->enqueue(pkt);

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);

	} if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			p1_container & c = pp2[m];

			for(queue64 * q = c.p64_begin; q != c.p64_end; ++q){ 
				if (q->common_bits.subset_of(pkt->filter)) 
					q->enqueue(pkt);
			}

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);

	} if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			p1_container & c = pp3[m];

			for(queue64 * q = c.p64_begin; q != c.p64_end; ++q) 
				if (q->common_bits.subset_of(pkt->filter)) 
					q->enqueue(pkt);

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);
	}

	pkt->frontend_done();
	
	// Now we need to check whether any packet has already been processed
	// and has already finished the match in the back_end
	//
	if (pkt->is_matching_complete()) {
		if (pkt->finalize_matching()) {
#ifdef WITH_MATCH_STATISTICS
			macthes_pre_merge += pkt->getpre();
			matches_post_merge += pkt->getpost();
			mtx.lock();
			matches.push_back(pkt->getpost());
			mtx.unlock();
#endif
		}
	}
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

static const size_t JOB_QUEUE_SIZE = 1024*1024; // must be a power of 2 for efficiency
volatile static job job_queue[JOB_QUEUE_SIZE];

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
void front_end::match(packet * pkt) noexcept {
	assert(processing_state == FE_INITIAL || processing_state == FE_MATCHING);
	unsigned int tail_plus_one = (matching_job_queue_tail + 1) % JOB_QUEUE_SIZE;

	while (tail_plus_one == matching_job_queue_head.load(std::memory_order_acquire))
		;		// full queue => spin

	job_queue[matching_job_queue_tail].p = pkt;
	matching_job_queue_tail = tail_plus_one;
}

static milliseconds flush_limit(0);

unsigned int front_end::get_latency_limit_ms() {
	return flush_limit.count();
}

void front_end::set_latency_limit_ms(unsigned int l) {
	flush_limit = milliseconds(l);
}

static void match_loop() {
	unsigned int head, head_plus_one;
	packet * p;

	for(;;) {
		if (flush_limit > milliseconds(0)) {
			partition_queue * q;
			while((q = partition_queue::first_pending(flush_limit)))
				q->flush();
		}
		head = matching_job_queue_head.load(std::memory_order_acquire);
		do {
			while (head == matching_job_queue_tail) {  
				// empty queue
				if (processing_state == FE_MATCHING) {
					// if we are still matching, then we spin
					head = matching_job_queue_head.load(std::memory_order_acquire);
					continue;
				} else {
					// otherwise, if we are stopped, then we switch to flushing
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
		head = flushing_job_queue_head.load(std::memory_order_acquire);
		do {
			while (head == flushing_job_queue_tail) {
				// empty queue
				if (processing_state == FE_FLUSHING) {
					// if we are still flushing, then we spin
					head = flushing_job_queue_head.load(std::memory_order_acquire);
					continue;		  
				} else {
					// otherwise, if we are done, then we terminate
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

	while (tail_plus_one == flushing_job_queue_head.load(std::memory_order_acquire)) 
		; // full queue => spin

	job_queue[flushing_job_queue_tail].q = q;
	flushing_job_queue_tail = tail_plus_one;
}

static void processing_thread() {
	match_loop();
	flush_loop();
}

static vector<thread *> thread_pool;

void front_end::start(unsigned int threads) {
#ifdef WITH_MATCH_STATISTICS
	macthes_pre_merge = 0;
	matches_post_merge = 0;
	matches.clear();
#endif
	if (processing_state == FE_INITIAL && threads > 0) {
		compile_fib();

		processing_state = FE_MATCHING;
		matching_threads = threads;
		do {
			thread_pool.push_back(new thread(processing_thread));
		} while(--threads > 0);
	}
}

void front_end::stop(unsigned int gpu_count) {
	do {
		unique_lock<mutex> lock(processing_mtx);
		processing_state = FE_FINALIZE_MATCHING;
		while(processing_state != FE_FLUSHING)
			processing_cv.wait(lock);
	} while(0);

	for(unsigned int j = 0; j < 64; ++j) {
		for(queue64 * i = pp1[j].p64_begin; i != pp1[j].p64_end; ++i)
			enqueue_flush_job(&(*i));

		for(queue64 * i = pp2[j].p64_begin; i != pp2[j].p64_end; ++i) 
			enqueue_flush_job(&(*i));

		for(queue64 * i = pp3[j].p64_begin; i != pp3[j].p64_end; ++i) 
			enqueue_flush_job(&(*i));
	}
	do {
		unique_lock<mutex> lock(processing_mtx);
		processing_state = FE_FINALIZE_FLUSHING;
		while(processing_state != FE_FINAL)
			processing_cv.wait(lock);
	} while(0);


	for(thread * t : thread_pool) {
		t->join();
		delete(t);
	}

	thread_pool.clear();

	for (unsigned int s = 0; s < gpu_count * GPU_STREAMS; s++){ 
		batch * bx = (batch *)back_end::flush_stream();
		if(!bx)
			continue;
		for (unsigned int r = 0; r < bx->bsize; r++) {
			packet * pkt = bx->packets[r];
			pkt->partition_done();
			if (pkt->is_matching_complete()) {
				if (pkt->finalize_matching()) {
#ifdef WITH_MATCH_STATISTICS
					macthes_pre_merge += pkt->getpre();
					matches_post_merge += pkt->getpost();
					mtx.lock();
					matches.push_back(pkt->getpost());
					mtx.unlock();
#endif
				}
			}
		}
		batch_pool::put(bx) ;
	}
	
	back_end::release_stream_handles(gpu_count);	
		
	for (unsigned int s = 0; s < gpu_count * GPU_STREAMS; s++){ 
		batch * bx = (batch *)back_end::second_flush_stream();
		if(!bx)
			continue;
		for (unsigned int r = 0; r < bx->bsize; r++) {
			packet * pkt = bx->packets[r];
			pkt->partition_done();
			if (pkt->is_matching_complete()) {
				if (pkt->finalize_matching()) {
#ifdef WITH_MATCH_STATISTICS
					macthes_pre_merge += pkt->getpre();
					matches_post_merge += pkt->getpost();
					mtx.lock();
					matches.push_back(pkt->getpost());
					mtx.unlock();
#endif
				}
			}
		}
		batch_pool::put(bx) ;
	}

	processing_state = FE_INITIAL;
#ifdef WITH_MATCH_STATISTICS
	std::cout << "Total matches before merge: " << macthes_pre_merge << std::endl;
	std::cout << "Total matches after merge: " << matches_post_merge << std::endl;
	for (uint32_t i=0; i < matches.size(); i++)
	{
		assert(matches[i]>0);
		std::cout << matches[i] << std::endl;
	}
#endif
}

void front_end::clear() {
	if (p64_table) {
		delete[](p64_table);
		p64_table = nullptr;
	}
	batch_pool::clear();
	use_identity_permutation = true;
}

ostream & front_end::print_statistics(ostream & os) {
	os << "partition  max latency (ms) enqueue count  flush count" << std::endl;

	for(int i = 0; i < 64; ++i) {
		pp1[i].print_statistics(os);
		pp2[i].print_statistics(os);
		pp3[i].print_statistics(os);
	}
	return os;
}
