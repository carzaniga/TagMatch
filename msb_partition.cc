#ifdef HAVE_CONFIG_H
#include "config.h"
#else
#define HAVE_BUILTIN_CTZL
#endif
#include <iostream>
#include <fstream>
#include <sstream>
#include <climits>
#include <cstdlib>
#include <cstdio>				// sscanf
#include <cstring>				// strcmp
#include <vector>
#include <cstdint>
#include <chrono>
#include <cassert>
#include <thread>
#include "predicate.hh"
#include "main_GPU.h"

#ifndef THREAD_COUNT
#define THREAD_COUNT 5 
#endif

#ifndef STOP_THREAD
#define STOP_THREAD UINT_MAX
#endif


using namespace std;
using namespace std::chrono;
//static main_GPU mgpu;

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
main_GPU mgpu;

// size_of_prefixes is a vector that, for each prefix that identifies
// a partition, stores the size of that partition, that is, the number
// of filters in the partition.  
//
vector<unsigned int> size_of_prefixes;

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

typedef prefix<192> prefix_filter_t;

// TODO: consider defining a Packet class that includes the filter and
// the tree and interface that the packet comes from, this would allow
// us to maintain a single vector of "query" packets.
// 
static vector<prefix_filter_t> queries;	// we store all the queries here
static vector<tree_interface_pair> queries_tiff; 

// TODO: we need to use only the input packets, so this vector of
// copies of the query packets should go away
// 
static vector<main_GPU::GPU_filter> GPU_queries;	// we store all the queries here


/*******************************************************************************/
/*************************** OUPUT MANAGEMENT **********************************/
/*******************************************************************************/

// msg_output is a wrapper/descriptor for each input "query" packet.
//
class msg_output {
	// we count the number of queues (each associated with a
	// partition) where we insert the message.  We use this counter to
	// determine when we are done with the matching in all the
	// partitions.
    volatile unsigned int counter; 

    // done is true when we are done with the prefix matching, that is
    // when we enqueued the messge in all the possible partitions.
    bool done; 

    unsigned char iff[INTERFACES]; //array of 0/1. when a cell is set we have a match
	//on the corresponding interface. We could use
	//a bit vector but this should be faster
    
    void reset_iff() {
        for(int i=0; i<INTERFACES; ++i)
            iff[i]=0;
    }

public:

    //here we miss a vector of something to return the set of mathing interfaces 
    msg_output (): counter(0), done(0) { reset_iff(); };

    void inc_counter() {

        unsigned int old_c;
        unsigned int new_c;

    try_inc:

        old_c = counter;
        new_c = old_c + 1;
        
        if (!__sync_bool_compare_and_swap (&counter, old_c, new_c))
            goto try_inc;
    }

    void dec_counter() {

        unsigned int old_c;
        unsigned int new_c;

    try_dec:

        old_c = counter;
        new_c = old_c - 1;
        
        if (!__sync_bool_compare_and_swap (&counter, old_c, new_c))
            goto try_dec;
    }

    void reset() {
        counter = 0;
        done = false;
    }    

    unsigned int get_counter() {
        return counter;
    }
    
    void set_done() {
        done=true;
    }

    bool get_done() {
        return done;
    }

    void set_iff(unsigned int i) {
        iff[i]=1;
    }

    bool get_iff(unsigned int i) {
        if(iff[i]==1)
            return 1;
        return 0;
    } 
};

// we use a circular buffer to store the message/packet wrappers while
// they are processed by the GPU back-end.  Right now we do not check
// that the matching is "done" before overwriting the wrapper.
// Therefore it is crucial that the buffer is large enough NOT to
// cause us to overwrite a message/packet whose matching isn't done.
// 
// TODO: figure out the size parameter
// 
const static unsigned int OUTPUT_SIZE = 500000;
static msg_output output[OUTPUT_SIZE];

/*******************************************************************************/
/******************************** QUEUE ****************************************/
/*******************************************************************************/

class queue {
    static const unsigned int MAX_SIZE = PACKETS_BATCH_SIZE;
    static const unsigned int MULTI_QUEUE = 4; //this is the number of queues that we can use
             
	unsigned int prefix_id ;
    unsigned int actual_queue; //indicates which queue is actually used
    volatile unsigned int flushed; //indicates if the actual queue was alredy flushed
	
    //next indexes/locks are used for free_queues
    volatile unsigned int lock_free_queues_read;
    volatile unsigned int lock_free_queues_write; 
    volatile unsigned int head_free_queues; //first available queue
    volatile unsigned int tail_free_queues; //next slot availabe to add a new queue

    //tail refers to q
	volatile unsigned int tail; //one-past the last element

    unsigned int free_queues[MULTI_QUEUE];
	unsigned int q[MULTI_QUEUE][MAX_SIZE]; 

    void init_free_queues(){
        for (unsigned int i=0; i<MULTI_QUEUE; i++)
            free_queues[i] = i;
    }

    void pop_free_queue(){
        
        //this is executed by at most one thread every time
        //because we lock free_queues on read in the enqueue
        //procedure

        //we can pop only when we have a queue available 
        //the buffer is empty when head_free_queues == tail_free_queues

    try_pop:

        if(head_free_queues == tail_free_queues)
            goto try_pop;

        actual_queue = free_queues[head_free_queues];

        tail = 0;
        
        head_free_queues = (head_free_queues + 1) % MULTI_QUEUE; 
    }
    
    void push_free_queue(int index){

        //many threads may try to write at the same time
        //so we lock free_queues to avoid this problem
        
    try_push:

        if(__sync_lock_test_and_set (&lock_free_queues_write, 1))
            goto try_push;

        //I am the only one that can write now!        
        
        //we don't need any kind of check on tail_free_queue because
        //we can push only after a pop, so we always have free space

        free_queues[tail_free_queues] = index;
        tail_free_queues = (tail_free_queues + 1) % MULTI_QUEUE;
        
        __sync_lock_release(&lock_free_queues_write);

    }   


    void flush (unsigned int size, unsigned int flush_index) {
			
		//pop a new stream id (TODO: check the stram id pool!)
        unsigned int stream_id = (*mgpu.stream_queue.pop()); 
			
		unsigned int * q_packets =	mgpu.host_queries[stream_id] ; 
		uint16_t * q_packets_tiff = mgpu.host_query_tiff[stream_id] ;


		for (unsigned int q_index=0; q_index < size; ++q_index) {
			for(int j=0; j<6 ; ++j){
				q_packets[q_index*6 + j] = GPU_queries.at(q[flush_index][q_index]).b[j] ; 
			}
			q_packets_tiff[q_index]= queries_tiff.at(q[flush_index][q_index]).get() ;
		}

		// here "size" is the number of packets not the actual size. 
		// later it gets multplied by 6 to allocate required space on GPU. 
		
		mgpu.gpu_matcher.async_copyMSG(q_packets, size, stream_id);
		mgpu.gpu_matcher.async_fillTiff(mgpu.host_query_tiff[stream_id], mgpu.dev_query_tiff[stream_id], size , stream_id) ;

		// this can be moved to after matching part. (only if we initially set all of itto zero) 
		mgpu.gpu_matcher.async_setZeroes(mgpu.dev_results[stream_id], PACKETS_BATCH_SIZE*INTERFACES, stream_id) ; 

		mgpu.match(prefix_id, size, stream_id) ;

		//now we need to copy result back to CPU:
		mgpu.async_getResults(size, stream_id) ;
		mgpu.gpu_matcher.syncStream(stream_id, 4) ;
    

        //handle output of the gpu 
        unsigned int out_index;
		for(unsigned int i=0; i<size; ++i){
            out_index = q[flush_index][i] % OUTPUT_SIZE;
            output[out_index].dec_counter();
			for(unsigned int j=0; j<INTERFACES; ++j){
				if(mgpu.host_results[stream_id][i*INTERFACES+j]==1){
                    output[out_index].set_iff(j);
                }
			}
		}
		
        mgpu.stream_queue.push(&mgpu.stream_array[stream_id]) ;
        push_free_queue(flush_index);
	}

public:

    //actual_queue = 0 use the first available queue;
    //no lock on free_queues; flushed is true to avoid to flush the first queue
    //the first queue in fact is full because tail = MAX_SIZE. In this way we
    //force a thread to pop a free queue index from the free_queue vector 
    //this is important in order to have always the same number of pop and push
    //operations and keep head and tail of free-queues synchronized. 
		
    queue(unsigned int prefix_id_): prefix_id(prefix_id_), actual_queue(0), flushed(1), lock_free_queues_read(0), 
            lock_free_queues_write(0), head_free_queues(1), tail_free_queues(0), tail(MAX_SIZE) { init_free_queues(); };

	void enqueue (unsigned int n) {

		unsigned int old_t;
		unsigned int new_t;

	try_push:
		old_t = tail;
		new_t = old_t+1;

		if (new_t > MAX_SIZE){
            //the actual queue is full and some other thread is now in the flush
            //method. Here we can try to use an other queue if available. Only
            //one thread can acquire a new queue so we use the read lock
            
            if(!__sync_lock_test_and_set(&lock_free_queues_read, 1)){

                //just to be sure that some one didn't get the lock
                //even if some one else alredy got a new queue. 
                //This may happen if a thread enter the  pass (new_t > MAX_SIZE) 
                //and before test the lock some one else got a new queue

                if(tail==MAX_SIZE){
                    pop_free_queue();
                    __sync_lock_release(&flushed);
                }
                __sync_lock_release(&lock_free_queues_read);
            }

			goto try_push;
        }

		if (!__sync_bool_compare_and_swap (&tail, old_t, new_t))
			goto try_push;	

        q[actual_queue][old_t]=n;

		if (new_t == MAX_SIZE && !__sync_lock_test_and_set(&flushed, 1)) {
			flush(new_t,actual_queue); 
		}
    }

    void safe_flush() {

        //this function is called by the flush_thread 
        //to flush the queue every N packets
        //see flush_timer to set the value of N
        
        //this check is not really precise but we don't care
        //in the worst case we flush the queue in the next iteration
        //it may happen that during the check on tail some other
        //thread enqueue a packet. The check on flushed is necessary 
        //in order to avoid to flush the same queue twice.

        if(tail!=0 && !__sync_lock_test_and_set(&flushed, 1)){
            unsigned int old_t;
            unsigned int last_used_q;

        try_flush:
            old_t = tail;

            last_used_q = actual_queue;

            if (!__sync_bool_compare_and_swap (&tail, old_t, MAX_SIZE))
		        goto try_flush;

            //at this point the queue is full so threads can't enqueue 
            //queries anymore. we can call the flush metod 
        
            flush(old_t,last_used_q); 

        }
    }

    unsigned int get_tail(){
        return tail;
    }
};

/*******************************************************************************/
/*******************************************************************************/

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

void fib_add_prefix(const prefix_filter_t & f, unsigned int n, queue * q) {
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

void fib_match(const unsigned int query_id){ //prefix_filter_t * q) {
	prefix_filter_t q = queries[query_id] ;

    unsigned int out_index = query_id % OUTPUT_SIZE;
    output[out_index].reset();

	const block_t * b = q.begin();

    if (*b) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p1_container & c = pp1[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b)){
                    output[out_index].inc_counter();
					i->q->enqueue(query_id);
                }

			for(vector<queue128>::const_iterator i = c.p128.begin(); i != c.p128.end(); ++i) 
				if (i->p.subset_of(b)){
                    output[out_index].inc_counter();
					i->q->enqueue(query_id);
                }

			for(vector<queue192>::const_iterator i = c.p192.begin(); i != c.p192.end(); ++i) 
				if (i->p.subset_of(b)){
                    output[out_index].inc_counter();
					i->q->enqueue(query_id);
                }

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);
			
    } else if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p2_container & c = pp2[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b)){
                    output[out_index].inc_counter();
					i->q->enqueue(query_id);
                }

			for(vector<queue128>::const_iterator i = c.p128.begin(); i != c.p128.end(); ++i) 
				if (i->p.subset_of(b)){
                    output[out_index].inc_counter();
					i->q->enqueue(query_id);
                }

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);

    } else if (*(++b)) {
		block_t curr_block = *b;
		do {
			int m = leftmost_bit(curr_block);
			const p3_container & c = pp3[m];

			for(vector<queue64>::const_iterator i = c.p64.begin(); i != c.p64.end(); ++i) 
				if (i->p.subset_of(b)){
                    output[out_index].inc_counter();
					i->q->enqueue(query_id);
                }

			curr_block ^= (BLOCK_ONE << m);
		} while (curr_block != 0);
    }
    output[out_index].set_done();
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

/*******************************************************************************/
/**************************** FLUSH TIMER **************************************/
/*******************************************************************************/


class flush_timer{

    const unsigned int FLUSH_RATE = 20000; 
    volatile unsigned int counter;
    volatile bool stop;

 public:   

    flush_timer(): counter(0), stop(0){};

    void inc_counter(){
        unsigned int old_c;
        unsigned int new_c;
    try_inc:
        old_c = counter;
        new_c = old_c+1;
        if (!__sync_bool_compare_and_swap (&counter, old_c, new_c))
            goto try_inc;
    }

    bool do_flush(){
        return (counter >= FLUSH_RATE);
    }

    bool stop_flush(){
        return stop;
    }

    void set_stop_flush(){
        stop=true;
    }

    void reset(){
        //maybe I don't even need to synch here!
        unsigned int old_c;
    try_reset:
        old_c = counter;
        if (!__sync_bool_compare_and_swap (&counter, old_c, 0))
            goto try_reset;
    }
};

void flush_queues(){

    for(int j = 0; j< 64; j++){
        for(vector<queue64>::const_iterator i = pp1[j].p64.begin(); i != pp1[j].p64.end(); ++i)
            if (i->q)
			    i->q->safe_flush();		
    }

    for(int j = 0; j< 64; j++){
        for(vector<queue128>::const_iterator i = pp1[j].p128.begin(); i != pp1[j].p128.end(); ++i) 
            if (i->q)
			    i->q->safe_flush();
    }

    for(int j = 0; j< 64; j++){
        for(vector<queue192>::const_iterator i = pp1[j].p192.begin(); i != pp1[j].p192.end(); ++i) 
            if (i->q)
	            i->q->safe_flush();
    }

    for(int j = 0; j< 64; j++){
        for(vector<queue64>::const_iterator i = pp2[j].p64.begin(); i != pp2[j].p64.end(); ++i) 
            if (i->q)
	            i->q->safe_flush();
    }
    
    for(int j = 0; j< 64; j++){	    
        for(vector<queue128>::const_iterator i = pp2[j].p128.begin(); i != pp2[j].p128.end(); ++i) 
            if (i->q)
                i->q->safe_flush();
    }
		
    for(int j = 0; j< 64; j++){	
        for(vector<queue64>::const_iterator i = pp3[j].p64.begin(); i != pp3[j].p64.end(); ++i) 
            if (i->q)
                i->q->safe_flush();
    }
}

void flushing_thread(flush_timer * ft){

    while(!ft->stop_flush()){
        //busy loop to wait for flush
        while(!ft->do_flush() && !ft->stop_flush());

        if(ft->stop_flush())
            return;		        
    
        flush_queues();
        
        ft->reset();
    }     
}


/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/******************************THREAD POOL**************************************/
/*******************************************************************************/

static const size_t JOB_QUEUE_SIZE = 1024; // must be a power of 2 for efficiency
//static const prefix_filter_t * job_queue[JOB_QUEUE_SIZE];
static unsigned int job_queue[JOB_QUEUE_SIZE];

static volatile unsigned int job_queue_head = 0;	// position of the first element 
static volatile unsigned int job_queue_tail = 0;	// one-past position of the last element

static void match_job_enqueue(unsigned int query_id){ //const prefix_filter_t * f) {
	unsigned int tail_plus_one;
	unsigned int my_tail;

	my_tail = job_queue_tail;
	tail_plus_one = (my_tail + 1) % JOB_QUEUE_SIZE;

	while (tail_plus_one == job_queue_head) {
		// full queue => busy loop
		;
	}

	job_queue[my_tail] = query_id; // f;
	job_queue_tail = tail_plus_one;
}

//static const prefix_filter_t * match_job_dequeue() {
static const unsigned int match_job_dequeue() {
	unsigned int my_head, head_plus_one;

 try_dequeue:
	my_head = job_queue_head;

	if (my_head == job_queue_tail)
		goto try_dequeue;		   // empty queue => busy loop

	head_plus_one = (my_head + 1) % JOB_QUEUE_SIZE;

	//const prefix_filter_t * result = job_queue[my_head];
	const unsigned int result = job_queue[my_head];

	if (!__sync_bool_compare_and_swap(&job_queue_head, my_head, head_plus_one))
		goto try_dequeue;

	return result;
}

std::thread * thread_pool[THREAD_COUNT];

void thread_loop(flush_timer * ft) {
	unsigned int f;
	while((f = match_job_dequeue()) != STOP_THREAD ){
		fib_match(f);
        ft->inc_counter();
    }
}

/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/****************************INPUT FILES MANAGEMENT*****************************/
/*******************************************************************************/

static std::vector<tree_interface_pair> ti_pairs;
//static std::vector<

void read_prefixes_vector(string fname){
	ifstream is (fname) ;
	string line;
	if (is.is_open()) {
		while(std::getline(is, line)) {
			istringstream line_s(line);
			string command;
			line_s >>command ;
			if(command != "p"){
				continue;
			}
			unsigned int prefix_id, prefix_size;
			std::string prefix_string;

			line_s >> prefix_id >> prefix_string >> prefix_size;
//			int * host_fib= (int *)malloc(prefix_size * 24 ) ;
			
			prefix_filter_t f(prefix_string);
			unsigned int n = prefix_string.size();
			fib_add_prefix(f,n,new queue(prefix_id));
			//p_size++ ;
			size_of_prefixes.push_back(prefix_size) ;
		}
	}
	else 
		cerr<< " prefix file doesn't exist! " << endl;
}

unsigned int correct_input (vector<main_GPU::filter_descr> * filters , unsigned int ti_counter) {
#if GPU_FAST
	string filter="111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111" ;
	unsigned int tree=7u ;// this is just a dummy tiff.
	unsigned int iface=8191u  ;// this is just a dummy tiff.

	for(unsigned int prefix_id=0 ; prefix_id < size_of_prefixes.size() ; ++prefix_id ){
		while(filters[prefix_id].size()%32!=0){
			ti_pairs.push_back(tree_interface_pair(tree, iface));
			ti_counter+=2 ;//one for the size + one for the tiff  
			main_GPU::filter_descr temp = main_GPU::filter_descr(filter, ti_pairs.size() -1, ti_pairs.size() ) ;		
			
			filters[prefix_id].push_back(temp);
			size_of_prefixes[prefix_id]++ ;
		}
	}
#endif
	return ti_counter; 
}

unsigned int read_filters_vector (vector<main_GPU::filter_descr> * filters , string fname) {
	ifstream is (fname) ;
	string line;

	unsigned int ti_counter=0;
	if (is.is_open()) {
		while(std::getline(is, line)) {
			istringstream line_s(line);
			string command;
			line_s >> command;
			if (command != "f") 
				continue;

			unsigned int prefix_no, iface, tree;
			std::string filter;

			line_s >> prefix_no >> filter;

			unsigned int begin = ti_pairs.size();

			while (line_s >> tree >> iface){
				ti_pairs.push_back(tree_interface_pair(tree, iface));
			} 

			unsigned int end = ti_pairs.size();
			ti_counter+= end-begin + 1 ;
			//(+1 is to also later store the size of each
			//ti_pair array for each fib entry
			
			//filters[prefix_no].push_back(main_GPU::filter_descr(filter, begin, end));
			main_GPU::filter_descr temp = main_GPU::filter_descr(filter, begin, end) ;
			filters[prefix_no].push_back(temp);
		}
	}
	else 
		cerr<< " filter file doesn't exist! " << endl;
	cout<< "#ti_sizes from file = " << ti_pairs.size() << endl ;
	return ti_counter; 
}

void read_queries_vector(string fname){
	ifstream is (fname) ;
	string line;
	if (is.is_open()) {
		while(std::getline(is, line)) {
			istringstream line_s(line);
			string command;
			line_s>>command ;
			if(command != "!")
				continue;
			unsigned int tree, interface;
			std::string query_string;
			
			line_s >> tree >> interface >> query_string;
			prefix_filter_t f(query_string);

			main_GPU::GPU_filter f_GPU ;
			f_GPU = mgpu.assign(query_string); //this creates a filter which is 
                                               //6 integer representation of the query string. 
			GPU_queries.push_back(f_GPU) ;
			queries_tiff.push_back(tree_interface_pair(tree, interface)) ;
			queries.push_back(f) ;
		}
	}
	else 
		cerr<< " query file doesn't exist! " << endl;
}

/*******************************************************************************/
/*******************************************************************************/


int main(int argc, const char * argv[]) {
	unsigned int N = 1;			// how many cycles throug the queries?
	string prefixes_fname, filters_fname, queries_fname; 

#if GPU_FAST
	cout<< "algorithm: Fast" << endl;
#else
	cout<< "algorithm: Normal" << endl;
#endif

	for(int i = 1; i < argc; ++i) {
		if (strncmp(argv[i],"p=",2)==0) {
			prefixes_fname = argv[i] + 2;
			continue;
		}
		if (strncmp(argv[i],"f=",2)==0) {
			filters_fname = argv[i] + 2;
			continue;
		}
		if (strncmp(argv[i],"q=",2)==0) {
			queries_fname = argv[i] + 2;
			continue;
		}

		if (strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0) {
			std::cout << "usage: " << argv[0] << " p=prefix_file_name f=filters_file_name q=queries_file_name\n"
					  << std::endl;
			return 1;
		}
	}

	// its better to pass the pointer to each vector for the following methods:
	read_prefixes_vector(prefixes_fname);
	
	std::vector<main_GPU::filter_descr> * filters;
	
	filters = new vector<main_GPU::filter_descr>[size_of_prefixes.size()] ;

	unsigned int ti_counter = read_filters_vector(filters, filters_fname);
	ti_counter = correct_input(filters,ti_counter);
	
	mgpu.init(size_of_prefixes, ti_counter);

	mgpu.read_tables(filters, ti_pairs);
	cout<< "filling tables done" << endl ;
	mgpu.move_to_GPU(); // moves Forwrding table to GPU. 
	cout<<"moving to GPU is done "<< endl;

	cout<<"reading queries ..." ;
	read_queries_vector(queries_fname);

	cout<<"done" << endl ;

	flush_timer * ft = new flush_timer();

    thread * thread_flush = new thread(flushing_thread,ft);
    
	for(size_t i = 0; i < THREAD_COUNT; ++i)
        thread_pool[i] = new thread(thread_loop,ft);

    high_resolution_clock::time_point start = high_resolution_clock::now();

	for(unsigned int round = 0; round < N; ++round) 
		for(unsigned int query_id=0; query_id < queries.size(); query_id++)
			match_job_enqueue(query_id);

	for(size_t i = 0; i < THREAD_COUNT; ++i)
		match_job_enqueue(STOP_THREAD);

	for(size_t i = 0; i < THREAD_COUNT; ++i)
		thread_pool[i]->join();

    flush_queues();

    high_resolution_clock::time_point stop = high_resolution_clock::now();

	for(size_t i = 0; i < THREAD_COUNT; ++i)
		delete(thread_pool[i]);

    ft->set_stop_flush();
    thread_flush->join();

    nanoseconds ns = duration_cast<nanoseconds>(stop - start);
    cout << "queries: " << queries.size() << endl
		 << "Average matching time: " << ns.count()/queries.size()/N << "ns" << endl;

#define PRINT_OUTPUT 1
#if PRINT_OUTPUT
    for (unsigned int i=0; i<OUTPUT_SIZE; ++i){
        //cout << i << " " << (output[i].get_done()==1) << " "  <<  (output[i].get_counter()==0) << endl; 
        if(output[i].get_done() && output[i].get_counter()==0){
            cout << "query " << i << " iff: ";
            for(unsigned int j=0; j<INTERFACES; ++j){
                if(output[i].get_iff(j)==1)
                    cout << j << " ";
            }
            cout << endl;
        }
    }
#endif
	
	destroy_fib();
   	mgpu.destroy_fibs() ; 
	return 0;
}
