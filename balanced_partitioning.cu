//
// This program is part of TagMatch, a subset-matching engine based on
// a hybrid CPU/GPU system.  TagMatch is also related to TagNet, a
// tag-based information and network system.
//
// This program implements the off-line partitioning algorithm of
// TagMatch.  The input is a set of Bloom filters, each representing a
// set of tags, and associated with a set of keys.  So:
//
// INPUT:
//
//    BF_1, k1, k2, ..
//    BF_2, k3, k4, ...
//    ...
//
// The program partitions the set of filters (and associated keys)
// into partitions so that all the filters in a partition share a
// common "mask" (a non-empty set of one-bits).  So, the output consists of two files:
//
// OUTPUT:
//
// Filters: This is the same as the input where each filter is also
// assigned a partition id:
//
//    Filters:
//    BF_1, partition-id_1, k1, k2, ..
//    BF_2, partition-id_2, k3, k4, ...
//    ...
//
// Partitions: This is the set of partitions, characterized by
// partition id, mask, size, etc.
//
//    Partitions:
//    partition-id_1, mask_1, size_1
//    partition-id_2, mask_2, size_2
//    ...
//
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "packet.hh"
#include "fib.hh"

#define GPU_THREADS 256
#define MAXTHREADS 24
#define CPU_MAX_PSIZE 128

// This does only reset the counters and the frequency buffers on the gpu 
__global__ void gzero(unsigned int *freqs, uint32_t * rcounter, uint32_t * lcounter) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= filter_t::WIDTH)
		return;
	freqs[tid] = 0;
	if (tid == 0) {
		*lcounter = 0;
		*rcounter = 0;
	}
}

// Computes the number of times every bit position in the range [0 - filter_t::WIDTH)
// is set to 1 in the fib, analyzing filters in the range [start - start + fib_size)
// Using __shared__ memory to store per block frequency arrays does not help in this
// application
__global__ void getfreqs(const uint32_t * __restrict__ fib,
								unsigned int fib_size,
								unsigned int start,
								uint32_t *freqs) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= fib_size)
		return;
	
	unsigned int mask = 1;
	for (int i = 0; i < filter_t::WIDTH; i++) {
		if (fib[(start + tid)*6 + (i / 32)] & (mask << (i % 32)))
			atomicAdd(&freqs[i], 1);
	}
}

// Does the same as std::stable_partition, but it is not guaranteed to be stable.
// Must be called before stable_partition_update (since a global barrier is needed)
__global__ void stable_partition(const uint32_t * __restrict__ fib,
		uint32_t begin, uint32_t size, uint32_t pivot,
		uint32_t * lcounter, uint32_t * rcounter, uint32_t * buffer) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= size)
		return;
	int idx;
	unsigned int mask = 1;
	if ((fib[(begin + tid) * 6 + (pivot / 32)] & (mask << (pivot % 32))) == 0) {
			idx = atomicAdd(lcounter, 1);
			#pragma unroll
			for (int i = 0 ; i < 6; i++) 
				buffer[(begin + idx) * 6 + i] = fib[(begin + tid) * 6 + i];
	}
	else {
			idx = atomicAdd(rcounter, 1);
			#pragma unroll
			for (int i = 0 ; i < 6; i++) 
				buffer[(begin + size - 1 - idx) * 6 + i] = fib[(begin + tid) * 6 + i];
	}
}

// Does the same as std::stable_partition, but it is not guaranteed to be stable.
// Must be called after stable_partition_update (since a global barrier is needed)
__global__ void stable_partition_update(uint32_t * fib,
		uint32_t begin, uint32_t size, const uint32_t * __restrict__ buffer) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= size)
		return;
	#pragma unroll
	for (int i = 0 ; i < 6; i++) 
		fib[(begin + tid) * 6 + i] = buffer[(begin + tid) * 6 + i];
}

using std::vector;
using std::endl;
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

static void print_usage(const char* progname) {
	std::cout << "usage: " << progname
	<< " [<params>...]\n"
	"\n  params: any combination of the following:\n"
	"     [m=<N>]         :: maximum size for each partition (default=100)\n"
	"     [t=<N>]         :: size of the thread pool (default=4)\n"
	"     [p=<filename>]  :: output for partitions, '-' means stdout (default=OFF)\n"
	"     [f=<filename>]  :: output for filters, '-' means stdout (default=OFF)\n"
	"     [in=<filename>]  :: input for filters (default=stdin)\n"
	"     [-a]  :: ascii input\n"
	"     [-b]  :: binary input\n"
	<< std::endl;
}

// We read and store fib_entry objects (see fib.hh).  We allocate each
// individual object, and then store the pointers to the objects in a
// vector that serves as an index.
//
static uint32_t *g_fib, *g_freqs[MAXTHREADS], *g_buffer, *g_rcounter[MAXTHREADS], *g_lcounter[MAXTHREADS];
static uint64_t *h_fib;
static cudaStream_t gstream[MAXTHREADS];
typedef vector<fib_entry *> fib_t;
static fib_t fib;

struct fib_entry_block {
	static const size_t FE_BLOCK_SIZE = 10000;
	fib_entry_block * next_block;
	size_t used_entries;
	fib_entry entries[FE_BLOCK_SIZE];
	fib_entry_block(fib_entry_block * n) : next_block(n), used_entries(0) {};
};

static fib_entry_block * fib_block_list = nullptr;

static fib_entry * new_fib_entry() {
	if (fib_block_list == nullptr || fib_block_list->used_entries == fib_entry_block::FE_BLOCK_SIZE) {
		fib_block_list = new fib_entry_block(fib_block_list);
	}
	return fib_block_list->entries + (fib_block_list->used_entries)++;
}

static void release_all_fib_entries() {
	while(fib_block_list) {
		fib_entry_block * tmp = fib_block_list;
		fib_block_list = fib_block_list->next_block;
		delete(tmp);
	}
}

static void read_filters(fib_t & fib, std::istream & input, bool binary_format) {
	fib_entry * f = new_fib_entry();
	if (binary_format) {
		while(f->read_binary(input)) {
			fib.push_back(f);
			f = new_fib_entry();
		}
	} else {
		while(f->read_ascii(input)) {
			fib.push_back(f);
			f = new_fib_entry();
		}
	}
}

// Compiles the fib on the host memory, so that it can be copied to the device
// global memory
void fibToArray(fib_t fib, uint32_t size) {
	for (uint64_t f = 0; f < size; f++) {
		const uint64_t * ptr = fib[f]->filter.begin64();
		for (uint64_t j = 0; j < 3; j++)
			h_fib[f*3 + j] = *ptr++; 
	}
}

// Predicate object used in stable partitioning.
//
struct has_zero_bit {
	unsigned int b;

	has_zero_bit(unsigned int bitpos): b(bitpos) {};
	bool operator()(const fib_entry * f) { return ! f->filter[b]; }
};

// We work with a list of partition candidates, which we then
// recursively partition.  Each partition candidate consists of a
// sequence of contiguous filters in the index (fib).
//
struct partition_candidate {
	fib_t::iterator begin;		// beginning of contiguous sequence in fib
	fib_t::iterator end;		// end of contiguous sequence in fib

	filter_t mask;				// mask of pivot bits that identify this partition
	filter_t used_bits;			// set of ALL pivot bits used so far

	bool clean_freq;					// frequencies do not need to be recalculated
	unsigned int freq[filter_t::WIDTH]; // frequencies of one bits in the filters

	struct partition_candidate * next;

	partition_candidate(fib_t::iterator b, fib_t::iterator e)
		: begin(b), end(e), clean_freq(false) {};

	void compute_frequencies(int tid) {
		unsigned int P_size = end - begin;

		// If the candidate_partition is too small, it may not be efficient to compute
		// the frequencies on the GPU
		if (P_size < CPU_MAX_PSIZE) {
			std::memset(freq, 0, sizeof(freq));
			for(fib_t::const_iterator i = begin; i != end; ++i)
				for(unsigned int b = (*i)->filter.next_bit(0); b < filter_t::WIDTH; b = (*i)->filter.next_bit(b + 1))
					freq[b] += 1;
		} else {
			unsigned int first = begin - fib.begin();
			unsigned int gridSize = P_size / GPU_THREADS + ((P_size % GPU_THREADS) != 0);
			getfreqs<<<gridSize, GPU_THREADS, 0, gstream[tid]>>>(g_fib, P_size, first, g_freqs[tid]);
			cudaMemcpyAsync(freq, g_freqs[tid], sizeof(freq), cudaMemcpyDeviceToHost, gstream[tid]);
			cudaStreamSynchronize(gstream[tid]);
			gzero<<<1, filter_t::WIDTH, 0, gstream[tid]>>>(g_freqs[tid], g_lcounter[tid], g_rcounter[tid]);
			int err;
			if ((err = cudaGetLastError()) != 0)
				std::cout << "CUDA error: " << err << " - " << filter_t::WIDTH << std::endl;
		}
	}

	void subtract_frequencies(const partition_candidate * x) {
		for(unsigned int b = 0; b < filter_t::WIDTH; ++b)
			freq[b] -= x->freq[b];
	}

	size_t size() const {
		return end - begin;
	}

	// Splits this partition in two sub-partitions accorting to the
	// given pivot.  This partition will contain all the filters that
	// have a one-bit in position pivot.  The result will be the
	// zero-partition, meaning the sequence of all filters with a zero
	// in the pivot position.
	// The same partition split must be performed on the GPU, in order
	// to have consistency at partition level.
	//
	partition_candidate * split_zero(unsigned int pivot, int tid) {
		uint32_t first = begin - fib.begin();
		unsigned int gridSize = size() / GPU_THREADS + ((size() % GPU_THREADS) != 0);
		gzero<<<1, filter_t::WIDTH, 0, gstream[tid]>>>(g_freqs[tid], g_lcounter[tid], g_rcounter[tid]);
		stable_partition<<<gridSize, GPU_THREADS, 0, gstream[tid]>>>(g_fib, first, (uint32_t)size(), pivot, g_lcounter[tid], g_rcounter[tid], g_buffer);
		stable_partition_update<<<gridSize, GPU_THREADS, 0, gstream[tid]>>>(g_fib, first, (uint32_t)size(), g_buffer);
		
		vector<fib_entry *>::iterator m = std::stable_partition(begin, end, has_zero_bit(pivot));
		partition_candidate * p0 = new partition_candidate(begin, m);
		begin = m;
		return p0;
	}
};

// We maintain a queue of pending partitions.  These are partitions
// whose size is creater than max_p and therefore that needs to be
// recursively partitioned.  A pool of threads will then pick
// partitions out of the queue for processing.
//
static std::mutex queue_mtx;
static std::condition_variable queue_cv;
static partition_candidate * pending_queue = nullptr;

static std::atomic<unsigned int> pending_partitions(0);

static void increment_pending_partitions() {
	++pending_partitions;
}

static void decrement_pending_partitions() {
	if (--pending_partitions == 0) {
		std::unique_lock<std::mutex> lock(queue_mtx);
		queue_cv.notify_all();
	}
}

static void enqueue_partition_candidate(partition_candidate * p) {
	increment_pending_partitions();
	std::unique_lock<std::mutex> lock(queue_mtx);
	p->next = pending_queue;
	pending_queue = p;
	queue_cv.notify_one();
}

static partition_candidate * dequeue_partition_candidate() {
	std::unique_lock<std::mutex> lock(queue_mtx);
	while (pending_queue == nullptr) {
		if (pending_partitions == 0)
			return nullptr;
		queue_cv.wait(lock);
	}
	partition_candidate * res = pending_queue;
	pending_queue = pending_queue->next;
	return res;
}

// We output a table of partition PT.  PT is the list of finished
// partitions, meaning partitions whose size is less than max_p and
// for which there is a non-zero mask.
//
static std::mutex pt_mtx;
static partition_candidate * PT;

// adds one partition to the PT list
//
static void PT_add(partition_candidate * p) {
	std::unique_lock<std::mutex> lock(pt_mtx);
	p->next = PT;
	PT = p;
}

// adds a list of partitions to the PT list.
//
static void PT_add_list(partition_candidate * first, partition_candidate * last) {
	std::unique_lock<std::mutex> lock(pt_mtx);
	last->next = PT;
	PT = first;
}

// Process a partition candidate p that is already smaller than the
// maximum partition size, but that is defined by an all-zero mask.
// Looks for bit positions that are common to all filters in p, and,
// if necessary, further partition p until there are no more all-zero
// masks, meaning that the zero split partition has a set of common
// non-zero bit positions.
//
// Create and return a list of partitions.  In case p can be returned
// without further partitioning (as a singleton list), returns p.
// Otherwise, return the head P0 of a list of partitions such that p
// is the last element of the list.  In other words, creates and
// connects every new partition as the head of the list, and then
// returns the head.
//
static partition_candidate * nonzero_mask_partitioning(partition_candidate * p, int tid) {
	for(;;) {
		p->compute_frequencies(tid);
		unsigned int max_freq = p->freq[0];
		unsigned int pivot = 0;
		for(unsigned int b = 1; b < filter_t::WIDTH; ++b) {
			if (p->freq[b] > max_freq) {
				pivot = b;
				max_freq = p->freq[b];
			}
			if (p->freq[b] == p->size())
				p->mask.set_bit(b);
		}
		if (max_freq == p->size())
			return p;
		partition_candidate * p0 = p->split_zero(pivot, tid);
		p->mask.set_bit(pivot);
		p0->next = p;
		p = p0;
	}
}

static unsigned int distance(unsigned int a, unsigned int b) {
	return (a > b) ? (a - b) : (b - a);
}

// Select a "pivot" bit position that divides p as evenly as possible
// into two sub-partitions p0 and p1.  The pivot bit is such that p0
// and p1 have a zero and a one in the pivot position, respectively.
// In other words, the pivot is the bit position whose frequency is
// the closest to 50% in p.
//
static unsigned int balancing_pivot(const partition_candidate * p) {
	unsigned int pivot;
	size_t p_half_size = p->size() / 2;
	for(pivot = 0; pivot < filter_t::WIDTH; ++pivot)
		if (! p->used_bits[pivot])
			break;
	unsigned int min_dist = distance(p_half_size, p->freq[pivot]);
	for(unsigned int b = pivot + 1; b < filter_t::WIDTH; ++b) {
		unsigned d = distance(p_half_size, p->freq[b]);
		if (d < min_dist) {
			pivot = b;
			if (d == 0) break;
			min_dist = d;
		}
	}
	return pivot;
}

static void balanced_partitioning(size_t max_p, int tid) {
	partition_candidate * p;
	while ((p = dequeue_partition_candidate()) != nullptr) {
		if (! p->clean_freq)
			p->compute_frequencies(tid);
		for (;;) {
			unsigned int pivot = balancing_pivot(p);
			partition_candidate * p0 = p->split_zero(pivot, tid);
			p->used_bits.set_bit(pivot);
			p0->used_bits = p->used_bits;
			p0->mask = p->mask;
			p->mask.set_bit(pivot);

			if (p0->size() > max_p && p->size() > max_p) {
				// Both sub-partitions need to be further processed.
				// So, we compute the frequencies for both parts,
				// which can be done more efficiently in a joint
				// computation, since p contains the total frequencies
				// for both p and p0.  Therefore we can simply compute
				// the frequencies of p0 and then subtract them from
				// those of p.  Or vice-versa when p is smaller than
				// p0.
				if (p0->size() <= p->size()) {
					p0->compute_frequencies(tid);
					p->subtract_frequencies(p0);
				} else {
					std::memcpy(p0->freq, p->freq, sizeof(p->freq));
					p->compute_frequencies(tid);
					p0->subtract_frequencies(p);
				}
				p0->clean_freq = true;
				enqueue_partition_candidate(p0);
			} else if (p->size() > max_p) {
				if (p0->mask.is_empty()) {
					p->clean_freq = false;
					enqueue_partition_candidate(p);
					PT_add_list(nonzero_mask_partitioning(p0, tid), p0);
					break; // loop to dequeue_partition_candidate()
				} else {
					p->compute_frequencies(tid);
					PT_add(p0);
				}
			} else if (p0->size() > max_p) {
				p0->compute_frequencies(tid);
				PT_add(p);
				p = p0;
			} else {
				p0->next = p;
				if (p0->mask.is_empty())
					p0 = nonzero_mask_partitioning(p0, tid);
				else
					gzero<<<1, filter_t::WIDTH, 0, gstream[tid]>>>(g_freqs[tid], g_lcounter[tid], g_rcounter[tid]);

				PT_add_list(p0, p);
				break;
			}
		}
		decrement_pending_partitions();
	}
}

int main(int argc, const char* argv[]) {
	const char* partitions_fname = nullptr;
	const char* filters_fname = nullptr;
	const char* input_fname = nullptr;

	unsigned int max_size = 200000;
	unsigned int thread_count = 4;
	bool binary_format = false;

	
	for (int i = 1; i < argc; ++i) {
		if (sscanf(argv[i], "m=%u", &max_size) || sscanf(argv[i], "N=%u", &max_size))
			continue;
		if (sscanf(argv[i], "t=%u", &thread_count))
			continue;
		if (strcmp(argv[i], "-a") == 0) {
			binary_format = false;
			continue;
		}
		if (strncmp(argv[i], "p=", 2) == 0) {
			partitions_fname = argv[i] + 2;
			continue;
		}
		if (strncmp(argv[i], "f=", 2) == 0) {
			filters_fname = argv[i] + 2;
			continue;
		}
		
		if (strcmp(argv[i], "-b") == 0) {
			binary_format = true;
			continue;
		}
		if (strncmp(argv[i], "in=", 3) == 0) {
			input_fname = argv[i] + 3;
			continue;
		}
		print_usage(argv[0]);
		return 1;
	}
	
	std::ofstream partitions_file;
	std::ofstream filters_file;
	std::ifstream input_file;

	std::ostream* partitions_output = nullptr;
	std::ostream* filters_output = nullptr;

	
	if (input_fname != nullptr) {
		std::ifstream input_file(input_fname);
		if (!input_file) {
			std::cout << "could not open input file " << input_fname << std::endl;
			return 1;
		}
		std::cout << "Reading filters..." << std::flush;
		read_filters(fib, input_file, binary_format);
		input_file.close();
	} else {
		std::cout << "Reading filters..." << std::flush;
		read_filters(fib, std::cin, binary_format);
	}
	std::cout << "\t\t\t" << std::setw(12) << fib.size() << " filters." << endl;
	std::cout << "Setting up gpu... ";
	cudaMallocHost(&h_fib, sizeof(uint64_t) * fib.size() * 3);
	fibToArray(fib, fib.size());
	cudaMalloc(&g_fib, sizeof(uint32_t) * fib.size() * 6);
	cudaMemcpy(g_fib, h_fib, sizeof(uint32_t) * fib.size() * 6, cudaMemcpyHostToDevice);
	
	for (int t=0; t<thread_count; t++) {
		cudaMalloc(&(g_rcounter[t]), sizeof(uint32_t));
		cudaMalloc(&(g_lcounter[t]), sizeof(uint32_t));
		cudaMalloc(&(g_freqs[t]), sizeof(uint32_t) * filter_t::WIDTH);
		gzero<<<1, filter_t::WIDTH>>>(g_freqs[t], g_lcounter[t], g_rcounter[t]);
		cudaStreamCreate(&(gstream[t]));
	}
	cudaMalloc(&g_buffer, sizeof(uint32_t) * fib.size() * 6);
	cudaDeviceSynchronize();
	cudaFreeHost(h_fib);
	int err;
	if ((err = cudaGetLastError()) != 0)
		std::cout << "Err: " << err << std::endl;
	std::cout << "\t\t\t" << std::setw(12) << "done!" << endl;
	
	if (partitions_fname) {
		if (strcmp(partitions_fname, "-") == 0) {
			partitions_output = &std::cout;
		} else {
			partitions_file.open(partitions_fname);
			if (!partitions_file) {
				std::cout << "error opening partitions file " << partitions_fname
				<< std::endl;
				return -1;
			}
			partitions_output = &partitions_file;
		}
	}
	
	if (filters_fname) {
		if (strcmp(filters_fname, "-") == 0) {
			filters_output = &std::cout;
		} else {
			filters_file.open(filters_fname);
			if (!filters_file) {
				std::cout << "error opening filters file " << filters_fname
				<< std::endl;
				return -1;
			}
			filters_output = &filters_file;
		}
	}
	
	std::cout << "Balanced partitioning..." << std::flush;

	high_resolution_clock::time_point start = high_resolution_clock::now();

	partition_candidate * p = new partition_candidate(fib.begin(), fib.end());
	p->used_bits.clear();
	p->mask.clear();

	if (p->size() > max_size) {
		std::vector<std::thread *> T(thread_count);
		enqueue_partition_candidate(p);
		for(unsigned int i = 0; i < thread_count; ++i)
			T[i] = new std::thread(balanced_partitioning, max_size, i);

		for(unsigned int i = 0; i < thread_count; ++i) {
			T[i]->join();
			delete(T[i]);
		}
	} else {
		p->next = nullptr;
		PT = nonzero_mask_partitioning(p, 0);
	}

	high_resolution_clock::time_point stop = high_resolution_clock::now();

	std::cout << "\t\t" << std::setw(12)
			  << duration_cast<milliseconds>(stop - start).count() << " ms." << std::endl;

	partition_id_t pid = 0;

	while(PT) {
		partition_prefix partition;

		partition.filter.fill();
		partition.length = filter_t::WIDTH;
		partition.partition = pid++;
		partition.size = PT->end - PT->begin;

		for(fib_t::iterator i = PT->begin; i != PT->end; ++i) {
			partition_fib_entry f;
			f.filter = (*i)->filter;
			partition.filter &= f.filter;
			f.keys = std::move((*i)->keys);
			f.partition = partition.partition;
			if (filters_output) {
				if (binary_format)
					f.write_binary(*filters_output);
				else
					f.write_ascii(*filters_output);
			}
		}
		if (partitions_output) {
			if (binary_format)
				partition.write_binary(*partitions_output);
			else
				partition.write_ascii(*partitions_output);
		}
		partition_candidate * tmp = PT;
		PT = PT->next;
		delete(tmp);
	}

	std::cout << "Number of partitions:\t\t\t" << std::setw(12) << pid << " partitions." << endl;

	release_all_fib_entries();
	fib.clear();
	cudaFree(&g_fib);
	for (int t=0; t<thread_count; t++) {
		cudaFree(&(g_rcounter[t]));
		cudaFree(&(g_lcounter[t]));
		cudaFree(&(g_freqs[t]));
		cudaStreamDestroy(gstream[t]);
	}
	cudaFree(&g_buffer);
	
	if (partitions_output == &partitions_file) partitions_file.close();
	if (filters_output == &filters_file) filters_file.close();
}
