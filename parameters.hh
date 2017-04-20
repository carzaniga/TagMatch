#ifndef PARAMETERS_HH_INCLUDED
#define PARAMETERS_HH_INCLUDED


// Static block size of 256, 1 dimension
//
#define GPU_BLOCK_SIZE 256 

// number of queries in the batches of queries passed from the CPU
// front-end to the GPU back-end
//
#define BATCH_SIZE_MULTIPLIER 1  
#define QUERIES_BATCH_SIZE (BATCH_SIZE_MULTIPLIER * GPU_BLOCK_SIZE)

//This represents the maximum number of matches that can be produced for a single query.
//
#define MAX_MATCHES_PP 16384

// Maximum number of results for a single kernel execution.
// In case there are more results, the list is truncated and one assertion in the back-end should be triggered
//
#define MAX_MATCHES MAX_MATCHES_PP * QUERIES_BATCH_SIZE

// COALESCED_READS lays out the data so that each thread reads from a
// consecutive address so as to obtain higher memory throughput.  In
// practice this does not seem to be that effective.  So, we exclude
// it by default.
//
#if 0
#define COALESCED_READS
#endif

#define DEFAULT_THREAD_COUNT	5U
#define DEFAULT_GPU_COUNT		1U

// Defines whether the match or match-unique algorithm is performed. Refer to the paper
// for more info
//
#define MATCH_UNIQUE 0

#endif /* PARAMETERS_HH_INCLUDED */
