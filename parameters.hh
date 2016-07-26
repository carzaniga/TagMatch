#ifndef PARAMETERS_HH_INCLUDED
#define PARAMETERS_HH_INCLUDED


#define GPU_BLOCK_DIM_X			32 
#define GPU_BLOCK_DIM_Y			8 

// Static block size of 32*32 (1024)0
#define GPU_BLOCK_SIZE (GPU_BLOCK_DIM_X * GPU_BLOCK_DIM_Y) 

// number of packets in the batches of packets passed from the CPU
// front-end to the GPU back-end
//
// now to MULTIPLIER has to be 1 because our output result_t only supports messageIDs of up to 256
#define MULTIPLIER 1 // DO NOT change this value! 
#define PACKETS_BATCH_SIZE (MULTIPLIER * GPU_BLOCK_SIZE) 
//This represents the maximum number of matches that can be produced for a single packet.
//In case there are more results, the list is truncated
#define MAX_MATCHES_PP 16384
#define MAX_MATCHES MAX_MATCHES_PP * PACKETS_BATCH_SIZE

#define NEW_PARTITIONING 1

// COALESCED_READS lays out the data so that each thread reads from a
// consecutive address so as to obtain higher memory throughput.  In
// practice this does not seem to be that effective.  So, we exclude
// it by default.
// 
#if 0
#define COALESCED_READS
#endif

#define INTERFACES				1U

#define DEFAULT_THREAD_COUNT	5U

// WITH_MATCH_STATISTICS makes cpu_gpu_matcher print statistics about the matches generated
// on the stdout
#if 0
#define WITH_MATCH_STATISTICS
#endif

#endif /* PARAMETERS_HH_INCLUDED */
