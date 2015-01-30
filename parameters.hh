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
#define MULTIPLIER 1  
#define PACKETS_BATCH_SIZE (MULTIPLIER * GPU_BLOCK_SIZE) 


#define INTERFACES				256U

#define DEFAULT_THREAD_COUNT	5U

#endif /* PARAMETERS_HH_INCLUDED */
