

#ifndef PARAMETERS_HH_INCLUDED
#define PARAMETERS_HH_INCLUDED

// number of packets in the batches of packets passed from the CPU
// front-end to the GPU back-end
//
//#define PACKETS_BATCH_SIZE		512U
//#define PACKETS_BATCH_SIZE		512U
#define GPU_BLOCK_DIM_X 32
#define GPU_BLOCK_DIM_Y  8
#define GPU_BLOCK_SIZE (GPU_BLOCK_DIM_X * GPU_BLOCK_DIM_Y) // Statc block size of 32*32 (1024)0

#define MULTIPLIER 2
#define PACKETS_BATCH_SIZE  (MULTIPLIER * GPU_BLOCK_SIZE) //  256U


#define INTERFACES				256U

#define DEFAULT_THREAD_COUNT	5U

#endif /* PARAMETERS_HH_INCLUDED */
