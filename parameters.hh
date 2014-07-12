#ifndef PARAMETERS_HH_INCLUDED
#define PARAMETERS_HH_INCLUDED

// number of packets in the batches of packets passed from the CPU
// front-end to the GPU back-end
//
#define PACKETS_BATCH_SIZE		200
#define WITH_GPU_FAST_KERNEL	1
#define INTERFACES				200

#define WITH_PINNED_HOST_MEMORY	1

#define THREAD_COUNT			4

#endif /* PARAMETERS_HH_INCLUDED */
