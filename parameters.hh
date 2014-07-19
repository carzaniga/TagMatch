#ifndef PARAMETERS_HH_INCLUDED
#define PARAMETERS_HH_INCLUDED

// number of packets in the batches of packets passed from the CPU
// front-end to the GPU back-end
//
#define PACKETS_BATCH_SIZE		256U
#define WITH_GPU_FAST_KERNEL	0
#define INTERFACES				256U

#define WITH_PINNED_HOST_MEMORY	1

#define DEFAULT_THREAD_COUNT	4U

#endif /* PARAMETERS_HH_INCLUDED */
