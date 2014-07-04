#include <stdio.h>
#include <stdint.h>

using namespace std;
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32
#define GPU_BLOCK_SIZE BLOCK_DIM_X * BLOCK_DIM_Y // Statc block size of 32*32 (1024)0
#define STREAMS 4 
#define PACKETS 500
#define GPU_FAST 1 
#define INTERFACES 3000

class GPU_matching{
public:
	static const unsigned int B_SIZE = 32;
	static const unsigned int Size= 192;
	typedef uint32_t GPU_block_t;
	static const unsigned int B_COUNT = Size / B_SIZE;
	void initialize();

	void memInfo();
	bool async_copyMSG(unsigned int * host_message, unsigned int packets, unsigned int stream_id);
	void deleteArray(unsigned int * dev_array);
	void syncStream(unsigned int stream_id, int k); 
	unsigned int * fillTable(unsigned int * host_array, unsigned int size);
	
	uint16_t * fillTiff(uint16_t * host_array, unsigned int size);
	void async_fillTiff(uint16_t * host_array, uint16_t * dev_array , unsigned int size, unsigned int stream_id);
	uint16_t * sync_alloc_tiff(unsigned int size);

	unsigned int * allocZeroes(unsigned int size);
	void async_setZeroes(unsigned int * dev_array, unsigned int size, unsigned int stream_id);
	void async_getResults(unsigned int * host_result, unsigned int * dev_result, unsigned int size, unsigned int stream_id);
	bool runKernel(unsigned int * dev_array, uint16_t * dev_global_tiff, unsigned int * local_tiff_index, uint16_t * dev_query_tiff, unsigned int * dev_result, unsigned int size, unsigned int packets, unsigned int stream_id);
	void finish();
	void releaseMem(unsigned int * p);
	struct stream_packets ;
	void init_streams() ;

};
