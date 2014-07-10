#include "GPU_matching.h"
#define test 0

#define cudaCheckErrors(msg) \
	do { \
		cudaError_t __err = cudaGetLastError(); \
		if (__err != cudaSuccess) { \
			fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
					msg, cudaGetErrorString(__err), \
					__FILE__, __LINE__); \
			fprintf(stderr, "*** FAILED - ABORTING\n"); \
			cudaDeviceReset() ;\
			exit(0);\
		} \
	} while (0)

cudaError_t err ; 
cudaStream_t stream [STREAMS] ;
__align__(32) __constant__ unsigned int dev_message [STREAMS][PACKETS_BATCH_SIZE*6]; //[194];

#define set_diff(a, b) (((a) & ~(b)))

__global__ void myKernel_minimal(unsigned int* data, uint16_t * global_tiff, unsigned int * prefix_tiff_index,  uint16_t * query_tiff ,  GPU_matching::iff_result_t * result, unsigned int n, unsigned int packets, unsigned int stream_id)
{
	//here I use packets, instead of PACKETS_BATCH_SIZE, because a matching queue can be sent to GPU even
	//when it is not full. (i.e if a timeout occures) 
	unsigned int id = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;
	if(id>=n)
		return;
	unsigned int d[6];
	unsigned int f_id=id*6 ;
	for(unsigned int i=0;i<6; i++)
		d[i]=data[f_id+i];
// i can replace it with packets.... 
	//int msg_id=-1 ;
	int msg_id=0 ;
//	if(n==9893)
//		printf("n = %d , f_id = %d , d[0] =%d , d[1] =%d \n", n,  f_id, d[0], d[1]) ;

//	for(unsigned int j=0; j<packets*6/*PACKETS_BATCH_SIZE*/ ; j+=6){
	for(unsigned int j=0; j<packets*6; j+=6,++msg_id){
//		msg_id++ ;
		if(set_diff(d[0], dev_message[stream_id][j])
		   || set_diff(d[1], dev_message[stream_id][j+1])
		   || set_diff(d[2], dev_message[stream_id][j+2])
		   || set_diff(d[3], dev_message[stream_id][j+3])
		   || set_diff(d[4], dev_message[stream_id][j+4])
		   || set_diff(d[5], dev_message[stream_id][j+5]))
			continue;

//		if(d[0] & ~dev_message[stream_id][j]!=0)
//			continue;
//		if(d[1] & ~dev_message[stream_id][j+1]!=0)
//			continue;
//		if(d[2] & ~dev_message[stream_id][j+2]!=0)
//			continue;
//		if(d[3] & ~dev_message[stream_id][j+3]!=0)
//			continue;
//		if(d[4] & ~dev_message[stream_id][j+4]!=0)
//			continue;
//		if(d[5] & ~dev_message[stream_id][j+5]!=0)
//			continue;
		
		uint16_t xor_temp;
		unsigned int tiff_index= prefix_tiff_index[id] ;
		unsigned int tiff_size = global_tiff[ tiff_index ] ;
//		uint16_t tiff_size= query_tiff[msg_id] ;
//		0x2000 = 2^13 
		for(unsigned int i=1; i <= tiff_size; i++){
			// may be this can be done with fewer operations.
			xor_temp= query_tiff[msg_id] ^ global_tiff[tiff_index+i] ;
			if((xor_temp<=0x1FFF) && (xor_temp!=0)){
				unsigned int temp = ((global_tiff[tiff_index+i]) & 0x1FFF) ; 
				result[(msg_id * INTERFACES) + temp] = 1 ;
			}
		}
	}
}

#define a_complement_not_subset_of_b(a,b) (~((a) | (b)))

__global__ void myKernel_fast(unsigned int* data, uint16_t * global_tiff, unsigned int * prefix_tiff_index,  uint16_t * query_tiff , GPU_matching::iff_result_t * result, unsigned int n, unsigned int packets, unsigned int stream_id)
{

	unsigned int t1 = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) ;
	unsigned int id = t1 + threadIdx.x;
//	unsigned int id = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;
	//unsigned int id = blockDim.x * blockIdx.x+ threadIdx.x;

	if(id>=n)
		return;

	unsigned int f_id = 6*(t1) + threadIdx.x;
//#if 1
//	unsigned int f_id = ((id / WARP_SIZE) * WARP_SIZE * 6) + (id % WARP_SIZE);
//#else
//#if (WARP_SIZE != 16) && (WARP_SIZE != 32)
//#error "WARP_SIZE must be either 16U or 32U"
//#endif
//	unsigned int f_id = (id & ~(WARP_SIZE-1))*6 + (id & (WARP_SIZE-1));
//#endif
		
	unsigned int d[6];
	d[0]=data[f_id];
	d[1]=data[f_id+32];
	d[2]=data[f_id+64];
	d[3]=data[f_id+96];
	d[4]=data[f_id+128];
	d[5]=data[f_id+160];
//	if(n==192)
//		printf("n = %d , f_id = %d , d[0] = %d, d[5]=%d msg[0]=%d \n", n,  f_id, d[0], d[5], dev_message[stream_id][0]) ;

	int msg_id=0;
	for(unsigned int j=0; j<packets*6 ; j+=6,++msg_id) {

#if 0
		if (a_complement_not_subset_of_b(d[0], dev_message[stream_id][j])
		    | a_complement_not_subset_of_b(d[1], dev_message[stream_id][j+1])
		    | a_complement_not_subset_of_b(d[2], dev_message[stream_id][j+2])
		    | a_complement_not_subset_of_b(d[3], dev_message[stream_id][j+3])
		    | a_complement_not_subset_of_b(d[4], dev_message[stream_id][j+4])
		    | a_complement_not_subset_of_b(d[5], dev_message[stream_id][j+5]))
			continue;
#else
		if (a_complement_not_subset_of_b(d[0], dev_message[stream_id][j]))
		    continue;
		if (a_complement_not_subset_of_b(d[1], dev_message[stream_id][j+1]))
		    continue;
		if (a_complement_not_subset_of_b(d[2], dev_message[stream_id][j+2]))
		    continue;
		if (a_complement_not_subset_of_b(d[3], dev_message[stream_id][j+3]))
		    continue;
		if (a_complement_not_subset_of_b(d[4], dev_message[stream_id][j+4]))
		    continue;
		if (a_complement_not_subset_of_b(d[5], dev_message[stream_id][j+5]))
		    continue;
#endif
		unsigned int tiff_index = prefix_tiff_index[id] ;
		unsigned int tiff_index_end = tiff_index + global_tiff[tiff_index] ;
//		result[(msg_id * INTERFACES) + 1] = 1 ; 
		while(++tiff_index <= tiff_index_end) {
			// may be this can be done with fewer operations.
			uint16_t xor_temp= query_tiff[msg_id] ^ global_tiff[tiff_index] ;
			if((xor_temp<=0x1FFF) && (xor_temp!=0)){
				unsigned int temp = ((global_tiff[tiff_index]) & 0x1FFF) ; 
				result[(msg_id * INTERFACES) + temp] = 1 ;
//				result[(msg_id * INTERFACES) + ((global_tiff[tiff_index]) & 0x1FFF)] = 1 ;
			}
		}
	}
}

void GPU_matching::initialize(){
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();
	init_streams();
}
void GPU_matching::memInfo(){
	size_t free;//size_t fr ;// (0);
	size_t total;
	err =cudaMemGetInfo(&free,&total);// cudaMemGetInfo(&free_bytes, n_bytes);
	if (err != cudaSuccess)
	{
		printf("cudaMemGetInfo returned the error: %s\n", cudaGetErrorString(err));
		cudaDeviceReset() ;
		exit(1);
	}
	printf("free= %f total_Mem=%f\n",(long unsigned)free*1.0/(1024*1024),(long unsigned)total*1.0/(1024*1024));
	err=cudaDeviceSynchronize() ;
	if (err != cudaSuccess){
		printf("wpw") ;
	}
}
bool GPU_matching::async_copyMSG(unsigned int * host_message, unsigned int packets , unsigned int stream_id){
	// err =cudaMemcpyToSymbol(&dev_message[stream_id][0], host_message, size_t(6*packets)*sizeof(unsigned int));
	err = cudaMemcpyToSymbolAsync(dev_message, host_message, 6*packets*sizeof(unsigned int), stream_id*PACKETS_BATCH_SIZE*6*sizeof(unsigned int), cudaMemcpyHostToDevice, stream[stream_id]);
	if (err != cudaSuccess){
		cudaCheckErrors("constant memory failed....\n");
		return false;
	}
	return true;
}

void GPU_matching::deleteArray(unsigned int * dev_array){
	//cudaFree(dev_array);
}

unsigned int * GPU_matching::fillTable(unsigned int * host_array, unsigned int size){
	unsigned int *dev_array = 0;
	err=cudaMalloc((void**)&dev_array, size * sizeof(unsigned int)); 
	if(err!=cudaSuccess){
		cudaCheckErrors("allocation fail");
		return NULL;
	}
	err=cudaMemcpy(dev_array, host_array, size * sizeof(unsigned int),cudaMemcpyHostToDevice);
	if(err!=cudaSuccess){
		cudaCheckErrors("memCpy fail!!");
		return NULL;
	}
	return dev_array;
}

//only used to send ann tiff pairs array for all fib entries. 
uint16_t * GPU_matching::fillTiff(uint16_t * host_array, unsigned int size){
	uint16_t * dev_array=0 ;
	err=cudaMalloc((void**)&dev_array, size * sizeof(uint16_t)); 
	if(err!=cudaSuccess){
		cudaCheckErrors("allocation fail");
	}
	err=cudaMemcpy(dev_array, host_array, size * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if(err!=cudaSuccess){
		cudaCheckErrors("memCpy fail!!");
	}
	return dev_array; 
}

// is used to send query_tiff to device. 
void GPU_matching::async_fillTiff(uint16_t * host_array, uint16_t * dev_array, unsigned int size, unsigned int stream_id){
	if(err!=cudaSuccess){
		printf("wtf") ;
		cudaDeviceReset() ;
		exit(0);
	}
	err=cudaMemcpyAsync(dev_array, host_array, size * sizeof(uint16_t),cudaMemcpyHostToDevice, stream[stream_id]);
	if(err!=cudaSuccess){
		cudaCheckErrors("memCpy fail!!");
	}
}

uint16_t * GPU_matching::sync_alloc_tiff( unsigned int size){
	uint16_t * dev_array= 0; 
	err=cudaMalloc((void**)&dev_array, size * sizeof(uint16_t)); 
	if(err!=cudaSuccess){
		cudaCheckErrors("allocation fail");
	}
	return dev_array; 
}

GPU_matching::iff_result_t * GPU_matching::allocZeroes(unsigned int size){ // this is useful for setting dev_res (interfaces) to 0 before calling the kernel
	GPU_matching::iff_result_t *dev_array = 0;
	err=cudaMalloc((void**)&dev_array, size * sizeof(iff_result_t)); 
	if(err!=cudaSuccess)
		return NULL;
	err=cudaMemset(dev_array, 0, size * sizeof(iff_result_t));   

	if(err!=cudaSuccess){
		cudaCheckErrors("allocation of zeroes failed");
		return NULL;
	}
	return dev_array; 
}

void GPU_matching::async_setZeroes(GPU_matching::iff_result_t * dev_array, unsigned int size, unsigned int stream_id){ // this is useful for setting dev_res (interfaces) to 0 before calling the kernel
	err= cudaMemsetAsync(dev_array, 0, size * sizeof(iff_result_t), stream[stream_id]);   

	if(err!=cudaSuccess){
		cudaCheckErrors("setMem fail");
	}
}


void GPU_matching::async_getResults(GPU_matching::iff_result_t * host_result, GPU_matching::iff_result_t * dev_result, unsigned int size, unsigned int stream_id){
	
	err=cudaMemcpyAsync(host_result, dev_result, size * sizeof(GPU_matching::iff_result_t), cudaMemcpyDeviceToHost, stream[stream_id]);
	if(err!=cudaSuccess){
		cudaCheckErrors("getResults failed");
	}
}

void GPU_matching::syncStream(unsigned int stream_id, int k){
        err = cudaStreamSynchronize(stream[stream_id]) ;
	if(err!=cudaSuccess){
		printf("k=%d fuck\n",k) ;
		cudaDeviceReset() ;
		exit(0) ;
	}
}

#if PINNED
void * GPU_matching::genericRequestHostPinnedMem(unsigned int size, size_t obj_size) {	
	void * host_array_pinned=0 ;
	err = cudaMallocHost((void**)&host_array_pinned, size * obj_size);
	if (err != cudaSuccess)
		  cudaCheckErrors("Error allocating pinned host memoryn");
	return host_array_pinned ;
}
#endif

//unsigned int * GPU_matching::requestHostPinnedMem32(unsigned int size) {	
//	unsigned int * host_array_pinned=0 ;
//	err = cudaMallocHost((void**)&host_array_pinned, size * sizeof(unsigned int));
//	if (err != cudaSuccess)
//		  cudaCheckError("Error allocating pinned host memoryn");
//	return host_array_pinned ;
//}
//
//unsigned uint16_t * GPU_matching::requestHostPinnedMem16(unsigned int size) {	
//	uint16_t * host_array_pinned=0 ;
//	err = cudaMallocHost((void**)&host_array_pinned, size * sizeof(uint16_t));
//	if (err != cudaSuccess)
//		  cudaCheckError("Error allocating pinned host memoryn");
//	return host_array_pinned ;
//}

void GPU_matching::init_streams(){
	unsigned int n= STREAMS;
	for(unsigned int i=0; i<n ; i++){
		cudaError_t err=cudaStreamCreate ( &stream[i]) ;	
		if(err!=cudaSuccess){
			printf("cannot create new cuda_stream!\n");
			cudaDeviceReset() ;
			exit(0);
		}
	}

}

bool GPU_matching::runKernel(unsigned int * dev_array, uint16_t * dev_global_tiff, unsigned int * prefix_tiff_index, uint16_t * dev_query_tiff, GPU_matching::iff_result_t * dev_result, unsigned int size, unsigned int packets, unsigned int stream_id){
//	int gridsize = 1;
//	// Work out how many blocks of size 1024 are required to perform all of nCombinations
//	for(unsigned int nsize = gridsize * BLOCK_SIZE; nsize < size;//nCombinations;
//			gridsize++, nsize = gridsize * BLOCK_SIZE)
//		;
//`	printf("steam_id= %d\n", stream_id);

	dim3 grid, block;
	block = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);
	unsigned int b=GPU_BLOCK_SIZE;
	unsigned int gridsize = ceil(size*1.0/b);
	//printf("gridsize = %d, blocksize = %d , size=%d \n", gridsize, b, size);
	//printf("bla gridsize= %i\n", gridsize);  
	grid = dim3(gridsize);
#if GPU_FAST
	//myKernel_fast<<<gridsize, block,0 , stream[stream_id] >>> ((int*)dev_array, (int*)dev_result, size, packets, stream_id);
	myKernel_fast<<<gridsize, block,0 , stream[stream_id] >>> ((unsigned int*)dev_array, dev_global_tiff, prefix_tiff_index, dev_query_tiff, dev_result, size, packets, stream_id);
#else
	myKernel_minimal<<<gridsize, block,0 , stream[stream_id] >>> ((unsigned int*)dev_array, dev_global_tiff, prefix_tiff_index, dev_query_tiff, dev_result, size, packets, stream_id);
//	printf(" alg: simple\n");
	//myKernel_minimal<<<gridsize, block, 0, stream[stream_id] >>> ((int*)dev_array, (int*)dev_result, size, packets, stream_id);
#endif
	cudaCheckErrors("kernel fail");
//	// Wait for synchronize
////	cudaDeviceSynchronize();
//	cudaStreamSynchronize(stream[stream_id]) ;
//	printf("cuda_stream %d --- %lu \n",stream_id, stream[stream_id]) ;
//	cudaCheckErrors("sync fail");
//
	return true;
}
void GPU_matching::finish(){
	//do something;
	cudaDeviceReset() ;
	return;
}
void GPU_matching::releaseMem(unsigned int * p){
	cudaFree(p); 
	if (err != cudaSuccess){
		cudaCheckErrors("couldn't free memory on GPU");  
//	printf("couldn't free memory on GPU"); 

	}
}
