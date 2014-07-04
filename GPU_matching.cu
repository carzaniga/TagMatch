//#include <stdlib.h>
//#include <cstdlib>
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
__align__(32) __constant__ unsigned int dev_message [STREAMS][PACKETS*6]; //[194];

#define not_subset_of(a, b) (((a) & ~(b)) != 0)

__global__ void myKernel_minimal(unsigned int* data, uint16_t * global_tiff, unsigned int * prefix_tiff_index,  uint16_t * query_tiff , unsigned int * result, unsigned int n, unsigned int packets, unsigned int stream_id)
{
	//here I use packets, instead of PACKETS, because a matching queue can be sent to GPU even
	//when it is not full. (i.e if a timeout occures) 
	unsigned int id = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;
	if(id>=n)
		return;
	unsigned int d[6];
	unsigned int f_id=id*6 ;
	for(unsigned int i=0;i<6; i++)
		d[i]=data[f_id+i];
// i can replace it with packets.... 
	int msg_id=-1 ;

	for(unsigned int j=0; j<packets*6/*PACKETS*/ ; j+=6){
		msg_id++ ;
		if(not_subset_of(d[0], dev_message[stream_id][j]))
			continue;
		if(not_subset_of(d[1], dev_message[stream_id][j+1]))
			continue;
		if(not_subset_of(d[2], dev_message[stream_id][j+2]))
			continue;
		if(not_subset_of(d[3], dev_message[stream_id][j+3]))
			continue;
		if(not_subset_of(d[4], dev_message[stream_id][j+4]))
			continue;
		if(not_subset_of(d[5], dev_message[stream_id][j+5]))
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

#define not_subset_of_complement(a,b) (((a) | (b)) != ~(0U))

__global__ void myKernel_fast(unsigned int* data, uint16_t * global_tiff, unsigned int * prefix_tiff_index,  uint16_t * query_tiff , unsigned int * result, unsigned int n, unsigned int packets, unsigned int stream_id)
{
	unsigned int id = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;
	//int id = blockDim.x * blockIdx.x+ threadIdx.x;

	if(id>=n)
		return;
	unsigned int f_id = id>>5;//(id / 32) ; // or id>>5
	f_id = (f_id*32*6)+ (id % 32) ; //32 warp size
		
	unsigned int d[6];
	d[0]=data[f_id];
	d[1]=data[f_id+32];
	d[2]=data[f_id+64];
	d[3]=data[f_id+96];
	d[4]=data[f_id+128];
	d[5]=data[f_id+160];

	int msg_id=-1 ;
#if 1 
	for(unsigned int j=0; j<packets*6 ; j+=6){
		msg_id++ ;
		if (not_subset_of_complement(d[0], dev_message[stream_id][j]))
			continue;
		if (not_subset_of_complement(d[1], dev_message[stream_id][j+1]))
			continue;
		if (not_subset_of_complement(d[2], dev_message[stream_id][j+2]))
			continue;
		if (not_subset_of_complement(d[3], dev_message[stream_id][j+3]))
			continue;
		if (not_subset_of_complement(d[4], dev_message[stream_id][j+4]))
			continue;
		if (not_subset_of_complement(d[5], dev_message[stream_id][j+5]))
			continue;
#else 

		if((d[0] | dev_message[stream_id][j])!= ~0)//0xFFFFFFFF)
		    continue;
		if((d[1] | dev_message[stream_id][j+1])!= ~0)//0xFFFFFFFF)
		    continue;
		if((d[2] | dev_message[stream_id][j+2])!= ~0)//0xFFFFFFFF)
		    continue;
		if((d[3] | dev_message[stream_id][j+3])!= ~0)//0xFFFFFFFF)
		    continue;
		if((d[4] | dev_message[stream_id][j+4])!= ~0)//0xFFFFFFFF)
		    continue;
		if((d[5] | dev_message[stream_id][j+5])!= ~0)//0xFFFFFFFF)
			continue;
#endif		
//		result[msg_id * INTERFACES ] = 1 ;
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
	err = cudaMemcpyToSymbolAsync(dev_message, host_message, 6*packets*sizeof(unsigned int), stream_id*PACKETS*6*sizeof(unsigned int), cudaMemcpyHostToDevice, stream[stream_id]);
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

unsigned int * GPU_matching::allocZeroes(unsigned int size){ // this is useful for setting dev_res (interfaces) to 0 before calling the kernel
	unsigned int *dev_array = 0;
	err=cudaMalloc((void**)&dev_array, size * sizeof(unsigned int)); 
	if(err!=cudaSuccess)
		return NULL;
	err=cudaMemset(dev_array, 0, size * sizeof(unsigned int));   

	if(err!=cudaSuccess){
		cudaCheckErrors("allocation of zeroes failed");
		return NULL;
	}
	return dev_array; 
}

void GPU_matching::async_setZeroes(unsigned int * dev_array, unsigned int size, unsigned int stream_id){ // this is useful for setting dev_res (interfaces) to 0 before calling the kernel
	err= cudaMemsetAsync(dev_array, 0, size * sizeof(unsigned int), stream[stream_id]);   

	if(err!=cudaSuccess){
		cudaCheckErrors("setMem fail");
	}
}


void GPU_matching::async_getResults(unsigned int * host_result, unsigned int * dev_result, unsigned int size, unsigned int stream_id){
	
	err=cudaMemcpyAsync(host_result, dev_result, size * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[stream_id]);
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

bool GPU_matching::runKernel(unsigned int * dev_array, uint16_t * dev_global_tiff, unsigned int * prefix_tiff_index, uint16_t * dev_query_tiff, unsigned int * dev_result, unsigned int size, unsigned int packets, unsigned int stream_id){
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
	myKernel_fast<<<gridsize, block,0 , stream[stream_id] >>> ((unsigned int*)dev_array, dev_global_tiff, prefix_tiff_index, dev_query_tiff, (unsigned int*)dev_result, size, packets, stream_id);
#else
	myKernel_minimal<<<gridsize, block,0 , stream[stream_id] >>> ((unsigned int*)dev_array, dev_global_tiff, prefix_tiff_index, dev_query_tiff, (unsigned int*)dev_result, size, packets, stream_id);
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
