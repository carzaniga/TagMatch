#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <sstream>
//#include <iostream>
#include <cstdlib>
#include <ctime>
#include <bitset>
#include <vector>
#include <math.h>
#include <stdint.h>
#include "GPU_matching.h"

#define cudaCheckErrors(msg) \
	do { \
		cudaError_t __err = cudaGetLastError(); \
		if (__err != cudaSuccess) { \
			fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
					msg, cudaGetErrorString(__err), \
					__FILE__, __LINE__); \
			fprintf(stderr, "*** FAILED - ABORTING\n"); \
		} \
	} while (0)
cudaError_t err ; 

__constant__ unsigned int dev_message [STREAMS][PACKETS*6]; //[194];
__global__ void myKernel_minimal(int* data, int * result,int n, int packets , int stream_id) //, int t)
{
	//here I use packets, instead of PACKETS, because a matching queue can be sent to GPU even
	//when it is not full. (i.e if a timeout occures) 
	unsigned int id = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;
	if(id>=n)
		return;
	unsigned int d[6];
	id=id*6 ;
	for(int i=0;i<6; i++)
		d[i]=data[id+i];
// i can replace it with packets.... 
	for(int j=0; j<packets*6/*PACKETS*/ ; j+=6){
		if(d[0] & ~dev_message[stream_id][j]!=0)
			continue;
		if(d[1] & ~dev_message[stream_id][j+1]!=0)
			continue;
		if(d[2] & ~dev_message[stream_id][j+2]!=0)
			continue;
		if(d[3] & ~dev_message[stream_id][j+3]!=0)
			continue;
		if(d[4] & ~dev_message[stream_id][j+4]!=0)
			continue;
		if(d[5] & ~dev_message[stream_id][j+5]!=0)
			continue;
		result[j/6]+=d[0];
	}
}
__global__ void myKernel_fast(int* data, int * result,int n, int packets, int stream_id)
{
////	unsigned int id = (/*6 * */ blockDim.x * blockDim.y * blockIdx.x) + (6 * blockDim.x * threadIdx.y) + threadIdx.x;
	unsigned int id = (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;
	//int id = blockDim.x * blockIdx.x+ threadIdx.x;

	if(id>=n)
		return;
	unsigned int d[6];
	d[0]=data[id];
	d[1]=data[id+32];
	d[2]=data[id+64];
	d[3]=data[id+96];
	d[4]=data[id+128];
	d[5]=data[id+160];
	for(int j=0; j<packets*6 ; j+=6){
		if(d[0] | dev_message[stream_id][j]!=0xFFFFFFFF)
			continue;
		if(d[1] | dev_message[stream_id][j+1]!=0xFFFFFFFF)
			continue;
		if(d[2] | dev_message[stream_id][j+2]!=0xFFFFFFFF)
			continue;
		if(d[3] | dev_message[stream_id][j+3]!=0xFFFFFFFF)
			continue;
		if(d[4] | dev_message[stream_id][j+4]!=0xFFFFFFFF)
			continue;
		if(d[5] | dev_message[stream_id][j+5]!=0xFFFFFFFF)
			continue;
		result[j/6]+=d[0];
	}
return;
#if 0	
	unsigned int B=1 ;
	unsigned int B2=1 ;
	//#pragma unroll
	for(int j=0; j<PACKETS ; j++){
		// I should think more about this. becasue 
		// I am just oring everthing :p something is not right.
		// or not ( since messages are stored as the ~ of original ones, this make sense);
//		B= (d[0] | dev_message[j*6]);
//		B2=(d[1] | dev_message[j*6+1]);
//		B&=(d[2] | dev_message[j*6+2]);
//		B2&=(d[3] | dev_message[j*6+3]);
//		B&=(d[4] | dev_message[j*6+4]);
//		B2&=(d[5] | dev_message[j*6+5]);
//		if(B==0xFFFFFFFF && B2==0xFFFFFFFF)
//			result[j]=d[0];
		B= (d[0] | dev_message[j*6]);
		B&=(d[1] | dev_message[j*6+1]);
		B&=(d[2] | dev_message[j*6+2]);
		B&=(d[3] | dev_message[j*6+3]);
		B&=(d[4] | dev_message[j*6+4]);
		B&=(d[5] | dev_message[j*6+5]);
//		if(b[0]==b[1]==b[2]==b[3]==b[4]==b[5]==0)
//			result[j]=d[0]; // just do something here!
//		if(B==0)
//			result[j]=B;
	}
#endif
}

cudaStream_t stream [STREAMS] ;

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
//		cout<<free*1.0/(1024*1024)<<" "<<total*1.0/(1024*1024)<<" mega byte\n"<<endl;
		printf("cudaMemGetInfo returned the error: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	printf("free= %f total_Mem=%f\n",(long unsigned)free*1.0/(1024*1024),(long unsigned)total*1.0/(1024*1024));

}
bool GPU_matching::copyMSG(unsigned int * host_message, int packets , int stream_id){
	cudaError_t err =cudaMemcpyToSymbol(&dev_message[stream_id][0], host_message, size_t(6*packets)*sizeof(unsigned int));
	if (err != cudaSuccess)
		return false;
//	printf("constant allocation done");
	return true;
}

void GPU_matching::deleteArray(int * dev_array){
	//cudaFree(dev_array);
}

int * GPU_matching::fillTable(int * host_array, int size){
//	printf("/");
	int *dev_array = 0;
	err=cudaMalloc((void**)&dev_array, size * sizeof(int)); 
	if(err!=cudaSuccess){
		cudaCheckErrors("allocation fail");
		cudaFree(dev_array) ;
		cudaDeviceReset() ;
		exit(0);
		return NULL;
	}
	
	err=cudaMemcpy(dev_array, host_array, size * sizeof(int),cudaMemcpyHostToDevice);
	if(err!=cudaSuccess){
		cudaCheckErrors("memCpy fail!!");
		cudaFree(dev_array) ;
		return NULL;
	}
//	printf("-\n");
	return dev_array;
}

int * GPU_matching::allocZeroes(int size){ // this is useful for setting dev_res (interfaces) to 0 before calling the kernel
	int *dev_array = 0;
	err=cudaMalloc((void**)&dev_array, size * sizeof(int)); 
	if(err!=cudaSuccess)
		return NULL;
	err=cudaMemset(dev_array, 0, size * sizeof(int));   

	cudaCheckErrors("setMem fail");
	if(err!=cudaSuccess)
		return NULL;
//	printf("+++");
	return dev_array; 
}
int * GPU_matching::getResults(int * host_result, int * dev_result, int size){
	
	cudaMemcpy(host_result, dev_result, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy 2 fail");
	return host_result ;
//	int sum =0;
//	for(int i=0;i <PACKETS ; i++)
//	{
//		sum+=host_res[i] ;	
//	}

}


struct GPU_matching::stream_packets {
	cudaStream_t st;
	int * packets;
	stream_packets (cudaStream_t st_, int * packets_ ){
		st = st_;
		packets = packets_ ;
	}
};

void GPU_matching::init_streams(){
	int n= STREAMS;
	for(int i=0; i<n ; i++){
		cudaStreamCreate ( &stream[i]) ;	
	}

}

bool GPU_matching::runKernel(int * dev_array, int * dev_result, int size, int packets, int stream_id){
//	int gridsize = 1;
//	// Work out how many blocks of size 1024 are required to perform all of nCombinations
//	for(unsigned int nsize = gridsize * BLOCK_SIZE; nsize < size;//nCombinations;
//			gridsize++, nsize = gridsize * BLOCK_SIZE)
//		;
//
//	cudaStream_t stream1 ;
//	cudaStreamCreate(&stream1);
//	cudaCheckErrors("stream creation failed.");
	//printf("steam_id= %d",stream1);
	dim3 grid, block;
	block = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);
	int b=GPU_BLOCK_SIZE;
	int gridsize = size/b;
//	printf("gridsize = %d, blocksize = %d", gridsize, b);
	//printf("bla gridsize= %i\n", gridsize);  
	grid = dim3(gridsize);
#if GPU_FAST
//	printf(" alg: Fast\n");
	myKernel_fast<<<gridsize, block,0 , stream[stream_id] >>> ((int*)dev_array, (int*)dev_result, size, packets, stream_id);
#else
//	printf(" alg: simple\n");
	myKernel_minimal<<<gridsize, block, 0, stream[stream_id] >>> ((int*)dev_array, (int*)dev_result, size, packets, stream_id);
#endif
	cudaCheckErrors("kernel fail");
	// Wait for synchronize
//	cudaDeviceSynchronize();
	cudaStreamSynchronize(stream[stream_id]) ;
	cudaCheckErrors("sync fail");

//we dont destroy streams anymore
//	cudaStreamDestroy(stream[stream_id]);
//	cudaCheckErrors("destruction of stream failed");
	return true;
}
void GPU_matching::finish(){
	//do something;
	cudaDeviceReset() ;
	return;
}
void GPU_matching::releaseMem(int * p){
	cudaFree(p); 
	if (err != cudaSuccess){
		cudaCheckErrors("couldn't free memory on GPU");  
//	printf("couldn't free memory on GPU"); 

	}
}
