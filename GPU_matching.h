#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <sstream>
//#include <iostream>
#include <cstdlib>
#include <stdint.h>

using namespace std;
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32
#define GPU_BLOCK_SIZE BLOCK_DIM_X * BLOCK_DIM_Y // Statc block size of 32*32 (1024)0
#define PACKETS 100 
#define GPU_FAST 1 


class GPU_matching{
public:
	static const int B_SIZE = 32;
	static const int Size= 192;
	typedef uint32_t block_t;
	static const int B_COUNT = Size / B_SIZE;
	//	__constant__ unsigned int dev_message [PACKETS*6]; //[194];
	void initialize();

	void memInfo();
	bool copyMSG(unsigned int * host_message, int packets);
	void deleteArray(int * dev_array);
	int * fillTable(int * host_array, int size);
	int * allocZeroes(int size);
	int * getResults(int * host_result, int * dev_result, int size);
	bool runKernel(int * dev_array, int * dev_result, int size, int packets, int stream_id);
	void finish();
	void releaseMem(int * p);
private:
//	__global__ void myKernel_minimal(int* data, int * result,int n, int packets);
};
