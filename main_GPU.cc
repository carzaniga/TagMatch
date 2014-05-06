//#include <stdio.h>
//#include <fstream>
//#include <stdlib.h>
//#include <sstream>
//#include <iostream>
//#include <cstdlib>
//#include <ctime>
//#include <bitset>
//#include <vector>
//#include <math.h>
//#include <stdint.h>
//#include "GPU_matching.h"
//#include <chrono>
#include "main_GPU.h"
//using namespace std;
//using namespace std::chrono;


//main_GPU::struct filter{
//	GPU_matching::block_t b[GPU_matching::B_COUNT];
//	 filter (){
//	 	reset() ;
//	 }
//	 void reset (){
//		 b[0]=0; b[1]=0 ; b[2]=0; b[3]=0; b[4]=0; b[5]=0; 
//	 }
//	 void set_all(){
//		 b[0]=0xFFFFFFFF; b[1]=0xFFFFFFFF ; b[2]=0xFFFFFFFF; b[3]=0xFFFFFFFF; b[4]=0xFFFFFFFF; b[5]=0xFFFFFFFF; 
//	 }
//	 void flip(){
//	 	for(int i=0; i<6; i++)
//			b[i] = ~b[i] ;
//	 }
//};


main_GPU::filter main_GPU::assign(const string & s) {
	main_GPU::filter f;
	std::string::const_iterator si = s.begin();
	for(int i = GPU_matching::B_COUNT-1; i >= 0; --i) {
//		f.b[i]=0;
		for(GPU_matching::block_t mask = (1U << 31); mask != 0; mask >>= 1) {
			if (si != s.end()) {
				if (*si == '1')
					f.b[i] |= mask;
				++si;
			} else {
				return f;
			}
		}
	}
	return f;
}
//void main_GPU::filp(string &s){
//	for(int i=0; i<s.length(); i++)
//		if(s[i]=='1')
//			s[i]='0';
//		else
//			s[i]='1';
//}

//vector<filter>* main_GPU::read_tables(vector<filter_descr> * filters_descr, int no_prefixes){
void main_GPU::read_tables(vector<filter_descr> * filters_descr){//, int no_prefixes){
//	cout<<"entering this fucking function" << endl; 
	vector<filter> * filters= new vector<filter>[no_prefixes];
	filter f;
	//long count=0;
	string filter_string;
	for(int prefix_id=0; prefix_id < no_prefixes; prefix_id++){
		for( unsigned int i=0; i< filters_descr[prefix_id].size(); i++) {
			filter_string = filters_descr[prefix_id].at(i).filter; 			
			f=assign(filter_string);
#if GPU_FAST
			f.flip() ;
#endif
			filters[prefix_id].push_back(f) ;			
		}
	}
//	while(filters.size()%32!=0)
//	{
//		filter f2;
//		f2.set_all() ;
//		filters.push_back(f2);	
//	}
#if GPU_FAST
	for(int prefix_id=0; prefix_id < no_prefixes; prefix_id++){
		for(unsigned int i=0; i< filters[prefix_id].size()/(32*6) ; i++){
			for(int j=0; j< 32 ; j++){
				host_fibs[prefix_id][i*192+j+0]  =filters[prefix_id][i*32+j].b[0]; 
				host_fibs[prefix_id][i*192+j+32] =filters[prefix_id][i*32+j].b[1]; 
				host_fibs[prefix_id][i*192+j+64] =filters[prefix_id][i*32+j].b[2]; 
				host_fibs[prefix_id][i*192+j+96] =filters[prefix_id][i*32+j].b[3]; 
				host_fibs[prefix_id][i*192+j+128]=filters[prefix_id][i*32+j].b[4]; 
				host_fibs[prefix_id][i*192+j+160]=filters[prefix_id][i*32+j].b[5]; 
			}
		}
	}
#else
	// I could've done this in the above code when I was filling "filters" ...
	for(int prefix_id=0; prefix_id < no_prefixes; prefix_id++){
		for(int i=0; i< filters[prefix_id].size() ; i++){
			for(int j=0; j< 6 ; j++){
				host_fibs[prefix_id][i*6+j]  =filters[prefix_id][i].b[j]; 
			}
		}
	}
#endif

	cout<<"done reading filters" << endl; 
}

void main_GPU::move_to_GPU(){
	int sum=0;
	int temp=0;
	for(int prefix_id=0; prefix_id < no_prefixes; prefix_id++){
		temp= 24*size_of_prefixes[prefix_id]/1024 ;
//		cout<< temp <<"kb, sofar allocated "<< sum/1024 << "mb"<< endl;
		dev_fibs[prefix_id] = gpu_matcher.fillTable( host_fibs[prefix_id], 
								size_of_prefixes[prefix_id] * 6 ) ;// filters.size()*6);
		sum+= temp ;//size_of_prefixes[prefix_id] * 6 ; 
//		cout<<"done for pid=" <<  prefix_id << " sum sofar=" << sum <<endl ;
	}
	cout<< "successfully allocated "<< sum/1024 << "mb"<< endl;
}
void main_GPU::allocate_result_on_GPU(int size){
	cout<<"#prefixes= " << no_prefixes << endl  ; 
	for(int prefix_id=0; prefix_id < no_prefixes; prefix_id++){
		dev_results[prefix_id] = gpu_matcher.allocZeroes(size* 6 ); // PACKETS*6 ) ; 
//		cout<<"done " << prefix_id << endl  ; 
	}
}

GPU_matching main_GPU::gpu_matcher; 

void main_GPU::destroy_fibs(){
	for(int i=0; i<no_prefixes; i++)
		gpu_matcher.releaseMem(dev_fibs[i]) ;
	cout<<"reseting device memory";
	gpu_matcher.finish() ;
	cout<< "... done " <<endl;
}
void main_GPU::match(int prefix_id, int no_packets ){

		gpu_matcher.runKernel( dev_fibs[prefix_id], dev_results[prefix_id], size_of_prefixes[prefix_id], no_packets, prefix_id) ; 
}

int main_GPU::init(vector<int> & size_of_prefixes_){//int argc, char *argv[]) 
	size_of_prefixes = size_of_prefixes_ ;
	no_prefixes= size_of_prefixes.size() ;
	gpu_matcher.initialize();
	gpu_matcher.memInfo();
	host_fibs = new int *[size_of_prefixes.size()] ;
	dev_fibs = new int *[size_of_prefixes.size()] ;
	dev_results = new int *[size_of_prefixes.size()] ;

	for(unsigned int i=0; i<size_of_prefixes.size(); i++){
		host_fibs[i]=new int[size_of_prefixes[i] * 6] ;
	}
	cout<<"init done" << endl;	
	return 0;
////	vector<filter> filters = read_table() ;
//	srand(0);//time(NULL));
//	int packets= PACKETS;
//	unsigned int * host_message= new unsigned int[6 * packets];
//	for (int i=0; i<GPU_matching::B_COUNT * packets ; i++){ 
//		host_message[i]= rand();
//	}
//	gpu_matcher.copyMSG(host_message,packets) ;
//
//	int * host_res= new int[packets];
//	int * host_res2= new int[packets];
//
//	for(int j=0; j<packets; j++)
//		host_res[j]=0;
//
//	printf("n = %lu filters, total=%i integers", filters.size(),nCombinations) ;
//	cout<<" require: "<<nCombinations*sizeof(int)*1.0/(1024*1024) <<" mega byte"<<endl;
//	int* host_array = new int[nCombinations];
//	long startTime=clock();
//	high_resolution_clock::time_point start, stop; 
//	int * dev_array = gpu_matcher.fillTable(host_fib,filters.size()*6);
//	int * dev_result = gpu_matcher.allocZeroes(packets);  
//	long finishTime=clock();
//	int count=1;
//	cout<<"alocation and memCpy: "<<(float)(finishTime-startTime)*1000*1000/CLOCKS_PER_SEC<<" us"<<endl;
//	startTime=clock();
//	start = high_resolution_clock::now();
//
//	gpu_matcher.runKernel(dev_array, dev_result, filters.size(), packets); 
//
//	stop = high_resolution_clock::now();
//	nanoseconds ns = duration_cast<nanoseconds>(stop - start);
//	cout<< "kernel took: "<< ns.count()/packets << "ns per message"<<endl;
//	
//	finishTime=clock();
//	cout<<"execution of kernel took: "<<((float)(finishTime-startTime)*1000*1000)/(packets*count*CLOCKS_PER_SEC)<<" us per message"<<endl;
//
//	cout<<"done"<<endl;
//	delete[] host_array;
//	delete[] host_fib;
//	gpu_matcher.finish() ;
//	return 0;
}
