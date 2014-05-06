#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <bitset>
#include <vector>
#include <math.h>
#include <stdint.h>
#include "GPU_matching.h"
#include <chrono>

using namespace std;
using namespace std::chrono;

class main_GPU{
	public:
		struct filter_descr {
			string filter;
			unsigned int ti_pairs_begin;
			unsigned int ti_pairs_end;

			filter_descr(const std::string & f, unsigned int b, unsigned int e)
				: filter(f), ti_pairs_begin(b), ti_pairs_end(e) {};

			filter_descr(const filter_descr & d)
				: filter(d.filter), ti_pairs_begin(d.ti_pairs_begin), ti_pairs_end(d.ti_pairs_end) {};

			filter_descr & operator = (const filter_descr & d) {
				filter = d.filter;
				ti_pairs_begin = d.ti_pairs_begin;
				ti_pairs_end = d.ti_pairs_end ;
				return *this;
			}
		};
		struct filter{
			GPU_matching::block_t b[GPU_matching::B_COUNT];
			filter (){
				reset() ;
			}
			void reset (){
				b[0]=0; b[1]=0 ; b[2]=0; b[3]=0; b[4]=0; b[5]=0; 
			}
			void set_all(){
				b[0]=0xFFFFFFFF; b[1]=0xFFFFFFFF ; b[2]=0xFFFFFFFF; b[3]=0xFFFFFFFF; b[4]=0xFFFFFFFF; b[5]=0xFFFFFFFF; 
			}
			void flip(){
				for(int i=0; i<6; i++)
					b[i] = ~b[i] ;
			}
		};


		//	struct filter ;
		//struct filter_descr; 
		int ** host_fibs ;
		int ** dev_fibs ; // to store pointers to tables on GPU. 
		int ** dev_results ;
		int no_prefixes ; 
		vector<int> size_of_prefixes ;

		filter assign(const string & s) ;
		//vector<filter> * read_tables(vector<filter_descr> * filters_descr, int no_prefixes);
		//void read_tables(vector<filter_descr> * filters_descr, int no_prefixes);
		void read_tables(vector<filter_descr> * filters_descr);
		static GPU_matching gpu_matcher;
		int init(vector<int> & sizes);
		//int main_GPU(){} ;
		void move_to_GPU() ;
		void allocate_result_on_GPU(int size) ;
		void destroy_fibs() ;
		void match(int prefix_id, int no_packets );
};

//struct main_GPU::filter_descr {

