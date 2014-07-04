#include "GPU_matching.h"
#include <chrono>
#include "predicate.hh"
#include <boost/lockfree/queue.hpp>

using namespace std;
using namespace std::chrono;

class main_GPU{
	public:
	unsigned int stream_array[STREAMS] ;
	GPU_matching gpu_matcher; 
//	static GPU_matching gpu_matcher;
		template<class T, unsigned long Q_SIZE = STREAMS>
		class BoostQueue {
			public:
				void
					push(T *x)
					{
						while (!q_.push(x))
							asm volatile("rep; nop" ::: "memory");
					}
				T *	pop()
				{
					T *x;
					while (!q_.pop(x))
						asm volatile("rep; nop" ::: "memory");
					return x;
				}
			private:
				boost::lockfree::queue<T *, boost::lockfree::capacity<Q_SIZE>> q_;
		};
		
		BoostQueue<unsigned int, 16> stream_queue;

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
		struct GPU_filter{
			//GPU_block_t has 32 bits.
			GPU_matching::GPU_block_t b[GPU_matching::B_COUNT];// = 6
			GPU_filter (){
				reset() ;
			}
			void reset (){
				for(unsigned int i=0; i< GPU_matching::B_COUNT; i++)
					b[i]= 0;
			}
			void set_all(){
				for(unsigned int i=0; i< GPU_matching::B_COUNT; i++)
					b[i]= ~0U; 
			}
			void flip(){
				for(unsigned int i=0; i<GPU_matching::B_COUNT; i++)
					b[i] = ~b[i] ;
			}
		};


		unsigned int ** host_fibs ;
		unsigned int ** dev_fibs ; // to store pointers to tables on GPU. 
		
		//uint16_t ** host_tiff ;
		//uint16_t ** dev_tiff ; 
		uint16_t * host_tiff ; // to store a big array of tree_iff pairs for all fib entries
		uint16_t * dev_tiff ; 

		unsigned int ** host_tiff_index ; //[#prefix][size_of_prefix]  // for each fib-entry, this stores an index to an entry in the host_tiff array. 
		unsigned int ** dev_tiff_index ;
		
		unsigned int ** host_queries ; // for each stream, we store queries of each batch here.
//		int ** dev_queries ; // dev_queries are stored in __const__ dev_message
//		// so here we don't need to have dev_queries. 

		uint16_t ** host_query_tiff ;// for each stream, we store tiff of each query in the batch here. 
		uint16_t ** dev_query_tiff ;

		GPU_matching::iff_result_t ** host_results ; // stream x [tiff x interfaces] 
		GPU_matching::iff_result_t ** dev_results ; // stream x [tiff x interfaces] 
//		int dev_results [STREAMS][PACKETS_BATCH_SIZE][INTERFACES] ; // stream x [msg x interfaces] 
//		int host_results [STREAMS][PACKETS_BATCH_SIZE][INTERFACES] ; // stream x [msg x interfaces] 

		unsigned int no_prefixes ; 

		unsigned int tiff_counter; 
		unsigned int * ti_sizes ;
		vector<unsigned int> size_of_prefixes ;

		GPU_filter assign(const string & s) ;
		//vector<filter> * read_tables(vector<filter_descr> * filters_descr, int no_prefixes);
		//void read_tables(vector<filter_descr> * filters_descr, int no_prefixes);
		void read_tables(vector<filter_descr> * filters_descr, vector<tree_interface_pair> ti_pairs);
		unsigned int init(vector<unsigned int> & sizes, unsigned int ti_counter);
		void move_to_GPU() ;
		void async_getResults(unsigned int size, unsigned int stream_id) ;
		void destroy_fibs() ;
		void match(unsigned int prefix_id, unsigned int no_packets, unsigned int stream_id );
		void async_copyToConstantMemory(unsigned int * packets, unsigned int no_packets, unsigned int stream_id);
};
