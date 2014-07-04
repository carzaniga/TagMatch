#include "main_GPU.h"

main_GPU::GPU_filter main_GPU::assign(const string & s) {
	main_GPU::GPU_filter f;
	std::string::const_iterator si = s.begin();
	for(unsigned int i = GPU_matching::B_COUNT-1; i >= 0; --i) {
		for(GPU_matching::GPU_block_t mask = (1U << 31); mask != 0; mask >>= 1) {
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

void main_GPU::read_tables(vector<filter_descr> * filters_descr, vector<tree_interface_pair> ti_pairs){//, int no_prefixes){
	// here I change a vector of filter_descr to a vector of filters.
	// also I put tree interface pairs into a new data structure.
	vector<GPU_filter> * filters= new vector<GPU_filter>[no_prefixes];
	GPU_filter f;
	string filter_string;
	tiff_counter= 0; 
	for(unsigned int prefix_id=0; prefix_id < no_prefixes; prefix_id++){

		for( unsigned int i=0; i< filters_descr[prefix_id].size(); i++) {
			filter_descr fsc = filters_descr[prefix_id].at(i) ;

			filter_string = fsc.filter; 			
			f=assign(filter_string);
#if GPU_FAST
			f.flip() ;
#endif
			filters[prefix_id].push_back(f) ;			
			
			host_tiff_index[prefix_id][i] = tiff_counter; 

			host_tiff[tiff_counter++] =  fsc.ti_pairs_end - fsc.ti_pairs_begin ; // this stores the size of tiff for that specific fib entry.
			for( unsigned int t = fsc.ti_pairs_begin; t != fsc.ti_pairs_end; t++){
				host_tiff[tiff_counter++] = ti_pairs.at(t).get() ; 
			}
		}
	}

#if GPU_FAST
	for(unsigned int prefix_id=0; prefix_id < no_prefixes; prefix_id++){
		for(unsigned int i=0; i< filters[prefix_id].size()/(32) ; i++){
			for(unsigned int j=0; j< 32 ; j++){
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
	for(unsigned int prefix_id=0; prefix_id < no_prefixes; prefix_id++){
		for(unsigned int i=0; i< filters[prefix_id].size() ; i++){
			for(unsigned int j=0; j< 6 ; j++){
				unsigned int ttt=filters[prefix_id][i].b[j];  
				host_fibs[prefix_id][i*6+j]  =ttt;
			}
		}
	}
#endif
}

void main_GPU::move_to_GPU(){
	unsigned int sum=0;
	unsigned int temp=0;
//	gpu_matcher.memInfo();
	for(unsigned int prefix_id=0; prefix_id < no_prefixes; prefix_id++){
		temp= 24*size_of_prefixes[prefix_id] ;
		sum+= temp ;
		cout<< "\r ("<< temp/1024 <<"kb, "<< sum/(1024*1024) << "mb)";//<< endl;
		dev_fibs[prefix_id] = gpu_matcher.fillTable( host_fibs[prefix_id], (size_of_prefixes[prefix_id]) * 6 ) ;
		
		dev_tiff_index[prefix_id] = gpu_matcher.fillTable( host_tiff_index[prefix_id], size_of_prefixes[prefix_id] ) ;
		temp= 4*size_of_prefixes[prefix_id] ;  
		sum+= temp ;
		cout<< " ("<< temp/1024 <<"kb, "<< sum/(1024*1024) << "mb)";//<< endl;
	}
	cout<<endl ;
	cout<< "tiff_counter= " << tiff_counter << " = " << tiff_counter*2/(1024*1024) <<"mb" <<endl ;
	dev_tiff = gpu_matcher.fillTiff( host_tiff, tiff_counter ) ;
	gpu_matcher.memInfo();
	sum+= tiff_counter*2 ;
}


void main_GPU::async_getResults(unsigned int size, unsigned int stream_id){
	gpu_matcher.async_getResults( host_results[stream_id], dev_results[stream_id], size*INTERFACES, stream_id) ;
}


void main_GPU::async_copyToConstantMemory(unsigned int * queries, unsigned int no_packets, unsigned int stream_id){
	gpu_matcher.async_copyMSG( queries, no_packets, stream_id) ;	
}

void main_GPU::destroy_fibs(){
	for(unsigned int i=0; i<no_prefixes; i++)
		gpu_matcher.releaseMem(dev_fibs[i]) ;
	gpu_matcher.finish() ;
}

void main_GPU::match(unsigned int prefix_id, unsigned int no_packets, unsigned int stream_id){
//	i am here. setting the kernerl parameters!
	gpu_matcher.runKernel( dev_fibs[prefix_id], dev_tiff, dev_tiff_index[prefix_id], dev_query_tiff[stream_id], dev_results[stream_id], size_of_prefixes[prefix_id], no_packets, stream_id) ; 
}

unsigned int main_GPU::init(vector<unsigned int> & size_of_prefixes_, unsigned int ti_counter){
	size_of_prefixes = size_of_prefixes_ ;

	no_prefixes= size_of_prefixes.size() ;
	gpu_matcher.initialize();
	gpu_matcher.memInfo();
	host_fibs = new unsigned int *[size_of_prefixes.size()] ;
	dev_fibs = new unsigned int *[size_of_prefixes.size()] ;

	host_results = new GPU_matching::iff_result_t *[STREAMS] ;
	dev_results = new GPU_matching::iff_result_t *[STREAMS] ;
	
	host_tiff_index = new unsigned int *[size_of_prefixes.size()] ;
	dev_tiff_index = new unsigned int *[size_of_prefixes.size()] ;
	
	host_query_tiff = new uint16_t *[STREAMS] ;
	dev_query_tiff = new uint16_t *[STREAMS] ; 	

	host_queries = new unsigned int *[STREAMS] ;
		
	for(unsigned int prefix_id=0; prefix_id < size_of_prefixes.size(); prefix_id++){
		host_fibs[prefix_id] = new unsigned int[(size_of_prefixes[prefix_id]) * 6] ;
		host_tiff_index[prefix_id] = new unsigned int[size_of_prefixes[prefix_id]] ;
	}
	cout <<"#ti_pairs = "<< ti_counter << endl; 
	host_tiff = new uint16_t [ti_counter] ;

	for(unsigned int stream_id=0; stream_id< STREAMS; stream_id++){
		host_query_tiff[stream_id] = new uint16_t[PACKETS_BATCH_SIZE] ;
		host_queries[stream_id] = new unsigned int[PACKETS_BATCH_SIZE*6] ;

		dev_query_tiff[stream_id] = gpu_matcher.sync_alloc_tiff(PACKETS_BATCH_SIZE) ;

		host_results[stream_id] = new GPU_matching::iff_result_t[PACKETS_BATCH_SIZE*INTERFACES] ;
		dev_results[stream_id] = gpu_matcher.allocZeroes(PACKETS_BATCH_SIZE*INTERFACES) ;
			
		stream_array[stream_id] = stream_id ;
		stream_queue.push(&stream_array[stream_id]) ;
	}
	return 0;
}
