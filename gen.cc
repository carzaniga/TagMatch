#include <cstdlib>
#include <iostream>
#include <random>
#include <bitset>

#include "params.h"

int main(int argc, char * argv[]) {
    const int M = BLOOM_FILTER_SIZE;
    const int k = BLOOM_FILTER_K;

    int N = 10000000;

    switch(argc) {
    case 2:
	N = atoi(argv[1]);
    case 1:
	break;
    default:
	std::cerr << "usage: " << argv[0] << " <N>" << std::endl;
	return 1;
    }

    std::default_random_engine generator;
    std::uniform_int_distribution<int> random_position(0,M-1);
    std::uniform_int_distribution<int> random_fsize(1,10);

    std::bitset<M> f;
    int bsize;
    for(int i = 0; i < N; ++i) {
	f.reset();
	bsize = random_fsize(generator) * k;
	for(int j = 0; j < bsize; ++j)
	    f.set(random_position(generator));

	for(int j = 0; j < M; ++j)
	    std::cout << f[j] ? '1' : '0';
	std::cout << std::endl;
    }
}
