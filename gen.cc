#include <cstdlib>
#include <iostream>
#include <random>
#include <bitset>

#include "params.h"

int main(int argc, char * argv[]) {
    const int M = BLOOM_FILTER_SIZE;
    const int k = BLOOM_FILTER_K;

    unsigned N = 1000;
    unsigned Amin = 1;
    unsigned Amax = 10;

    unsigned if_id = 0;
    unsigned tree_id = 0;
    unsigned seed = 0;
    

    switch(argc) {
    case 7:
	Amin = atoi(argv[6]);
    case 6:
	Amax = atoi(argv[5]);
    case 5:
	N = atoi(argv[4]);
    case 4:
	seed = atoi(argv[1]);
	tree_id = atoi(argv[2]);
	if_id = atoi(argv[3]);
	break;
    default:
	std::cerr << "usage: " << argv[0] << " <seed> <tree-id> <if-id> [N default=1000] [Amax default=10] [Amin default=1]" << std::endl;
	return 1;
    }

    std::default_random_engine generator;
    std::uniform_int_distribution<int> random_position(0,M-1);
    std::uniform_int_distribution<int> random_fsize(Amin,Amax);

    std::bitset<M> f;
    int bsize;
    for(int i = 0; i < N; ++i) {
	f.reset();
	bsize = random_fsize(generator) * k;
	for(int j = 0; j < bsize; ++j)
	    f.set(random_position(generator));

	std::cout << tree_id << ' ' << if_id << ' ';
	for(int j = 0; j < M; ++j)
	    std::cout << f[j] ? '1' : '0';
	std::cout << std::endl;
    }
}
