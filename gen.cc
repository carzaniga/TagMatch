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

    const char * command = "+";
    const char * if_id = "0";
    const char * tree_id = "0";
    unsigned seed = 0;
    

    switch(argc) {
    case 8:
	Amin = atoi(argv[7]);
    case 7:
	Amax = atoi(argv[6]);
    case 6:
	N = atoi(argv[5]);
    case 5:
	seed = atoi(argv[1]);
	command = argv[2];
	tree_id = argv[3];
	if_id = argv[4];
	break;
    default:
		std::cerr << "usage: " << argv[0] << " <seed> <command> <tree-id> <if-id> [N default=1000] [Amax default=10] [Amin default=1]" << std::endl;
	return 1;
    }

    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> random_position(0,M-1);
    std::uniform_int_distribution<int> random_fsize(Amin,Amax);

    std::bitset<M> f;
    int bsize;
    for(unsigned int i = 0; i < N; ++i) {
	f.reset();
	bsize = random_fsize(generator) * k;
	for(int j = 0; j < bsize; ++j)
	    f.set(random_position(generator));

	std::cout << command << ' ' << tree_id << ' ' << if_id << ' ';
	for(int j = 0; j < M; ++j)
	    std::cout << (f[j] ? '1' : '0');
	std::cout << std::endl;
    }
}
