#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <random>
#include <bitset>

#include "predicate.hh"

const unsigned int M = filter_t::WIDTH;

void do_gen(unsigned int seed, 
			const char * command,
			unsigned int if_id,
			unsigned int tree_id,
			unsigned int N,
			unsigned int k,
			unsigned int Amin,
			unsigned int Amax) {
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<unsigned int> random_position(0,M-1);
    std::uniform_int_distribution<unsigned int> random_fsize(Amin,Amax);

    std::bitset<M> f;
    unsigned int bsize;
    for(unsigned int i = 0; i < N; ++i) {
		f.reset();
		bsize = random_fsize(generator) * k;
		for(unsigned int j = 0; j < bsize; ++j)
			f.set(random_position(generator));

		std::cout << command << ' ' << tree_id << ' ' << if_id << ' ';
		for(unsigned int j = 0; j < M; ++j)
			std::cout << (f[j] ? '1' : '0');
		std::cout << std::endl;
    }
}

int main(int argc, char * argv[]) {

    unsigned int N = 1000;
    unsigned int k = 7;
    unsigned int Amin = 1;
    unsigned int Amax = 10;

    unsigned int if_id = 0;
    unsigned int tree_id = 0;
    unsigned int seed = 0;
    
	bool done = false;

	for(int i = 1; i < argc; ++i) {
		if (sscanf(argv[i],"k=%u", &k))
			continue;

		switch (sscanf(argv[i],"a=%u,%u", &Amin, &Amax)) {
		case 1: Amax=Amin;
		case 2: continue;
		default: break;
		}

		if (sscanf(argv[i], "n=%u", &N) || sscanf(argv[i], "N=%u", &N))
			continue;

		if (sscanf(argv[i], "s=%u", &seed) || sscanf(argv[i], "seed=%u", &seed))
			continue;

		if (sscanf(argv[i], "i=%u", &if_id) || sscanf(argv[i], "interface=%u", &if_id))
			continue;

		if (sscanf(argv[i], "t=%u", &tree_id) || sscanf(argv[i], "tree=%u", &tree_id))
			continue;

		do_gen(seed, argv[i], if_id, tree_id, N, k, Amin, Amax);
		done = true;
	}

	if (!done)
		do_gen(seed, "+", if_id, tree_id, N, k, Amin, Amax);

	return 0;
}
