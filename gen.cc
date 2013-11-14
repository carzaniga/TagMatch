#include <cstdlib>
#include <iostream>
#include <random>
#include <bitset>

#include "params.h"

#define APP 1
#define TREE 1

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
#if APP
    std::uniform_int_distribution<int> random_fsize(Amin-1,Amax);   //Amin-1 becuase we alway add 1
                                                                    //for the applicaiton
#else
    std::uniform_int_distribution<int> random_fsize(Amin,Amax); 
#endif

    std::uniform_int_distribution<int> random_app(0,4);
    std::uniform_int_distribution<int> random_tree(0,7);
    std::uniform_int_distribution<int> random_iff(0,100);


    std::bitset<M> f;
    int bsize;
    for(unsigned int i = 0; i < N; ++i) {
	f.reset();
	bsize = random_fsize(generator) * k;
	for(int j = 0; j < bsize; ++j)
	    f.set(random_position(generator));

#if APP
    //add application tag yt=0, tw=1, blog=2, del=3, bt=5
    
    switch ( random_app(generator) )
      {
        case 0:         //youtube
            f.set(191-9);
            f.set(191-15);
            f.set(191-81);
            f.set(191-95);
            f.set(191-97);
            f.set(191-104);
            f.set(191-139);
            break;
        case 1:         //twitter
            f.set(191-5);
            f.set(191-14);
            f.set(191-61);
            f.set(191-74);
            f.set(191-96);
            f.set(191-161);
            f.set(191-169);            
            break;
        case 2:         //blog
            f.set(191-1);
            f.set(191-34);
            f.set(191-44);
            f.set(191-72);
            f.set(191-149);
            f.set(191-151);
            f.set(191-183);
            break;
        case 3:         //deliciuos
            f.set(191-40);
            f.set(191-43);
            f.set(191-44);
            f.set(191-72);
            f.set(191-88);
            f.set(191-103);
            f.set(191-143);
            break;
        case 4:         //bit-torrent
            f.set(191-26);
            f.set(191-35);
            f.set(191-54);
            f.set(191-108);
            f.set(191-116);
            f.set(191-146);
            f.set(191-171);
            break;
      }
#endif

#if TREE
    std::cout << command << ' ' << random_tree(generator) << ' ' << random_iff(generator) << ' ';
#else
	std::cout << command << ' ' << tree_id << ' ' << if_id << ' ';
#endif
	for(int j = 0; j < M; ++j)
	    std::cout << (f[j] ? '1' : '0');
	std::cout << std::endl;
    }
}
