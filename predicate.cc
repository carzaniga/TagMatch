#include <vector>
#include <set>
#include <iostream>
#include <cstring>
#include <sstream>

#include "predicate.h"
#include "allocator.h"
#include "timing.h"

siena::FTAllocator Mem;
siena::Timer match_timer;

class TreeIffPair { 
public:
    vector<int> treeIff[8];
    
    void addTreeIff(int tree, int iff){
        treeIff[tree].push_back(iff);
    }
    
    string print() {
        stringstream out;
        for(int i=0; i<8;++i){
            out << " tree " << i << " : " ;
            for(int j=0; j<treeIff[i].size(); ++j){
                out << " iff " << treeIff[i][j] << " ";
            }
        }
        out<< endl;
        return out.str();
    }
};

class node {
public:
    filter::pos_t pos;
    unsigned char treeMask;
    bool set;
    node * t;
    node * f;
    TreeIffPair * ti;
    

    node(filter::pos_t p): pos(p), t(0), f(0), set(false), treeMask(0), ti(NULL) {};

   void addIff(int tree, int iff){
        if (ti==NULL){
            ti = new (Mem) TreeIffPair();
        }
        ti->addTreeIff(tree,iff);
    }
    
    bool matchTreeMask(int tree) const{
        unsigned char tmp = 0;
        tmp |= 1 << tree;
        return ((treeMask & tmp)==tmp);
    }
    
    void setMask (int tree){
        treeMask |= 1 << tree;
    }
    
};

void predicate::add_filter(const filter & f, int tree, int iff) {
    node ** np = &root;
    filter::const_iterator fi = f.begin();
    
    node * last; //last node visited
    while (fi != f.end()) {
	if (*np == 0) {
	    *np = new (Mem) node(*fi);
	    last = *np;
	    np = &((*np)->t);
	    ++fi;
	} else if ((*np)->pos < *fi) {
	    last=*np;
	    np = &((*np)->f);
	} else if ((*np)->pos > *fi) {
	    node * tmp = *np;
	    *np = new (Mem) node(*fi);
	    last=*np;
	    (*np)->f = tmp;
	    np = &((*np)->t);
	    ++fi;
	} else {
	    last=*np;
	    np = &((*np)->t);
	    ++fi;
	}
	last->setMask(tree);
    }
    last->addIff(tree,iff);
    last->set= true;
}


std::set<int> predicate::match_result;


void match(const node *n, filter::const_iterator fi, filter::const_iterator end, int tree) {
    while (fi != end && n != 0 && n->matchTreeMask(tree)) {
        if(n->pos==*fi){
            if (n->set) {
                const std::vector<int> & ifs = n->ti->treeIff[tree];
                for(std::vector<int>::const_iterator ii = ifs.begin(); ii != ifs.end(); ++ii) {
                    predicate::match_result.insert(*ii);
                }
            }
            ++fi;
            match(n->f,fi,end,tree);
            n = n->t;		// equivalent to recursion: match(n->t,fi,end,tree);
        } else if (n->pos > *fi) {
            ++fi;		// equivalent to recursion: match(n,++fi,end,tree);
        } else {//(n->pos < *fi)
            n = n->f;		// equivalent to recursion: match(n->f,fi,end,tree);
        }
    }
}



#define TAIL_RECURSIVE_IS_ITERATIVE

bool suffix_contains_subset(bool prefix_is_good, const node * n, 
			    filter::const_iterator fi, filter::const_iterator end) {
#ifdef TAIL_RECURSIVE_IS_ITERATIVE
    while(n != 0) {
	if (fi == end) {
	    return false;
	} else if (n->pos > *fi) {
	    ++fi; 
	} else if (n->pos < *fi) {
	    prefix_is_good = false;
	    n = n->f;
	} else if (suffix_contains_subset(true, n->t, ++fi, end)) {
	    return true;
	} else {
	    prefix_is_good = false;
	    n = n->f;
	}
    }
    return prefix_is_good;
#else
     if (n == 0)
 	return prefix_is_good;
     if (fi == end)
 	return false;
     if (n->pos > *fi) 
 	return suffix_contains_subset(prefix_is_good, n, ++fi, end);
     if (n->pos < *fi)
 	return suffix_contains_subset(false, n->f, fi, end);
     ++fi;
     return suffix_contains_subset(true, n->t, fi, end)
 	|| suffix_contains_subset(false, n->f, fi, end);
#endif
}

bool predicate::contains_subset(const filter & f) const {
    return suffix_contains_subset(false, root, f.begin(), f.end());
}

void predicate::findMatch(const filter & f, int tree) const {
    predicate::match_result.clear();
    match(root,f.begin(),f.end(),tree);
    // DO SOMETHING WITH THE MATCH RESULT
}

int visit (const node * n) {
    //printf("node %u treeMask %u set %s\n",(*n).pos,(*n).treeMask,(*n).set?"true":"false");
    int count=1;
    if(n->t!=NULL)
        count+=visit(n->t);
    if(n->f!=NULL)
        count+=visit(n->f);
    return count;
}

int countIff(const node * n){
    int iff=0;
    if(n->ti!=NULL){
        for(int i=0; i<8; i++){
            iff+=n->ti->treeIff[i].size();
        }
    }
    if(n->t!=NULL)
        iff+=countIff(n->t);
    if(n->f!=NULL)
        iff+=countIff(n->f);
    return iff;
}

void predicate::treevisit() const{
    int count = visit(root);
    printf("number of nodes %u\n",count);
    int iff = countIff(root);
    printf("number of iff %u\n",iff);
}

static void print_r(ostream & os, string & prefix, const node * n) {
    if (n == 0) {
	os << prefix;
#if 1
	for(filter::pos_t p = prefix.size(); p < filter::FILTER_SIZE; ++p) {
	    os << '0';
	}
#endif
	os << std::endl;
	return;
    } else {
	const filter::pos_t cur_pos = prefix.size();
	prefix.append(n->pos - cur_pos, '0');
	prefix.push_back('1');
	print_r(os, prefix, n->t);
	prefix.erase(prefix.size() - 1);
	if (n->f != 0) {
	    prefix.push_back('0');
	    print_r(os, prefix, n->f);
	}
	if (prefix.size() >= cur_pos)
	    prefix.erase(cur_pos);
    }
}

ostream & predicate::print(ostream & os) const {
    string s;
    print_r(os, s, root);
    return os;
}

#define HAVE_RDTSC

#ifdef HAVE_RDTSC
    typedef unsigned long long cycles_t;
    
    static cycles_t rdtsc() {
        cycles_t x;
        __asm__ volatile ("rdtsc" : "=A" (x));
        return x;
    }
#endif
    
#ifdef HAVE_MFTB
    typedef unsigned long long cycles_t;

    static inline cycles_t rdtsc(void) {
        cycles_t ret;
    
        __asm__ __volatile__("mftb %0" : "=r" (ret) : );

        return ret;
    }
#endif


int main(int argc, char *argv[]) {
    filter f;
    predicate p;
    bool quiet = false;
    unsigned long tot = 0;
    unsigned long added = 0;
    unsigned long test =0 ;
    cycles_t start, stop;
    cycles_t tot_time=0;
    cycles_t tot_hw=0;
    
    
    if (argc > 1 && strcmp(argv[1], "-q") == 0)
	quiet = true;

    try {
	std::string l;
    bool end=false;
	while(getline(std::cin,l) && test<1000000) {
	    if (l.size() > 0) {
            if(l=="end"){
                end=true;
            }else{                
                char del[] = " ";
                std::string token = strtok( (char*) l.c_str(), del );
                int tree = atoi(token.c_str());
                token = strtok( NULL, del );
                int iff = atoi(token.c_str());
                f = strtok( NULL, del );
                if(!end){
                    //create the tree
                    if(f.count()!=0){
                       ++tot;
                       ++added;	
                       if((added%1000000)==0)
                           cout << "added " << added << endl;
                       p.add_filter(f,tree,iff);
		    } 
                }else{
                    if((test%10000)==0){
                        cout << "avg match time hw " << (test/10000)+6 << " " << (tot_hw/10000) << endl;
                        tot_hw=0;
                    }
                   
                    test++;
                    start = rdtsc();
                    p.findMatch(f,tree);
                    
                    stop = rdtsc();
                    tot_hw+=(stop-start);
                    tot_time+=(stop-start);
                }
            }
        }
	}
        cout << "Memory: " << Mem.size() << endl;
	p.treevisit();
        cout << "Matching time: " << (tot_time/test) << endl;
	    
	if (quiet) {
	    cout << ' ' << added << '/' << tot << endl;
	} else {
	    cout << p;
	}

    } catch (int e) {
	cerr << "bad format." << endl; 
    }
}

