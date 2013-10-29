#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
	
#include <vector>
#include <exception>
#include <set>
#include <iostream>
#include <cstring>
#include <sstream>
#include "predicate.h"
#include "allocator.h"
#include "timing.h"
#include <set>

siena::FTAllocator Mem;

#define Verbose
#undef Match_old
static vector<TreeIffPair> ti_vec ;//I shoud initialize the capacity of this vector to speed it upe
static set<unsigned short> match_result;
static int maxSize;
static int leaves;
static int collisions;
static int maxTIF;

class bv192 {
	typedef unsigned long block_t;
	//assuming sizeof(block_t) * CHAR_BIT = 64
	block_t bv[3];
public:
	bv192() { reset(); }
	void set(unsigned int pos);
	bv192(const string &s){
		for(int i=0;i<192;i++)
			if(s[i]=='1')
				set(i);
	}
	bv192 & operator=(const bv192 &rhs);
	bool operator==(const bv192 &rhs);
	bool subset_of(const bv192 & x) const;
	void add(const bv192 & x);
	bool at(unsigned int pos)const;
	unsigned char lastpos();
	void reset() {
#ifdef BV192_USE_MEMSET
		memset(bv,0,sizeof(bv)); 
#else
		bv[0] = 0; bv[1] = 0; bv[2] = 0;
#endif
	}
};

void bv192::add(const bv192 & x) {
	bv[0] |= x.bv[0];
	bv[1] |= x.bv[1];
	bv[2] |= x.bv[2];
}

bv192 & bv192::operator=(const bv192 &rhs){
	bv[0] = rhs.bv[0];
	bv[1] = rhs.bv[1];
	bv[2] = rhs.bv[2];
	return *this;
}

bool bv192::operator==(const bv192 &rhs){
	return (bv[0] == rhs.bv[0] && bv[1] == rhs.bv[1] &&	bv[2] == rhs.bv[2]) ;
}

bool bv192::subset_of(const bv192 & x) const {
	return (bv[0] & x.bv[0]) == bv[0]
		&& (bv[1] & x.bv[1]) == bv[1]
		&& (bv[2] & x.bv[2]) == bv[2];
}

void bv192::set(unsigned int pos) {
	bv[pos >> 6] |= (1U << (pos && 63));
}

bool bv192::at(unsigned int pos) const {
	return bv[pos >> 6] & (1U << (pos && 63)); //if 0 returns false if not zero return ture(?).
}

unsigned char bv192::lastpos() {
	//returns the index of most significat bit.
	if(bv[2]!=0)
		return 128+(63-__builtin_clzl(bv[2]));
	if(bv[1]!=0)
		return 64+(63-__builtin_clzl(bv[1]));
	return (63-__builtin_clzl(bv[0]));
}


class TreeIffPair { 
public:
	unsigned short *tf_array; // the first element stores the size of the array.
	void addTreeIff(unsigned short tree, unsigned short iff){
		unsigned short temp=0;
		temp=tree<<13;
		temp|=iff;
		if(tf_array==0){
			tf_array=new unsigned short[2];
			tf_array[0]=2; //including the first element.
			tf_array[1]=temp;
			return;
		}
		unsigned short size=tf_array[0];
		for(int k=1;k<size;k++)
			if(tf_array[k]==temp)
				return;		
		if(size==65535){
			throw (-1); // means we haved reached the maximum size for our array. 			
		}
		size++;
#ifdef Verbose
		if(maxTIF<size)
			maxTIF=size;
#endif
		unsigned short *tf_array2=new unsigned short[size];

		for(int k=1;k<size-1;k++)
			tf_array2[k]=tf_array[k];
		delete [] tf_array;
		tf_array2[size-1]=temp;
		tf_array2[0]=size;
		tf_array=tf_array2;
	}
	void match(set<unsigned short> & match_result, const unsigned short tree){
		for(int i=1;i<tf_array[0];i++){
			if (tree==tf_array[i]>>13)
				match_result.insert(tf_array[i]&8191); //8191 = 0001111111111111
		}
	}

	TreeIffPair():tf_array(0){};
	string print() {
	}
};

class node {
public:
	filter::pos_t pos;
	unsigned char treeMask;
	unsigned short size;		//stores size of the endings array.
	int ti_pos;
	node * f; 
	union {
		node * t;
		end_node_entry * endings;
	};

	node(filter::pos_t p): pos(p), treeMask(0),f(0),ti_pos(-1), t(0),size(0) {};

	void addIff(unsigned char tree, unsigned short iff) {
		if(ti_pos<0){
			ti_pos=ti_vec.size();
			TreeIffPair *ti = new (Mem) TreeIffPair();
			ti->addTreeIff(tree,iff);
			ti_vec.push_back(*ti);
			return;
		}
		ti_vec.at(ti_pos).addTreeIff(tree,iff);
	}

	bool matchTreeMask(int tree) const{
		unsigned char tmp = 0;
		tmp |= 1 << tree;
		return ((treeMask & tmp)==tmp);
	}

	void setMask (int tree){
		treeMask |= 1 << tree;
	}

	void initMask(unsigned char m){
		treeMask |= m;
	}
};

class end_node_entry {
public:
	// bitset representing the full (Bloom) filter
    bv192 bs;

    // each entry represents a filter and consists of a set of
	// (tree,interface) pairs associated with the filter.  ti_pos is
	// the index into vector ti_vec, which contains a list of
	// (tree,interface) pairs
    int ti_pos;	

	end_node_entry(const string &s): bs(s),ti_pos(-1) {};
	end_node_entry():ti_pos(-1){};

	void addIff(unsigned char tree, unsigned short iff) {
		if(ti_pos<0){
			ti_pos=ti_vec.size();
			TreeIffPair *ti = new (Mem) TreeIffPair();
			ti->addTreeIff(tree,iff);
			ti_vec.push_back(*ti);
			return;
		}
		ti_vec.at(ti_pos).addTreeIff(tree,iff);
	}
};

void predicate::init(){
	for(int i=0;i<192;++i){
		root[i]=new node(i);
	}
}

static const int DEPTH_THRESHOLD = 15;
void predicate::add_filter(const filter & f, unsigned char tree, unsigned short iff, const string & bitstring) {
	int depth=0;
	bv192 bs(bitstring);

	filter::const_iterator fi = f.begin();
	node ** np = &root[*fi];
	node * last; //last node visited
	while (fi != f.end() && depth<DEPTH_THRESHOLD) {
		if (*np == 0) {
			++depth;
			*np = new (Mem) node(*fi);
			last = *np;
			np = &(last->t);
			++fi;
		} else if ((*np)->pos < *fi) {
			last=*np;
			np = &((*np)->f);
		} else if ((*np)->pos > *fi) {
			node * tmp = *np;
			depth++;
			*np = new (Mem) node(*fi);
			last=*np;
			last->initMask(tmp->treeMask);
			last->f = tmp;
			np = &(last->t);
			++fi;
		} else {
			depth++;
			last=*np;
			np = &(last->t);
			++fi;
		}
		last->setMask(tree);
	}
	if(fi!=f.end()){
		if (!last->endings){ 
			if(last->size==0){ // I used default alocator because the size of the object itself is 32 and to avoid dealocation problm later if want to delete stuff from the trie. 
				last->size++;
#ifdef Verbose				
				leaves++;
#endif
				last->endings = new end_node_entry[1]; 
				last->endings[0].addIff(tree,iff);
				last->endings[0].bs=bs; //(bitstring);
				return;	
			}	
		}
		for (int i=0;i<last->size;i++)
			if(last->endings[i].bs==bs){
#ifdef Verbose
				collisions++;
#endif
				last->endings[i].addIff(tree,iff);
				return;
			}
		if(last->size==65535)
			throw (-3);
		last->size++;
		end_node_entry * temp_entry = new end_node_entry[last->size];
		for (int i=0;i<last->size-1;i++){
			temp_entry[i].bs=last->endings[i].bs;
			temp_entry[i].ti_pos=last->endings[i].ti_pos;
		}
		delete [] last->endings ;
		temp_entry[last->size-1].bs=bs;
		temp_entry[last->size-1].addIff(tree,iff);
		last->endings=temp_entry;
		if(maxSize<last->size){
			cout<<endl<<"size is: "<<last->size<<endl;
			maxSize=last->size;
		}
#ifdef Verbose
		//cout<<endl<<"size is: "<<last->size<<endl;
#endif
	} else {
		last->addIff(tree,iff);
	}
}

#ifdef Match_old
void match(const node *n, filter::const_iterator fi, filter::const_iterator end, const int tree,const bv192 & bs)
{
	while (fi != end && n != 0 && n->matchTreeMask(tree)){
		while (n->pos>*fi){
			++fi;
			if(fi==end)
				return;
		}
		if(n->pos==*fi){
			if (n->ti_pos>=0)
				ti_vec.at(n->ti_pos).match(match_result,tree);

			if (n->size>0){ 
				if (n->ti_pos>=0 && n->endings==0){ //this happens when hw of filter is exactly equal to Depth_Threshold. 
					return;
				}
				end_node_entry *en= n->endings; 
				for(int i = 0;i<n->size;++i){
					if(en[i].bs.subset_of(bs))
						ti_vec.at(en[i].ti_pos).match(match_result,tree);
				}
				break;
			}
			++fi;
			match(n->f,fi,end,tree,bs);
			n = n->t;		// equivalent to recursion: match(n->t,fi,end,tree);
		}
		else {//(n->pos < *fi)
			n = n->f;		// equivalent to recursion: match(n->f,fi,end,tree);
		}
	}
}
#else
void match(const node *n, const int tree,const bv192 & bs,const string & bitstring,unsigned char lastonePos)
{
	while ( n != 0 && n->matchTreeMask(tree) && n->pos<=lastonePos){
		if(bitstring[n->pos]=='1'){
			if (n->ti_pos>=0)
				ti_vec.at(n->ti_pos).match(match_result,tree);
			if (n->size>0){ 
				if (n->ti_pos>=0 && n->endings==0){ //this happens when hw of filter is exactly equal to the Depth_Threshold. 
					return;
				}
				end_node_entry *en= n->endings; 
				for(int i = 0;i<n->size;++i){
					if(en[i].bs.subset_of(bs))
						ti_vec.at(en[i].ti_pos).match(match_result,tree);
				}
				break;
			}
			match(n->f,tree,bs,bitstring,lastonePos);
			n=n->t;
		}
		else{
			n = n->f;		// equivalent to recursion: match(n->f,fi,end,tree);
		}
	}
}
#endif


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

//		bool predicate::contains_subset(const filter & f) const {
//			return suffix_contains_subset(false, root, f.begin(), f.end());
//		}
//
void predicate::findMatch(const filter & f, int tree,const string & bitstring) {
	match_result.clear();
	bv192 bs(bitstring);
#ifdef Match_old	
	for(filter::const_iterator fi = f.begin();fi!=f.end();++fi)
		match(root[*fi],fi,f.end(),tree,bs);
#else
	for(filter::const_iterator fi = f.begin();fi!=f.end();++fi)
		match(root[*fi],tree,bs,bitstring,bs.lastpos());
#endif
	if(match_result.size()>0){
		//DO SOMETHING WITH THE MATCH RESULT
	}
}

static int count_nodes_r (const node * n,const int depth) {
	if (n == 0)
		return 0;
	if(depth+1==DEPTH_THRESHOLD)// +1 is for the matching of the current node. 
		return 1+1+count_nodes_r(n->f,depth);//we count end_node as one node. may be I should change this!
	return 1 + count_nodes_r(n->t,depth+1) + count_nodes_r(n->f,depth);
}

unsigned long predicate::count_nodes() const {
	long tot=0;
	for(int i=0;i<192;++i)
		tot+=count_nodes_r(root[i],0);
	return tot;
}


int count_interfaces_r(const node * n) {
	return 0;    
}

unsigned long predicate::count_interfaces() const {
	long tot=0;
	for(int i=0;i<192;++i)
		tot+=count_interfaces_r(root[i]);
	return tot;
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
	////			string s;
	////			print_r(os, s, root);
	return os;
}

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

struct empty_filter : std::exception {
	const char* what() const throw() {return "empty filter!\n";}
};

int main(int argc, char *argv[]) {
#ifdef Verbose
	cout<<"verbose mode"<<endl;
#endif
	maxSize=0;
	leaves=0;
	collisions=0;
	maxTIF=0;
	predicate p;
	p.init();
	cout<<"Depth Threshold: "<< DEPTH_THRESHOLD << " size of node:"<<sizeof(node)<<endl;	
	cout<<"size of end_node_entry:"<<sizeof(end_node_entry)<<endl;
	cout<<"size of TreeIffpair:"<<sizeof(TreeIffPair)<<endl;
	filter f;
	bool quiet = false;
	unsigned long tot = 0;
	unsigned long added = 0;

#ifdef WITH_TIMERS
	cout<<"timer is defined"<<endl;
	Timer match_timer;
	Timer build_timer;
#endif

	if (argc > 1 && strcmp(argv[1], "-q") == 0)
		quiet = true;

	try {
		std::string l;
		bool building;

		while(getline(std::cin,l)) {
			if (l.size()==0) 
				continue;
			if (l=="end")
				break;

			istringstream is(l);
			int tree, iff;
			string fstr;
			is >> tree >> iff >> fstr;

			f = fstr;

			if (f.count()!=0) {
				++tot;
				++added;	
				if((added%1000000)==0)
					cout<<added << " added " << "#leaves: "<<leaves<< " collisions: "<<collisions<<" maxTIff array size: "<<maxTIF<<endl;// '\r';
#ifdef WITH_TIMERS
				build_timer.start();
#endif
				p.add_filter(f,tree,iff,fstr);
#ifdef WITH_TIMERS
				build_timer.stop();
#endif
			} else {
				continue;
				//throw(empty_filter());
			}
		}
		unsigned int sumTreeIff=0;
		for(int i=0;i<ti_vec.size();i++)
			sumTreeIff+=(ti_vec.at(i).tf_array[0]-1);
		if (quiet) {
			cout << ' ' << added << '/' << tot << endl;
		} else {
			cout << p;
		}
#ifdef Verbose
		cout<<"average size of tf_array(in TreeIffPair): "<<sumTreeIff*1.0/ti_vec.size()<<"total sum: "<<sumTreeIff<<endl;
		cout<<"maximum size of array(end_node_entry): "<<maxSize<<endl;
		cout<<"maximum size of Tiff array: "<<maxTIF<<endl;
#endif	
		cout << "Memory (allocated): " << Mem.size() << endl;
		cout << "Memory (requested): " << Mem.requested_size() << endl;
		cout << "Number of nodes: " << p.count_nodes() << endl;
		cout << "Number of iff: " << p.count_interfaces() << endl;
		cout << "Size of Ti_vec: " << ti_vec.size()<< endl;
#ifdef WITH_TIMERS
		cout << "Total building time (us): " << build_timer.read_microseconds() << endl;
#endif

		unsigned long count = 0;
		cout<<"sleeping for 1sec"<<endl;
#if 0 // is this really necessary
		sleep(1);
#endif
		while(getline(std::cin,l)) {
			if (l.size()==0) 
				continue;
			++count;
			istringstream is(l);
			int tree, iff;
			string fstr;
			is >> tree >> iff >> fstr;

			f = fstr;

#ifdef WITH_TIMERS
			match_timer.start();
#endif
			p.findMatch(f,tree,fstr);
#ifdef WITH_TIMERS
			match_timer.stop();
#endif

#ifdef WITH_TIMERS
			if ((count % 10000) == 0) {
				cout << "\rAverage matching cycles (" << count << "): " 
					 << (match_timer.read_microseconds() / count) << flush;
			}
#endif
		}

		if (count > 0) {
			cout << "Total calls: " << count << endl;
#ifdef WITH_TIMERS
			cout << "Average matching time (us): " 
				 << (match_timer.read_microseconds() / count) << endl;
#endif

		}

	} catch (int e) {
		cerr << "bad format. " <<e<< endl; 
	}
}
