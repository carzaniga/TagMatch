// EMACS indentation settings (please, do not delete)
//
// Local Variables:
// c-file-style: "linux"
// indent-tabs-mode: t
// tab-width: 8
// End:
//
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
	
#include <vector>
#include <exception>
#include <set>
#include <iostream>
#include <cstring>
#include <sstream>
#include <bitset>

#include <unistd.h> // necessary for sleep()

#include "predicate.h"
#include "allocator.h"
#include "timing.h"

siena::FTAllocator Mem;

class TreeIffPair { 
public:
	union{
		unsigned short *tf_array;
		unsigned short v[4];
	};	
	unsigned short size; // I can still fit 3 more unsigned short here. because size of the object is 16 byte while only 10 is used right now.
	void addTreeIff(unsigned short tree, unsigned short iff){
		unsigned short temp=0;
		temp=tree<<13;
		temp|=iff;
		if (size<=4){
			for(int k=0;k<size;k++)
				if(v[k]==temp)
					return;
		}
		else{
			for(int k=0;k<size;k++)
				if(tf_array[k]==temp)
					return;
		}
		size++;
		if(size<=4){
			v[size-1]=temp;
			return;
		}
		unsigned short *tf_array2=new unsigned short[size];

		if(size==5){
			for(int k=0;k<size-1;k++)
				tf_array2[k]=v[k];
			tf_array2[size-1]=temp;
			tf_array=tf_array2;
			return;
		}
		for(int k=0;k<size-1;k++)
			tf_array2[k]=tf_array[k];
		delete [] tf_array;
		tf_array2[size-1]=temp;
		tf_array=tf_array2;
	}

	TreeIffPair():size(0){};
	string print() {
	}
};

static vector<TreeIffPair> ti_vec ;//I shoud initialize the capacity of this vector to speed it up.
static vector<int> match_result;

class node {
public:
	filter::pos_t pos;
	unsigned char treeMask;
	int ti_pos;
	node * f; 
	union {
		node * t;
		end_node * ending;
	};

	node(filter::pos_t p): pos(p), treeMask(0),f(0),ti_pos(-1), t(0) {};


	void addIff(unsigned char tree, unsigned char iff) {
		if(ti_pos<0){
			ti_pos=ti_vec.size();
			TreeIffPair *ti = new (Mem) TreeIffPair();
			ti->addTreeIff(tree,iff);
			ti_vec.push_back(*ti);
			return;
		}
		ti_vec.at(ti_pos).addTreeIff(tree,iff);
	}

	bool matchTreeMask(int tree) const {
		unsigned char tmp = 0;
		tmp |= 1 << tree;
		return ((treeMask & tmp)==tmp);
	}

	void setMask (int tree){
		treeMask |= 1 << tree;
	}
};

class end_node_entry {
public:
	bitset<192> bs;
	int ti_pos;
	end_node_entry(const string &s): bs(s),ti_pos(-1) {};

	void addIff(unsigned char tree, unsigned char iff) {
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

class end_node {
public:
	vector<end_node_entry> v;
	void addFilter(const string & bitstring, unsigned char tree, unsigned char iff);
};

void end_node::addFilter(const string & bitstring, unsigned char tree, unsigned char iff){
	end_node_entry e(bitstring);

	for(vector<end_node_entry>::iterator i = v.begin(); i != v.end(); ++i)
		if(i->bs == e.bs) {
			i->addIff(tree,iff);
			return;
		}
	e.addIff(tree,iff);
	v.push_back(e);
}

static const int DEPTH_THRESHOLD = 15;
void predicate::add_filter(const filter & f, unsigned char tree, unsigned char iff, const string & bitstring) {
	int depth=0;
	node ** np = &root;
	filter::const_iterator fi = f.begin();

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
		if (!last->ending) 
			last->ending = new (Mem) end_node();
		last->ending->addFilter(bitstring,tree,iff);
	} else {
		last->addIff(tree,iff);
	}
}

void match(const node *n, filter::const_iterator fi, filter::const_iterator end, int tree,int depth,const bitset<192> & bs)
{
	while (fi != end && n != 0){// && n->matchTreeMask(tree)) {// I should re-incorporate treeMask check.
		if(n->pos==*fi){
			if (n->ti_pos>=0){
				match_result.push_back(n->ti_pos);
			}
			if (depth+1==DEPTH_THRESHOLD) {
				if (n->ti_pos<0 && n->ending==0)
					cout<<endl<<"errrrrrrror in the code"<<endl;
				if (n->ti_pos>=0 && n->ending==0){ //this happens when hw of filter is exactly 15. 
					return;
				}
				end_node *en= n->ending; 
				bitset<192> temp;
				for(vector<end_node_entry>::iterator i = en->v.begin(); i != en->v.end(); ++i){
					temp=bs;
					temp&=i->bs;
					if(temp==bs){
						cout<<endl<<"."<<endl;
						match_result.push_back(i->ti_pos);
					}
				}
				break;
			}
			++fi;
			match(n->f,fi,end,tree,depth,bs);
			depth++;
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

void predicate::findMatch(const filter & f, int tree,const string & bitstring) {
	match_result.clear();
	bitset<192> bs(bitstring);
	match(root,f.begin(),f.end(),tree,0,bs);
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
	return count_nodes_r(root,0);
}


int count_interfaces_r(const node * n) {
	return 0;    
}

unsigned long predicate::count_interfaces() const {
	return count_interfaces_r(root);
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
	cout<< DEPTH_THRESHOLD << " Cut size of node:"<<sizeof(node)<<endl;	
	cout<<"size of end_node:"<<sizeof(end_node)<<endl;
	cout<<"size of TreeIffpair:"<<sizeof(TreeIffPair)<<endl;
	filter f;
	predicate p;
	bool quiet = false;
	unsigned long tot = 0;
	unsigned long added = 0;

	Timer match_timer;
	Timer build_timer;

	Timer::calibrate();		// this will cost us about 0.5 seconds

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
					cout << " added " << added <<endl;// '\r';
				build_timer.start();
				p.add_filter(f,tree,iff,fstr);
				build_timer.stop();
			} else {
				continue;
				//throw(empty_filter());
			}
		}

		if (quiet) {
			cout << ' ' << added << '/' << tot << endl;
		} else {
			cout << p;
		}

		cout << "Memory (allocated): " << Mem.size() << endl;
		cout << "Memory (requested): " << Mem.requested_size() << endl;
		cout << "Number of nodes: " << p.count_nodes() << std::endl;
		cout << "Number of iff: " << p.count_interfaces() << std::endl;
		cout << "Total building time (us): " << build_timer.read_microseconds() << endl;

		unsigned long count = 0;
		cout<<"sleeping for 1sec"<<endl;
		sleep(1);
		while(getline(std::cin,l)) {
			if (l.size()==0) 
				continue;
			++count;
			istringstream is(l);
			int tree, iff;
			string fstr;
			is >> tree >> iff >> fstr;

			f = fstr;

			match_timer.start();
			p.findMatch(f,tree,fstr);
			match_timer.stop();

			if ((count % 10000) == 0) {
				cout << "\rAverage matching cycles (" << count << "): " 
				     << (match_timer.read_microseconds() / count) << flush;
			}
		}

		if (count > 0) {
			cout << "Total calls: " << count << endl;
			cout << "Average matching time (us): " 
			     << (match_timer.read_microseconds() / count) << endl;
		}

	} catch (int e) {
		cerr << "bad format." << endl; 
	}
}


