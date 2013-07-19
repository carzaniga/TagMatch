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
#include "predicate.h"
#include "allocator.h"
#include "timing.h"

siena::FTAllocator Mem;

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
			unsigned short *tf_array2=new unsigned short[size];

			for(int k=1;k<size-1;k++)
				tf_array2[k]=tf_array[k];
			delete [] tf_array;
			tf_array2[size-1]=temp;
			tf_array2[0]=size;
			tf_array=tf_array2;
		}
		vector<unsigned short> match(const unsigned short tree){
			vector<unsigned short> res;
			for(int i=1;i<tf_array[0];i++){
				if (tree==tf_array[i]>>13)
					res.push_back(tf_array[i]&8191); //8191 = 0001111111111111
			}
			return res;
		}

		TreeIffPair():tf_array(0){};
		string print() {
		}
};
static vector<TreeIffPair> ti_vec ;//I shoud initialize the capacity of this vector to speed it up.
static vector<unsigned short> match_result;

class node {
	public:
		filter::pos_t pos;
		unsigned char treeMask;
		unsigned short size;//stores size of the endings array.
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
		bitset<192> bs;
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
	bitset<192> bs(bitstring);
	
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
			if(last->size==0){ // I used default alocator because the size of the object itself is 32. 
				last->size++;
				last->endings = new end_node_entry[1]; //its a dynamic array. first_element.size stores the size of the array.
				last->endings[0].addIff(tree,iff);
				last->endings[0].bs=bs; //(bitstring);
				return;	
			}	
		}
		for (int i=0;i<last->size;i++)
			if(last->endings[i].bs==bs){
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
	} else {
		last->addIff(tree,iff);
	}
}

void match(const node *n, filter::const_iterator fi, filter::const_iterator end, int tree,int depth,const bitset<192> & bs)
{
	vector<unsigned short> tempVec;
	while (fi != end && n != 0 && n->matchTreeMask(tree)){
		if(n->pos==*fi){
			if (n->ti_pos>=0){
				tempVec=ti_vec.at(n->ti_pos).match(tree);
				match_result.insert(match_result.end(),tempVec.begin(),tempVec.end()); 
				tempVec.clear();
			}
			if (depth+1==DEPTH_THRESHOLD) {
				if (n->ti_pos<0 && n->endings==0)
					throw (-2);
				if (n->ti_pos>=0 && n->endings==0){ //this happens when hw of filter is exactly 15. 
					return;
				}
				end_node_entry *en= n->endings; 
				bitset<192> temp;
				for(int i = 0;i<n->size;++i){
					temp=bs;
					temp&=en[i].bs;
					if(temp==en[i].bs){
						tempVec=ti_vec.at(en[i].ti_pos).match(tree);
						match_result.insert(match_result.end(),tempVec.begin(),tempVec.end()); 
						tempVec.clear();
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

//		bool predicate::contains_subset(const filter & f) const {
//			return suffix_contains_subset(false, root, f.begin(), f.end());
//		}
//
		void predicate::findMatch(const filter & f, int tree,const string & bitstring) {
			match_result.clear();
			bitset<192> bs(bitstring);
			for(filter::const_iterator fi = f.begin();fi!=f.end();++fi)
				match(root[*fi],fi,f.end(),tree,0,bs);
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
			predicate p;
			p.init();
			cout<< DEPTH_THRESHOLD << " Cut size of node:"<<sizeof(node)<<endl;	
			cout<<"size of end_node_entry:"<<sizeof(end_node_entry)<<endl;
			cout<<"size of TreeIffpair:"<<sizeof(TreeIffPair)<<endl;
			filter f;
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
				cout << "Number of nodes: " << p.count_nodes() << endl;
				cout << "Number of iff: " << p.count_interfaces() << endl;
				cout << "Size of Ti_vec: " << ti_vec.size()<< endl;
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
				cerr << "bad format. " <<e<< endl; 
			}
		}

