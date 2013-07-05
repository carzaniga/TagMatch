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
    //vector<int> treeIff[8];//these vectors store interface numbers. unsigned short is sufficient.
	unsigned char n[8];
	union v_or_p {
		vector<short> *p;
		unsigned short v[4];
	} treeIff[8];
   
    void addTreeIff(unsigned short tree, unsigned short iff){
		if(n[tree]<3)
			treeIff[tree].v[n[tree]]=iff;
		else {
			if(n[tree]==3){
				// i should copy previous ones here.
				treeIff[tree].p=new (Mem) vector<short>;			
				treeIff[tree].p->push_back(treeIff[tree].v[0]);
				treeIff[tree].p->push_back(treeIff[tree].v[1]);
				treeIff[tree].p->push_back(treeIff[tree].v[2]);
				treeIff[tree].p->push_back(treeIff[tree].v[3]);
			}
			treeIff[tree].p->push_back(iff);
		}
		n[tree]+=1;
    }
	TreeIffPair():n(){};
    
    string print() {
/*
        stringstream out;
        for(int i=0; i<8;++i){
            out << " tree " << i << " : " ;
            for(int j=0; j<treeIff[i].size(); ++j){
                out << " iff " << treeIff[i][j] << " ";
            }
        }
        out<< endl;
        return out.str();
*/
    }
};
static vector<TreeIffPair> ti_vec ;

class node {
public:
    filter::pos_t pos;
    unsigned char treeMask;
    unsigned char set;
    int ti_pos;
    //vector<int> treeIff[8];//these vectors store interface numbers. unsigned short is sufficient.
    node * f; 
    union {
	node * t;
	end_node * ending;
	};
//	static vector<TreeIffPair> ti_vec ;
//    static int ti_index;
    

    //TreeIffPair * ti;
    
	/*void addTreeIff(int tree, int iff){
        treeIff[tree].push_back(iff);
    }*/
    node(filter::pos_t p): pos(p), treeMask(0), set(0), f(0),ti_pos(-1), t(0) {};


    void addIff(unsigned char tree, unsigned char iff) {
		if(ti_pos<0){
			ti_pos=ti_vec.size();//ti_index;
			TreeIffPair *ti = new (Mem) TreeIffPair();
			ti->addTreeIff(tree,iff);
			ti_vec.push_back(*ti);
//			ti_index++;
			//ti_vec.at(ti_pos).addTreeIff(tree,iff);
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
    //TreeIffPair * ti;
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

#if 0
vector< vector<bool> > list;
void end_node::addFilter(const filter & f, filter::pos_t offset){//const vector<bool> & f){
    filter::const_iterator fi = f.begin();
    vector <bool> temp(192-offset-1);		
    for(filter::pos_t i=offset+1;i<192;i++){
	if(*fi==i && fi!=f.end()){
	    temp.push_back(true);
	    fi++;
	}
	else{}
	temp.push_back(false);
    }
    list.push_back(temp);
}
#endif

void end_node::addFilter(const string & bitstring, unsigned char tree, unsigned char iff){
    end_node_entry e(bitstring);

    for(vector<end_node_entry>::iterator i = v.begin(); i != v.end(); ++i)
	if(i->bs == e.bs) {
	    i->addIff(tree,iff);
	    return;
	}
	e.addIff(tree,iff);
    v.push_back(e);
    //v.back().addIff(tree,iff);
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
    last->set=2;
	} else {
	last->set= 1;
	last->addIff(tree,iff);
    }
}


std::set<int> predicate::match_result;


void match(const node *n, filter::const_iterator fi, filter::const_iterator end, int tree) {
    while (fi != end && n != 0 && n->matchTreeMask(tree)) {
        if(n->pos==*fi){
            if (n->set) {
                /*const std::vector<int> & ifs = n->ti->treeIff[tree];
                for(std::vector<int>::const_iterator ii = ifs.begin(); ii != ifs.end(); ++ii) {
                    predicate::match_result.insert(*ii);
                }*/
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

static int count_nodes_r (const node * n) {
    if (n == 0)
	return 0;
	//cout<< n->set <<endl;
    if(n->set==2)
		return 1+1+count_nodes_r(n->f); //we count end_node as one node.
	return 1 + count_nodes_r(n->t) + count_nodes_r(n->f);
}

unsigned long predicate::count_nodes() const {
    return count_nodes_r(root);
}


int count_interfaces_r(const node * n) {
	return 0;    
/*if (n == 0)
	return 0;

    int iff=0;
    if (n->ti != 0) {
        for (int i=0; i<8; i++)
            iff += n->ti->treeIff[i].size();
    }
    return iff + count_interfaces_r(n->t) + count_interfaces_r(n->f);
*/
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
  	//ti_index=0;
	//vector<int> nodeIndex(349040030);
  
    //end_node temp;
	//temp->getzero(10);
	//cout<< temp->v <<endl;
    cout<< DEPTH_THRESHOLD << " Cut size of node:"<<sizeof(node)<<endl;	
    cout<<"size of end_node:"<<sizeof(end_node)<<endl;
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
	cout<<"sleeping for 20sec"<<endl;
	sleep(20);
	while(getline(std::cin,l)) {
	    if (l.size()==0) 
		continue;

	    istringstream is(l);
	    int tree, iff;
	    string fstr;
	    is >> tree >> iff >> fstr;

	    f = fstr;

	    match_timer.start();
	    p.findMatch(f,tree);
	    match_timer.stop();
	    ++count;

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

